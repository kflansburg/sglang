"""Unit tests for the joint schema head of Clef checkpoints: decision layouts, and
the batched forward against a per-request loop over the stock torch modules."""

import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

from sglang.srt.layers import joint_schema_head
from sglang.srt.layers.joint_schema_head import (
    JointSchemaHead,
    JointSchemaPooler,
    LayoutQuestion,
    pack_decision_layout,
    parse_decision_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _questions(offset=0):
    return [
        LayoutQuestion(
            1,
            (10 + offset, 14 + offset),
            (
                (16 + offset, 17 + offset),
                (17 + offset, 19 + offset),
                (19 + offset, 21 + offset),
            ),
        ),
        LayoutQuestion(
            0,
            (22 + offset, 25 + offset),
            ((26 + offset, 28 + offset), (29 + offset, 31 + offset)),
        ),
    ]


class TestDecisionLayout(CustomTestCase):
    def test_rejects_layouts_that_do_not_fit_the_prompt(self):
        layout = pack_decision_layout(40, _questions())
        invalid = {
            "shorter prompt": (layout, 39),
            "not ints": ([float(value) for value in layout], 40),
            "truncated": (layout[:-1], 40),
            "trailing values": (layout + [0], 40),
            "no questions": ([40, 0], 40),
            "empty span": (
                pack_decision_layout(40, [LayoutQuestion(0, (5, 5), ((6, 7),))]),
                40,
            ),
            "past the prompt": (
                pack_decision_layout(40, [LayoutQuestion(0, (5, 6), ((6, 41),))]),
                40,
            ),
            "unknown type": (
                pack_decision_layout(40, [LayoutQuestion(3, (5, 6), ((6, 7),))]),
                40,
            ),
        }
        for name, (values, num_tokens) in invalid.items():
            with self.subTest(name), self.assertRaises(ValueError):
                parse_decision_layout(values, num_tokens)


class _RecordingHead:
    """Stands in for the head, scoring each option with its request's length."""

    def __init__(self):
        self.calls = []

    def forward_batch(
        self, hidden, input_ids, lengths, layouts, embedding_weight, prefix_memory
    ):
        self.calls.append(lengths)
        return [
            torch.full((sum(len(q.option_spans) for q in layout),), float(length))
            for length, layout in zip(lengths, layouts)
        ]


class TestJointSchemaPooler(CustomTestCase):
    def test_a_layout_cut_from_its_prompt_is_skipped_and_neighbors_are_scored(self):
        """A prompt shorter than its layout, as after a truncation, must not stop the
        scheduler or the other requests of its batch."""
        head = _RecordingHead()
        pooler = JointSchemaPooler(head, lambda: torch.zeros(1))
        layout = pack_decision_layout(40, _questions())
        lengths = [40, 35, 40]
        forward_batch = SimpleNamespace(
            decision_layouts=[layout, layout, layout],
            extend_seq_lens_cpu=lengths,
            extend_prefix_lens_cpu=[0, 0, 0],
            decision_prefix_memory=None,
            return_decision_memory=None,
            batch_size=3,
            input_ids=torch.zeros(sum(lengths), dtype=torch.long),
        )
        output = pooler(torch.zeros(sum(lengths), 8), forward_batch)
        self.assertEqual([len(scores) for scores in output.embeddings], [5, 0, 5])
        self.assertEqual(head.calls, [[40], [40]])


def _reference_logits(head, hidden, input_ids, questions, lm_head):
    """One request through the modules' own forward methods, as the checkpoint's loop does."""
    normalized = head.hidden_norm(hidden)
    memory = head.memory_projection(normalized).unsqueeze(0)
    global_vector = normalized[-1]

    def mean(values, span):
        return values[span[0] : span[1]].mean(dim=0)

    question_vectors = torch.stack(
        [mean(normalized, q.question_span) for q in questions]
    )
    queries, lexicals = [], []
    for index, question in enumerate(questions):
        context = torch.stack(
            [mean(normalized, span) for span in question.option_spans]
        )
        lexical = torch.stack(
            [
                lm_head[input_ids[start:end]].mean(dim=0)
                for start, end in question.option_spans
            ]
        )
        queries.append(
            head.option_context_projection(context)
            + head.option_lexical_projection(lexical)
            + head.option_question_projection(question_vectors[index]).unsqueeze(0)
        )
        lexicals.append(lexical)
    routed = torch.cat(queries).unsqueeze(0)
    for layer in head.evidence_layers:
        attended, _ = layer.attention(
            layer.query_norm(routed),
            layer.memory_norm(memory),
            layer.memory_norm(memory),
            need_weights=False,
        )
        routed = routed + attended
        routed = routed + layer.feedforward(layer.feedforward_norm(routed))
    options = list(torch.split(routed[0], [len(q.option_spans) for q in questions]))
    base_fields = head.question_projection(question_vectors)
    summaries = []
    for field, routed_options in zip(base_fields, options):
        weights = torch.softmax(
            routed_options @ field / math.sqrt(field.shape[-1]), dim=0
        )
        summaries.append((weights.unsqueeze(-1) * routed_options).sum(dim=0))
    fields = (
        base_fields
        + head.option_summary_norm(torch.stack(summaries))
        + head.global_projection(global_vector).unsqueeze(0)
        + head.type_embedding(
            torch.tensor([q.question_type for q in questions], device=hidden.device)
        )
    ).unsqueeze(0)
    for layer in head.layers:
        fields = layer(fields, memory)
    fields = head.field_norm(fields[0])
    prior_scale = head.prior_logit_scale.clamp(max=math.log(100.0)).exp()
    joint_scale = head.joint_logit_scale.clamp(max=math.log(100.0)).exp()
    logits = []
    for index, (field, lexical, routed_options) in enumerate(
        zip(fields, lexicals, options)
    ):
        anchor = F.normalize(question_vectors[index] + global_vector, dim=-1)
        prior = prior_scale * (F.normalize(lexical, dim=-1) @ anchor)
        normed = head.option_norm(routed_options)
        repeated = field.unsqueeze(0).expand_as(normed)
        features = torch.cat(
            [repeated, normed, repeated * normed, torch.abs(repeated - normed)], dim=-1
        )
        residual = head.residual_scorer(features).squeeze(-1)
        cosine = F.cosine_similarity(repeated, normed, dim=-1)
        logits.append(
            prior
            + torch.sigmoid(head.residual_gate) * (joint_scale * cosine + residual)
        )
    return torch.cat(logits).float()


def _packed_attention(module, queries, keys, query_bounds, key_bounds):
    """The head's packed attention, one request at a time with the stock module."""
    keys = queries if keys is None else keys
    query_cu, key_cu = query_bounds[0].tolist(), key_bounds[0].tolist()
    outputs = []
    for request in range(len(query_cu) - 1):
        query = queries[query_cu[request] : query_cu[request + 1]].unsqueeze(0)
        key = keys[key_cu[request] : key_cu[request + 1]].unsqueeze(0)
        outputs.append(module(query, key, key, need_weights=False)[0][0])
    return torch.cat(outputs)


def _split(hidden, input_ids, lengths, splits):
    """Each request's tokens after its first ``splits[i]``, back to back."""
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    keep = torch.cat(
        [
            torch.arange(offsets[i] + splits[i], offsets[i + 1])
            for i in range(len(lengths))
        ]
    )
    prefixes = [
        hidden[offsets[i] : offsets[i] + splits[i]] for i in range(len(lengths))
    ]
    return (
        hidden[keep],
        input_ids[keep],
        [a - b for a, b in zip(lengths, splits)],
        prefixes,
    )


@mock.patch.object(joint_schema_head, "_attention", _packed_attention)
class TestHeadPrefixMemory(CustomTestCase):
    """Scoring a request from cached prefix memory and the rest of its tokens must
    equal scoring its whole prompt."""

    def setUp(self):
        torch.manual_seed(0)
        hidden_size, vocab = 64, 128
        head = JointSchemaHead(
            hidden_size, width=32, routing_layers=2, layers=2, heads=4, feedforward=64
        )
        for parameter in (
            head.prior_logit_scale,
            head.joint_logit_scale,
            head.residual_gate,
        ):
            parameter.data.fill_(0.5)
        self.head = head.double().eval()
        self.lm_head = torch.randn(vocab, hidden_size, dtype=torch.float64)
        self.lengths = [40, 140, 64]
        self.layouts = [
            _questions(),
            _questions(offset=100),
            [
                LayoutQuestion(
                    2, (3, 9), tuple((10 + 2 * i, 12 + 2 * i) for i in range(20))
                )
            ],
        ]
        self.hidden = torch.randn(sum(self.lengths), hidden_size, dtype=torch.float64)
        self.input_ids = torch.randint(0, vocab, (sum(self.lengths),))

    def _full(self):
        return self.head.forward_batch(
            self.hidden, self.input_ids, self.lengths, self.layouts, self.lm_head
        )

    def test_whole_prompt_matches_the_reference_loop(self):
        start = 0
        for length, questions, logits in zip(self.lengths, self.layouts, self._full()):
            expected = _reference_logits(
                self.head,
                self.hidden[start : start + length],
                self.input_ids[start : start + length],
                questions,
                self.lm_head,
            )
            torch.testing.assert_close(logits, expected)
            start += length

    def test_prefix_memory_matches_the_whole_prompt(self):
        full = self._full()
        cases = {
            "no request cached": [0, 0, 0],
            "one request cached": [0, 90, 0],
            "every request cached": [10, 100, 3],
            "cached up to the first span": [10, 110, 3],
            "one token cached": [1, 1, 1],
        }
        for name, splits in cases.items():
            with self.subTest(name):
                hidden, input_ids, lengths, prefixes = _split(
                    self.hidden, self.input_ids, self.lengths, splits
                )
                prefix_memory = [
                    self.head.project_memory(prefix) if len(prefix) else None
                    for prefix in prefixes
                ]
                split = self.head.forward_batch(
                    hidden,
                    input_ids,
                    lengths,
                    self.layouts,
                    self.lm_head,
                    prefix_memory=prefix_memory,
                )
                for expected, logits in zip(full, split):
                    torch.testing.assert_close(logits, expected)

    def test_prefix_memory_splits_into_groups(self):
        splits = [10, 100, 3]
        hidden, input_ids, lengths, prefixes = _split(
            self.hidden, self.input_ids, self.lengths, splits
        )
        prefix_memory = [self.head.project_memory(prefix) for prefix in prefixes]
        with mock.patch.object(joint_schema_head, "_MAX_GROUP_TOKENS", 150):
            split = self.head.forward_batch(
                hidden,
                input_ids,
                lengths,
                self.layouts,
                self.lm_head,
                prefix_memory=prefix_memory,
            )
        for expected, logits in zip(self._full(), split):
            torch.testing.assert_close(logits, expected)

    def test_a_span_inside_the_prefix_is_refused(self):
        hidden, input_ids, lengths, prefixes = _split(
            self.hidden, self.input_ids, self.lengths, [11, 0, 0]
        )
        with self.assertRaisesRegex(ValueError, "inside the cached prefix"):
            self.head.forward_batch(
                hidden,
                input_ids,
                lengths,
                self.layouts,
                self.lm_head,
                prefix_memory=[self.head.project_memory(prefixes[0]), None, None],
            )


@unittest.skipUnless(torch.cuda.is_available(), "the batched head uses FlashAttention")
class TestJointSchemaHeadBatch(CustomTestCase):
    def test_batched_forward_matches_each_request_alone(self):
        torch.manual_seed(0)
        hidden_size, vocab = 256, 512
        head = JointSchemaHead(
            hidden_size, width=128, routing_layers=2, layers=2, heads=4, feedforward=256
        )
        for parameter in (
            head.prior_logit_scale,
            head.joint_logit_scale,
            head.residual_gate,
        ):
            parameter.data.fill_(0.5)
        head = head.to("cuda", torch.bfloat16).eval()
        lm_head = torch.randn(vocab, hidden_size, device="cuda", dtype=torch.bfloat16)
        lengths = [40, 140, 64]
        layouts = [
            _questions(),
            _questions(offset=100),
            [
                LayoutQuestion(
                    2, (3, 9), tuple((10 + 2 * i, 12 + 2 * i) for i in range(20))
                )
            ],
        ]
        hidden = torch.randn(
            sum(lengths), hidden_size, device="cuda", dtype=torch.bfloat16
        )
        input_ids = torch.randint(0, vocab, (sum(lengths),), device="cuda")
        with torch.no_grad():
            batched = head.forward_batch(hidden, input_ids, lengths, layouts, lm_head)
            start = 0
            for length, questions, logits in zip(lengths, layouts, batched):
                end = start + length
                expected = _reference_logits(
                    head, hidden[start:end], input_ids[start:end], questions, lm_head
                )
                self.assertEqual(logits.shape, expected.shape)
                torch.testing.assert_close(logits, expected, atol=5e-2, rtol=5e-2)
                torch.testing.assert_close(
                    logits.softmax(-1), expected.softmax(-1), atol=1e-2, rtol=0
                )
                start = end
            splits = [10, 100, 3]
            split_hidden, split_ids, split_lengths, prefixes = _split(
                hidden, input_ids, lengths, splits
            )
            split = head.forward_batch(
                split_hidden,
                split_ids,
                split_lengths,
                layouts,
                lm_head,
                prefix_memory=[head.project_memory(prefix) for prefix in prefixes],
            )
            for expected, logits in zip(batched, split):
                torch.testing.assert_close(logits, expected, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
