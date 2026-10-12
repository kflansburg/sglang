"""Unit tests for the joint schema head memory of cached prompt prefixes: the
store, the cap it puts on a decision request's radix match, and the pooler
scoring a request from cached prefix memory."""

import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers import joint_schema_head
from sglang.srt.layers.joint_schema_head import (
    JointSchemaHead,
    JointSchemaPooler,
    LayoutQuestion,
    pack_decision_layout,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.decision_prefix_memory import (
    DecisionPrefixMemory,
    cache_salt,
    decision_prefix_limit,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SALT = cache_salt(None, None)


def _rows(n, width=4, fill=None):
    rows = torch.arange(n * width, dtype=torch.float32).reshape(n, width)
    return rows if fill is None else torch.full((n, width), float(fill))


def _layout(prompt_length, first_span):
    return pack_decision_layout(
        prompt_length,
        [
            LayoutQuestion(
                0,
                (first_span, first_span + 2),
                ((first_span + 3, first_span + 4), (first_span + 5, first_span + 6)),
            )
        ],
    )


class TestDecisionPrefixMemory(CustomTestCase):
    def test_longest_held_prefix_on_the_grid(self):
        memory = DecisionPrefixMemory(1 << 20, grid=4)
        tokens = list(range(100, 118))
        memory.store(SALT, tokens, _rows(18))
        length, rows = memory.longest_prefix(SALT, tokens + [1, 2], limit=20)
        self.assertEqual(length, 16)
        torch.testing.assert_close(rows, _rows(18)[:16])
        self.assertEqual(memory.longest_prefix(SALT, tokens, limit=11)[0], 8)
        diverged = tokens[:9] + [0] * 9
        self.assertEqual(memory.longest_prefix(SALT, diverged, limit=18)[0], 8)
        self.assertEqual(memory.longest_prefix(SALT, tokens, limit=3), (0, None))

    def test_a_different_cache_key_shares_nothing(self):
        memory = DecisionPrefixMemory(1 << 20, grid=4)
        tokens = list(range(8))
        memory.store(cache_salt("lora", None), tokens, _rows(8))
        self.assertEqual(memory.longest_prefix(SALT, tokens, limit=8), (0, None))
        self.assertEqual(
            memory.longest_prefix(cache_salt(None, "salt"), tokens, limit=8)[0], 0
        )

    def test_rows_already_held_are_not_stored_again(self):
        memory = DecisionPrefixMemory(1 << 20, grid=4)
        tokens = list(range(8))
        memory.store(SALT, tokens, _rows(8, fill=1))
        used = memory.used_bytes
        memory.store(SALT, tokens, _rows(8, fill=2))
        self.assertEqual(memory.used_bytes, used)
        _, rows = memory.longest_prefix(SALT, tokens, limit=8)
        self.assertEqual(rows[0, 0].item(), 1.0)

    def test_least_recently_used_rows_are_dropped_to_fit(self):
        size = _rows(8).numel() * 4
        memory = DecisionPrefixMemory(2 * size, grid=4)
        first, second, third = ([i] * 8 for i in (1, 2, 3))
        memory.store(SALT, first, _rows(8))
        memory.store(SALT, second, _rows(8))
        memory.longest_prefix(SALT, first, limit=8)
        memory.store(SALT, third, _rows(8))
        self.assertEqual(memory.used_bytes, 2 * size)
        self.assertEqual(memory.longest_prefix(SALT, first, limit=8)[0], 8)
        self.assertEqual(memory.longest_prefix(SALT, second, limit=8)[0], 0)
        self.assertEqual(memory.longest_prefix(SALT, third, limit=8)[0], 8)
        memory.store(SALT, [4] * 40, _rows(40))
        self.assertEqual(memory.used_bytes, 2 * size)
        memory.clear()
        self.assertEqual(memory.longest_prefix(SALT, first, limit=8), (0, None))
        self.assertEqual(memory.used_bytes, 0)


class TestDecisionPrefixLimit(CustomTestCase):
    def test_the_match_ends_before_the_first_span(self):
        memory = DecisionPrefixMemory(1 << 20, grid=4)
        tokens = list(range(40))
        memory.store(SALT, tokens, _rows(40))
        self.assertEqual(
            decision_prefix_limit(_layout(40, 18), tokens, 39, SALT, memory)[0], 16
        )
        self.assertEqual(
            decision_prefix_limit(_layout(40, 30), tokens, 22, SALT, memory)[0], 20
        )

    def test_expanded_images_move_the_first_span(self):
        memory = DecisionPrefixMemory(1 << 20, grid=4)
        tokens = list(range(50))
        memory.store(SALT, tokens, _rows(50))
        # The layout indexes a 40-token prompt whose image expanded by 10.
        self.assertEqual(
            decision_prefix_limit(_layout(40, 18), tokens, 49, SALT, memory)[0], 28
        )

    def test_no_memory_means_no_reuse(self):
        self.assertEqual(
            decision_prefix_limit(_layout(40, 18), list(range(40)), 39, SALT, None),
            (0, None),
        )


class TestDecisionRequestMatch(CustomTestCase):
    def _match(self, memory):
        with published_topology():
            cache = RadixCache.create_simulated()
            cache.insert(InsertParams(key=RadixKey(token_ids=array("q", range(30)))))
            req = Req(
                "decision",
                None,
                array("q", list(range(24)) + [99] * 16),
                SamplingParams(max_new_tokens=1),
                decision_layout=_layout(40, 26),
            )
            req.init_next_round_input(cache, decision_prefix_memory=memory)
            return req

    def test_a_decision_request_reuses_only_prefixes_with_memory(self):
        self.assertEqual(self._match(None).prefix_len, 0)
        memory = DecisionPrefixMemory(1 << 20, grid=4)
        memory.store(SALT, list(range(30)), _rows(30))
        req = self._match(memory)
        self.assertEqual(req.prefix_len, 24)
        torch.testing.assert_close(req.decision_prefix_memory, _rows(30)[:24])
        self.assertFalse(req.return_decision_memory)

    def test_an_uncached_decision_request_returns_its_memory(self):
        req = self._match(DecisionPrefixMemory(1 << 20, grid=4))
        self.assertEqual(req.prefix_len, 0)
        self.assertIsNone(req.decision_prefix_memory)
        self.assertTrue(req.return_decision_memory)


def _packed_attention(module, queries, keys, query_bounds, key_bounds):
    keys = queries if keys is None else keys
    query_cu, key_cu = query_bounds[0].tolist(), key_bounds[0].tolist()
    outputs = []
    for request in range(len(query_cu) - 1):
        query = queries[query_cu[request] : query_cu[request + 1]].unsqueeze(0)
        key = keys[key_cu[request] : key_cu[request + 1]].unsqueeze(0)
        outputs.append(module(query, key, key, need_weights=False)[0][0])
    return torch.cat(outputs)


@mock.patch.object(joint_schema_head, "_attention", _packed_attention)
class TestPoolerPrefixMemory(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        hidden_size, vocab = 32, 64
        self.head = JointSchemaHead(
            hidden_size, width=16, routing_layers=1, layers=1, heads=2, feedforward=32
        ).double()
        self.lm_head = torch.randn(vocab, hidden_size, dtype=torch.float64)
        self.pooler = JointSchemaPooler(self.head, lambda: self.lm_head)
        self.layout = _layout(40, 26)
        self.hidden = torch.randn(40, hidden_size, dtype=torch.float64)
        self.input_ids = torch.randint(0, vocab, (40,))

    def _pool(self, prefix, memory, return_memory):
        return self.pooler(
            self.hidden[prefix:],
            SimpleNamespace(
                decision_layouts=[self.layout],
                extend_seq_lens_cpu=[40 - prefix],
                extend_prefix_lens_cpu=[prefix],
                decision_prefix_memory=[memory],
                return_decision_memory=[return_memory],
                batch_size=1,
                input_ids=self.input_ids[prefix:],
            ),
        )

    def test_cached_prefix_memory_scores_like_the_whole_prompt(self):
        full = self._pool(0, None, True)
        (rows,) = full.decision_memory
        torch.testing.assert_close(rows, self.head.project_memory(self.hidden[:26]))
        cached = self._pool(24, rows[:24], False)
        self.assertIsNone(cached.decision_memory)
        torch.testing.assert_close(cached.embeddings[0], full.embeddings[0])

    def test_a_cached_prefix_without_memory_is_not_scored(self):
        self.assertEqual(len(self._pool(24, None, False).embeddings[0]), 0)
        rows = self.head.project_memory(self.hidden[:20])
        self.assertEqual(len(self._pool(24, rows, False).embeddings[0]), 0)


if __name__ == "__main__":
    unittest.main()
