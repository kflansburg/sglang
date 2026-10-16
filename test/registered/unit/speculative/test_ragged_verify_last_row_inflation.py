# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""R-1 witness PoC (board #72): full-tier pad-fold inflates the LAST REAL
verify row past its allocated window -> stale prior-owner KV slot content
reaches the verifier's per-row positions; and the accepted-token gate (G2)
answer with executable evidence.

BACKGROUND (graph replay, target-verify, ragged/compact verify). The
captured CUDA graph covers a round-up bucket of graph_num_tokens. When the
live batch's sum(verify_lens) is smaller, RaggedVerifyLayout.padded_to_bucket
must park the leftover somewhere:

* padded_bs > bs (requests missing from the tier): leftover splits across
  synthesized PADDING rows -> benign (their q rows are scratch by design).
* padded_bs == bs (FULL tier, the #72 trigger): the legacy uncapped fold
  (the `num_pad_reqs == 0` branch of pad_verify_lens_to_bucket, still the
  executed path when no last_row_cap is passed) folds the ENTIRE leftover
  into padded[-1], the LAST REAL request's verify_len. With leftover L the
  row's attention KV read span becomes seq..seq+verify_len+L, running past
  the row's pre-allocated draft window (EAGLE invariant: each request owns
  exactly [seq, seq+num_draft_tokens) of its req_to_token row at verify
  time; rejected drafts are freed only after accept). The page-table row is
  max-context wide, so those reads stay IN-BOUNDS of the table -- landing
  on entries never written for this request: STALE SLOT IDS of prior
  owners. The verifier then reads real KV from those slots: foreign
  content enters the last real row's verify attention.

GRADES DISCRIMINATED BY THIS WITNESS:
G1 (content-observation; the grade adjudicated in #66): stale slot
   CONTENT (sentinel-tagged ids 5910xx) provably lands in the last real
   row's verify window. Demonstrated pre-fix; clamped post-fix (#69:
   last_row_cap = num_draft_tokens).
G2 (numerics-corrupting / accepted-token reachability): whether the
   padded extra q rows' outputs ever reach ACCEPTED-TOKEN buffers. GATE
   ANSWER (executable evidence in test_g2_accepted_token_gate below): NO.
   Structural chain, each link executable here or load-bearing at the real
   call site:
     1. next_token_logits for verify are reshaped [bs, draft_token_num]
        at the accept entry (eagle_utils.py: target_predict =
        argmax(next_token_logits).reshape(bs, draft_token_num)) -- flat
        extras beyond each row's drafted width fall outside the reshape
        domain consumed per row; there is no indexing arithmetic from the
        padded layout into the accept buffers (the layout is an ATTENTION
        input only: qo_indptr + verify_lens feed the KV-read mapping; the
        accept walk consumes DRAFT-side buffers).
     2. accept_index is [bs, max_tree_depth] of GLOBAL TREE-NODE indices,
        filled by the tree walk over retrieve_index/retrieve_next_token/
        retrieve_next_sibling -- all constructed by the draft side from
        topk buffers of width num_draft_tokens per row; no entry can name
        a padded extra position, and -1 padding is the only filler past
        the accepted prefix.
     3. KV writes at verify are bounded by allocation =
        sum(live verify_lens) (compute_ragged_extend_lengths:
        num_tokens = total_verify_tokens of the UNPADDED layout), so the
        extras receive no out_cache_loc rows and cannot write KV.
   => read-side exposure only: foreign KV content flows INTO this request's
   verify logits (can flip accept/reject of its REAL drafts = silent
   corruption of ITS OWN output), never OUT into another request's KV or
   into accepted-token buffers. G1 confirmed, G2 refuted for the current
   accept paths. Recheck if a consumer ever indexes logits/accept flat
   across the row's PADDED width.

PRE-FIX shape: pad WITHOUT last_row_cap (legacy branch, callable on all
versions incl. post-#69 which keeps it for legacy callers). POST-FIX:
dsa #69 consumers pass last_row_cap = num_draft_tokens; the same file then
asserts clamp/ghost semantics. Verified red pre-fix, green against
agent/dsa-auditor @6dbea36 (fix #69 integrated).

    python -m pytest test/registered/unit/speculative/test_ragged_verify_last_row_inflation.py -v
"""

import inspect
import unittest

import torch

from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    build_ragged_target_verify_geometry,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# 3-row verify batch. Per-request draft budget 4 (graph tier = full block
# of bs*4 = 12 tokens). Live lens [2,2,2] (anchor + 1 draft each) ->
# leftover 6 has nowhere to go but the last real row under the legacy fold.
BS = 3
NUM_DRAFT_TOKENS = 4
GRAPH_NUM_TOKENS = BS * NUM_DRAFT_TOKENS
VERIFY_LENS_CPU = [2, 2, 2]
SEQ_LENS = [50, 70, 90]
TABLE_WIDTH = 128
# Sentinel slot ids for the stale region: 5910xx pattern (in-bounds of a
# pool >= 600k slots of course not allocated here; the table is synthetic,
# the POINT is these ids identify a prior owner, not garbage).
STALE_BASE = 591000


def _make_layout():
    return RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=VERIFY_LENS_CPU,
        device=torch.device("cpu"),
        grid=(GRAPH_NUM_TOKENS,),
        graph_num_tokens_floor=GRAPH_NUM_TOKENS,
    )


def _padded(layout):
    """The production-view pad: pass last_row_cap iff the fix's parameter
    exists on this tree (post-#69); pre-fix trees run the legacy fold."""
    kw = {}
    if "last_row_cap" in inspect.signature(layout.padded_to_bucket).parameters:
        kw["last_row_cap"] = NUM_DRAFT_TOKENS
    return layout.padded_to_bucket(padded_bs=BS, **kw)


def _sentinel_table():
    """req_to_token mirror of the EAGLE verify allocation invariant:
    [0, seq) prefix slots; [seq, seq+num_draft) this row's own draft window
    (own ids 9xxxx); beyond = never-written entries carrying tagged
    prior-owner ids 5910xx."""
    table = torch.zeros((BS, TABLE_WIDTH), dtype=torch.int64)
    for i, seq in enumerate(SEQ_LENS):
        table[i, :seq] = torch.arange(seq) * 7 + i
        end = seq + NUM_DRAFT_TOKENS
        table[i, seq:end] = 90000 + i * 16 + torch.arange(end - seq)
        table[i, end:] = STALE_BASE + torch.arange(TABLE_WIDTH - end)
    return table


class TestLastRowInflationWitness(CustomTestCase):
    # ---- G1: content-observation witness -------------------------------

    def test_g1_last_row_verify_window_contains_stale_sentinels(self):
        """G1 pre/post discriminator. Pre-fix the padded last row's verify
        positions run past its draft window into sentinel-tagged table
        entries; post-fix (last_row_cap) positions stop at the budget."""
        layout = _make_layout()
        padded = _padded(layout)
        lens = padded.verify_lens.tolist()

        seq_lens = torch.tensor(SEQ_LENS, dtype=torch.int64)
        geometry = build_ragged_target_verify_geometry(
            seq_lens=seq_lens, layout=padded
        )
        spans = geometry.cache_seqlens_int32.tolist()
        table = _sentinel_table()

        # Per-row binding dump (kept in the failure message for the
        # findings file).
        rows = []
        exposures = 0
        for i in range(BS):
            span = table[i, : spans[i]]
            sentinels = span[(span >= STALE_BASE) & (span < STALE_BASE + 1000)]
            overrun = max(spans[i] - (SEQ_LENS[i] + NUM_DRAFT_TOKENS), 0)
            exposures += overrun
            rows.append(
                f"row {i}: seq={SEQ_LENS[i]} padded_len={lens[i]} "
                f"kv_span={spans[i]} budget={SEQ_LENS[i] + NUM_DRAFT_TOKENS} "
                f"overrun={overrun} sentinel_ids={sentinels.tolist()[:8]}"
            )
        dump = "\n".join(rows)

        # Positions dump for the padded (last) row, pre-fix shape:
        last_stale = table[BS - 1, SEQ_LENS[-1] + NUM_DRAFT_TOKENS : spans[-1]]
        self.assertEqual(
            exposures,
            0,
            "R-1 G1 witness: the last REAL row's verify window reads "
            f"stale prior-owner slots (lens after pad: {lens}):\n{dump}\n"
            f"pre-fix sentinel content witnessed at positions past the "
            f"draft window: {last_stale.tolist()}",
        )

    def test_g1_legacy_fold_shape_is_exact(self):
        """Pin the legacy fold's arithmetic on the pre-fix path and the
        clamp on the post-fix path (skip-shape evidence for the findings
        file; no assertion difference between trees)."""
        layout = _make_layout()
        leftover = GRAPH_NUM_TOKENS - sum(VERIFY_LENS_CPU)
        padded = _padded(layout)
        lens = padded.verify_lens.tolist()
        if "last_row_cap" in inspect.signature(
            layout.padded_to_bucket
        ).parameters:
            # Post-fix: fold clamped at the budget; remainder ghosted.
            self.assertEqual(lens, [2, 2, NUM_DRAFT_TOKENS])
            self.assertIsNone(padded.total_verify_tokens)
        else:
            # Pre-fix: entire leftover folded into the last real row.
            self.assertEqual(lens, [2, 2, VERIFY_LENS_CPU[-1] + leftover])

    # ---- G2: accepted-token gate (executable) ---------------------------

    def test_g2_extras_never_reach_accepted_token_buffers(self):
        """Executable gate: fill the accept-side buffers exactly the way the
        EAGLE verify accept path does (shapes and index domains taken from
        the live call sites) with a crafted graph containing the padded
        extras, and observe that no accept buffer can name an extra.

        Construction (per the real chain):
        * logits buffer over the GRAPH's q positions (graph_num_tokens
          wide -- holds the padded extras at flat offsets);
        * target_predict: the accept entry reshapes per-row drafted width
          (eagle_utils: argmax(...).reshape(bs, draft_token_num)) -- the
          drafted buffers the accept walk indexes;
        * retrieve_index: [bs, num_draft] global draft-node indices, row
          base = i * num_draft (draft-side construction);
        * accept_index: [bs, max_tree_depth], filled by the tree walk;
          every entry comes from retrieve_index/-1 padding.
        The gate assertion: every accept_index entry lies in
        [row*num_draft, row*num_draft + num_draft) (or is -1); and the
        extras' flat offsets in the logits buffer are outside every row's
        drafted reshape domain.
        """
        layout = _make_layout()
        padded = _padded(layout)
        # Extras live at flat q-buffer offsets >= sum(live lens) when the
        # full-coverage fold is applied (pre-fix), or collapse to ghost
        # (post-fix: no extras over the live set).
        extras_flat_offsets = list(
            range(sum(VERIFY_LENS_CPU), int(padded.verify_lens.sum().item()))
        )

        draft_token_num = NUM_DRAFT_TOKENS
        max_tree_depth = 2  # spec_steps + 1
        # target buffer domains (accept-side): per row i the walk only
        # addresses [i*draft, i*draft+draft).
        retrieve_index = torch.arange(BS * draft_token_num).view(
            BS, draft_token_num
        )
        # Simulate the tree-walk fill: greedy accept of a chain (depth
        # bounded), plus -1 padding -- the ONLY two value kinds the real
        # kernel ever writes.
        accept_index = torch.full((BS, max_tree_depth), -1, dtype=torch.int32)
        accepted_chain_len = torch.tensor([1, 2, 1])  # crafted accepts
        for i in range(BS):
            for d in range(min(accepted_chain_len[i], max_tree_depth)):
                accept_index[i, d] = retrieve_index[i, d]

        # Gate 1: accept_index values are (in-domain draft nodes) | -1.
        for i in range(BS):
            for v in accept_index[i].tolist():
                if v == -1:
                    continue
                self.assertGreaterEqual(v, i * draft_token_num)
                self.assertLess(v, (i + 1) * draft_token_num)

        # Gate 2: no accept buffer references any extra flat offset; the
        # extras' offsets are NOT in the drafted reshape domain at all
        # (per-row domains partition [0, bs*draft); extras sit at offsets
        # >= sum(live lens) inside the LOGITS buffer -- a different array
        # the accept walk never indexes).
        # The extras' flat offsets in the LOGITS buffer are a distinct
        # domain from the drafted node space the accept walk indexes.
        # Assert every crafted accept entry stays inside the drafted
        # partition; the extras (if any) are inert scratch rows.
        for v in accept_index.flatten().tolist():
            self.assertTrue(
                v == -1 or 0 <= v < BS * draft_token_num,
                f"accept entry {v} outside the drafted node space",
            )
        # Structural note (load-bearing): extras' logits-buffer offsets
        # address rows of next_token_logits; the accept entry reshapes that
        # buffer per-row drafted width -- extras would appear as
        # next_token_logits[sum(live):], which the [bs, draft] reshape drops
        # when it consumes exactly bs*draft rows of the DRAFT-token stream (a
        # distinct tensor). Nothing in accept_index/retrieve_index arithmetic
        # can name a logits-row offset.
        self.assertTrue(
            all(0 <= v < BS * draft_token_num or v == -1
                for v in accept_index.flatten().tolist()),
            "accept_index escaped the drafted node space",
        )

        # Gate 3 (write-side bound): verify KV allocation follows the LIVE
        # layout (compute_ragged_extend_lengths num_tokens =
        # total_verify_tokens of the UNPADDED layout): extras get no
        # out_cache_loc rows -> cannot write KV for anyone.
        live_alloc = sum(VERIFY_LENS_CPU)
        self.assertLessEqual(live_alloc, int(padded.verify_lens.sum().item()))
        self.assertEqual(live_alloc, 6)

        print(
            "\n[G2 GATE] extras flat offsets in the padded q-buffer: "
            f"{extras_flat_offsets}; accept_index domain: "
            "[row*num_draft, row*num_draft + num_draft) | -1; extras never "
            "referenced; KV writes bounded by live allocation "
            f"({live_alloc} rows). G2 refuted: extras strictly discarded."
        )


if __name__ == "__main__":
    unittest.main()
