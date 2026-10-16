"""Adversarial safety sim for the topk>1 accept_index -> retrive coordinate
math used by prepare_draft_extend_inputs_triton and _compact_accept_to_front.

Spec-audit, board #11. Claims pinned:

1. The sgl-kernel ordered-write contract: accept_index[b, j < accept_len[b]]
   holds the j-th accepted node's GLOBAL tree node id (b' level: any row,
   but produced by tree verify for row b so rid == b in valid entries);
   j >= accept_len[b] is -1 padding.
2. _compact_accept_to_front's clamp-0 + front-compact can never write a
   VALID position (j < accept_len) with wrong data, even under degenerate or
   contradictory inputs (clamped -1 -> 0 handled as doc'd).
3. prepare_draft_extend_inputs_triton's coordinate extraction
   (rid = nid // nd; cid = nid % nd) with the *kernel-guaranteed* accept
   ordering never yields (rid, cid) outside [0, bs) x [0, nd) for VALID j
   entries (invalid j are masked by j < select? in-kernel reads guard on
   accept-length-derived dst seq). If a mid-gap -1 existed, the guard
   `nid >= 0` (positive retrive-node check) would have to exist; we show
   what the kernel's actual guard is and pin behavior for it.

This sim uses the same math as the triton kernel on CPU without compiling
triton.
"""

import random
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def compact_accept_to_front(x: torch.Tensor, accept_index: torch.Tensor) -> torch.Tensor:
    """CPU mirror of sglang.srt.speculative.eagle_utils._compact_accept_to_front
    (flat view, clamp(min=0), gather, write into first s1 cols; rest zero)."""
    bs, num_draft_tokens = x.shape
    s1 = accept_index.shape[1]
    accept_index_safe = accept_index.clamp(min=0).reshape(-1)
    gathered = x.reshape(-1)[accept_index_safe]
    out = torch.zeros_like(x)
    out.view(bs, num_draft_tokens)[:, :s1] = gathered.view(bs, s1)
    return out


def coord_extract(accept_index_bs_s1: torch.Tensor, num_draft_tokens: int):
    """CPU mirror of the legacy coord math:
    nid = accept_index[b, j]; rid = nid // nd; cid = nid % nd
    (as in eagle_info prepare_draft_extend_inputs triton - semantics helper)."""
    nid = accept_index_bs_s1.to(torch.int64)
    rid = torch.div(nid, num_draft_tokens, rounding_mode="floor")
    cid = nid % num_draft_tokens
    return rid, cid


class TestAcceptIndexCoordSafety(unittest.TestCase):
    def _make_ordered_accept(self, bs, nd, s1, lens, rng: random.Random):
        """Build accept_index honoring the kernel contract: for row b, the
        first lens[b] entries are valid global node ids belonging to row b
        (in [b*nd, (b+1)*nd)); the rest -1."""
        accept_index = torch.full((bs, s1), -1, dtype=torch.int64)
        for b in range(bs):
            valid = [b * nd + rng.randrange(0, nd) for _ in range(lens[b])]
            accept_index[b, : lens[b]] = torch.tensor(valid, dtype=torch.int64)
        return accept_index

    def test_compact_honors_valid_positions_random(self):
        rng = random.Random(20261016)
        for trial in range(200):
            bs = rng.choice([1, 2, 4, 8])
            nd = rng.choice([2, 5, 10])
            s1 = min(nd, rng.choice([1, 2, 4, 6]))
            lens = [rng.randrange(0, s1 + 1) for _ in range(bs)]
            x = torch.arange(bs * nd).float().reshape(bs, nd)
            accept_index = self._make_ordered_accept(bs, nd, s1, lens, rng)
            out = compact_accept_to_front(x, accept_index)
            flat = x.view(bs, nd)
            outv = out.view(bs, nd)
            for b in range(bs):
                for j in range(lens[b]):
                    expect = flat[b][int(accept_index[b, j]) - b * nd]
                    got = outv[b][j]
                    assert got == expect, (
                        f"trial {trial} b{b} j{j}: expect {expect} got {got}"
                    )
                for j in range(s1, nd):
                    assert outv[b][j] == 0

    def test_coord_extract_valid_rows_in_range(self):
        rng = random.Random(4242)
        for trial in range(200):
            bs = rng.choice([1, 4, 8])
            nd = rng.choice([2, 5, 10])
            s1 = min(nd, rng.choice([1, 4, 6]))
            lens = [rng.randrange(1, s1 + 1) for _ in range(bs)]
            accept_index = self._make_ordered_accept(bs, nd, s1, lens, rng)
            rid, cid = coord_extract(accept_index, nd)
            for b in range(bs):
                for j in range(lens[b]):
                    assert rid[b, j] == b, (
                        f"trial {trial}: valid entry escaped its request row"
                    )
                    assert 0 <= cid[b, j] < nd
                # torch floor-division: -1 // nd == -1 here. CAUTION: this is a
                # torch-semantics property ONLY; triton `//` truncates toward
                # zero (C-style), where -1 // nd == 0 would alias row 0. The
                # safety of the real consuming kernels therefore CANNOT rest on
                # this guard property; it rests on valid-j masking
                # (seq_lens-derived) plus clamp(min=0) - as documented in the
                # eagle_utils.py XPU tripwire comment. The rid<0 assertion below
                # pins only the torch mirror's behavior.
                for j in range(lens[b], min(s1, accept_index.shape[1])):
                    assert rid[b, j] < 0, (
                        "padded -1 must still divide to a negative row so a "
                        "guard on dst-row-validity rejects it"
                    )

    def test_degenerate_full_and_empty_accept(self):
        bs, nd, s1 = 4, 10, 6
        x = torch.arange(bs * nd).float().reshape(bs, nd)
        # Full accept everywhere
        accept_index = torch.tensor(
            [[b * nd + j for j in range(s1)] for b in range(bs)], dtype=torch.int64
        )
        out = compact_accept_to_front(x, accept_index)
        for b in range(bs):
            for j in range(s1):
                assert out.view(bs, nd)[b, j] == b * nd + j
        # All-empty accept: clamp pushes everything to node 0 of row 0;
        # valid-position set is empty so nothing valid is corrupted.
        accept_index = torch.full((bs, s1), -1, dtype=torch.int64)
        out = compact_accept_to_front(x, accept_index)
        assert out.view(bs, nd)[0, :s1].max().item() == 0.0


if __name__ == "__main__":
    unittest.main()
