"""REGRESSION TEST for R-1 (board #66 -> #69): RaggedVerifyLayout pad fold cap.

pad_verify_lens_to_bucket's num_pad_reqs == 0 (batch exactly fills the
captured tier) legacy fold pushed the whole `leftover` into the LAST REAL
row, widening its verify past its drafted width: the row's read positions
then reach never-allocated per-row table entries (stale prior-owner slot
ids; content exposure into its verify attention). The fix caps the fold at
the per-request draft budget (`last_row_cap = num_draft_tokens`); uncovered
remainder collapses into ghost semantics (layout tolerates
total_verify_tokens=None), exactly as the pre-existing cap=N path.

Regression coverage: torch impl of PaddedToBucket + the RaggedVerifyLayout
entry, both at (a) padded_bs == bs with a deficit trigger, (b) full-padding
padded_bs > bs shape unchanged (phantom-row path unaffected).
"""

import unittest

import torch

from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _layout(lens, total_tokens, *, num_draft):
    return RaggedVerifyLayout._assemble(
        verify_lens_cpu=lens,
        total_verify_tokens=total_tokens,
        graph_num_tokens=total_tokens + 7,  # tier has 7 slack tokens
        device=torch.device("cpu"),
    )


class TestRaggedVerifyPadCap(CustomTestCase):
    def test_full_tier_fold_is_capped_at_draft_budget(self):
        # bs == padded_bs (no padding row), sum(lens) < graph_num_tokens.
        # 3 requests, per-req num_draft = 6; lens sum = 11, graph has slack 7.
        layout = _layout([4, 5, 2], 11, num_draft=6)
        padded = layout.padded_to_bucket(padded_bs=3, last_row_cap=6)
        lens = padded.verify_lens.cpu().tolist()
        # Last real row: 2 + min(7, 6 - 2) = 6 (NOT 2 + 7 = 9 as pre-fix).
        self.assertEqual(lens[:2], [4, 5])
        self.assertEqual(lens[2], 6, "last real row exceeds its draft budget")
        for v in lens:
            self.assertLessEqual(v, 6)

    def test_full_tier_no_deficit_unchanged(self):
        # sum(lens) == graph_num_tokens: no leftover, sentinel fold zero.
        layout = RaggedVerifyLayout._assemble(
            verify_lens_cpu=[4, 5, 2],
            total_verify_tokens=11,
            graph_num_tokens=11,
            device=torch.device("cpu"),
        )
        padded = layout.padded_to_bucket(padded_bs=3, last_row_cap=6)
        self.assertEqual(padded.verify_lens.cpu().tolist(), [4, 5, 2])

    def test_phantom_row_path_unaffected_by_cap(self):
        # padded_bs > bs (padding rows exist to absorb leftover).
        layout = _layout([4, 5, 2], 11, num_draft=6)
        padded = layout.padded_to_bucket(padded_bs=5, last_row_cap=6)
        lens = padded.verify_lens.cpu().tolist()
        # Phantom rows distribute the slack: base=7//2=3, rem=1 → [4, 3].
        self.assertEqual(lens[:3], [4, 5, 2])
        self.assertEqual(sorted(lens[3:]), [3, 4])
        self.assertEqual(sum(lens), 11 + 7)

    def test_last_row_cap_smaller_than_actual_leftover_room(self):
        # Leftover > cap - last len: fold truncated, remainder dropped.
        layout = _layout([1, 1], 2, num_draft=8)
        padded = layout.padded_to_bucket(padded_bs=2, last_row_cap=8)
        lens = padded.verify_lens.cpu().tolist()
        self.assertEqual(lens, [1, 8], "fold must stop at last_row_cap")

    def test_uncapped_legacy_shape_preserved_without_param(self):
        # Backward compat: callers that pass nothing keep the legacy fold.
        layout = _layout([4, 5, 2], 11, num_draft=6)
        padded = layout.padded_to_bucket(padded_bs=3)
        lens = padded.verify_lens.cpu().tolist()
        self.assertEqual(lens, [4, 5, 2 + 7], "legacy uncapped path changed")

    def test_assembled_layout_total_tokens_none_with_cap(self):
        # R-1 layout assembly drops total_verify_tokens (tolerated ghost path)
        # when last_row_cap is supplied — mirrors the cap=N path's contract.
        layout = _layout([4, 5, 2], 11, num_draft=6)
        padded = layout.padded_to_bucket(padded_bs=3, last_row_cap=6)
        self.assertIsNone(padded.total_verify_tokens)


if __name__ == "__main__":
    unittest.main()
