"""Regression tests for DSV4 PD request-state transfer index math.

Pins the invariants that keep PD-disaggregated DeepSeek-V4 deployments safe:

1. ``c4_state_transfer_indices`` / ``request_scoped_state_transfer_indices``
   never emit rows outside the per-request ring region they address.
   On disaggregated decode, ``req_pool_idx`` can exceed ``max_running_requests``
   (DecodeReqToTokenPool pre-allocates transfer slots); pools must therefore be
   sized by the *real* req-to-token capacity, and these functions must stay
   consistent with the ring layout the pools allocate.

2. Boundary behavior: nothing transfers at ``seq_len == 0`` or at exact block
   boundaries (``seq_len % ratio == 0``); prefill (ring 8) and decode (ring 16)
   derive the same logical tail positions under different physical rows.

3. ``CompressStatePool.transfer_indices`` dispatch: ratio-2 pair state only
   transfers for odd sequence lengths; request-scoped pools reject transfer
   calls on page-scoped pools.
"""

import unittest

import numpy as np
import torch

from sglang.srt.mem_cache.deepseek_v4_compress_state import (
    CompressStatePool,
    c4_state_transfer_indices,
    request_scoped_state_transfer_indices,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_REQ_SLOTS = 5  # stands in for req_to_token_pool rows (incl. PD prealloc)


class TestRequestScopedStateTransferIndices(CustomTestCase):
    def _assert_within_ring(self, idx, req_pool_idx, *, ring_size, ratio):
        pages_per_req = ring_size // ratio
        self.assertGreaterEqual(idx.min(), 0)
        self.assertLess(idx.max(), NUM_REQ_SLOTS * pages_per_req)
        # The transferred page must live inside *this* request's ring region.
        lo = req_pool_idx * pages_per_req
        hi = lo + pages_per_req - 1
        self.assertTrue(lo <= idx.min() <= idx.max() <= hi)

    def test_bounds_for_all_boundary_seq_lens(self):
        for ratio, ring_size in ((4, 16), (128, 128), (4, 8)):
            for req_pool_idx in range(NUM_REQ_SLOTS):
                for seq_len in range(0, 300):
                    idx = request_scoped_state_transfer_indices(
                        req_pool_idx,
                        seq_len,
                        ratio=ratio,
                        online=False,
                        ring_size=ring_size,
                    )
                    self.assertEqual(idx.dtype, np.int32)
                    if seq_len == 0 or seq_len % ratio == 0:
                        self.assertEqual(len(idx), 0)
                        continue
                    self.assertEqual(len(idx), 1)
                    self._assert_within_ring(
                        idx, req_pool_idx, ring_size=ring_size, ratio=ratio
                    )

    def test_online_c128_addresses_the_request_row(self):
        # Online C128 (ring_size=1) is one row per request slot, bounded by
        # num_req_slots — decode prealloc slots must fit the caller-sized pool.
        for req_pool_idx in range(3):
            for seq_len in (1, 127, 129):
                idx = request_scoped_state_transfer_indices(
                    req_pool_idx, seq_len, ratio=128, online=True, ring_size=1
                )
                self.assertEqual(idx.tolist(), [req_pool_idx])

    def test_ring_size_must_be_ratio_multiple(self):
        with self.assertRaises(AssertionError):
            request_scoped_state_transfer_indices(
                0, 5, ratio=128, online=False, ring_size=16
            )


class TestC4StateTransferIndices(CustomTestCase):
    def test_rows_match_live_tail_positions(self):
        ring_size = 16
        for req_pool_idx in range(NUM_REQ_SLOTS):
            for seq_len in range(1, 500):
                rows = c4_state_transfer_indices(
                    req_pool_idx, seq_len, ring_size=ring_size
                )
                state_len = seq_len % 4 + 4
                positions = np.arange(max(0, seq_len - state_len), seq_len)
                expect = (req_pool_idx * ring_size + positions % ring_size).astype(
                    np.int32
                )
                np.testing.assert_array_equal(rows, expect)
                self.assertLess(rows.max(), NUM_REQ_SLOTS * ring_size)

    def test_empty_request_transfers_nothing(self):
        self.assertEqual(len(c4_state_transfer_indices(0, 0, ring_size=8)), 0)

    def test_prefill_decode_ring_skew_is_positional(self):
        # PD prefill/decode rings differ (8 vs 16 with EAGLE/MTP); both sides
        # must derive the same *logical* tail positions row-for-row.
        for seq_len in range(1, 300):
            r16 = c4_state_transfer_indices(0, seq_len, ring_size=16)
            r8 = c4_state_transfer_indices(0, seq_len, ring_size=8)
            self.assertEqual(len(r16), len(r8))
            positions = np.arange(
                max(0, seq_len - (seq_len % 4 + 4)), seq_len, dtype=np.int64
            )
            np.testing.assert_array_equal(r16 - (0 * 16), 0 * 16 + positions % 16)
            np.testing.assert_array_equal(r8, 0 * 8 + positions % 8)

    def test_ring_size_validation(self):
        with self.assertRaises(ValueError):
            c4_state_transfer_indices(0, 10, ring_size=6)
        with self.assertRaises(ValueError):
            c4_state_transfer_indices(0, 10, ring_size=4)


class TestCompressStatePoolTransferDispatch(CustomTestCase):
    def _make_pool(self, *, ratio, ring_size, online=False, size=NUM_REQ_SLOTS):
        """CPU CompressStatePool sized by real request-row capacity."""
        return CompressStatePool(
            size=size * (ring_size if not online else 1),
            ring_size=ring_size,
            overlap=False,
            head_dim=8,
            dtype=torch.float32,
            device="cpu",
            enable_memory_saver=False,
            ratio=ratio,
            online=online,
            request_scoped=True,
        )

    def test_ratio2_pair_state_only_for_odd_tail(self):
        pool = self._make_pool(ratio=2, ring_size=1)
        for req_pool_idx in range(NUM_REQ_SLOTS):
            self.assertEqual(
                pool.transfer_indices(req_pool_idx, 10).tolist(), []
            )
            self.assertEqual(
                pool.transfer_indices(req_pool_idx, 11).tolist(), [req_pool_idx]
            )

    def test_offline_request_scoped_matches_module_fn(self):
        pool = self._make_pool(ratio=128, ring_size=128)
        for req_pool_idx in range(NUM_REQ_SLOTS):
            idx = pool.transfer_indices(req_pool_idx, 130)
            expect = request_scoped_state_transfer_indices(
                req_pool_idx, 130, ratio=128, online=False, ring_size=128
            )
            np.testing.assert_array_equal(idx, expect)

    def test_page_scoped_pool_rejects_transfer(self):
        pool = CompressStatePool(
            size=NUM_REQ_SLOTS * 16,
            ring_size=16,
            overlap=True,
            head_dim=8,
            dtype=torch.float32,
            device="cpu",
            enable_memory_saver=False,
            ratio=4,
            request_scoped=False,
        )
        with self.assertRaises(AssertionError):
            pool.transfer_indices(0, 10)


if __name__ == "__main__":
    unittest.main()
