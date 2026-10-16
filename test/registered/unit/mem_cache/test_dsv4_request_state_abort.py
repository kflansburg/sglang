"""Regression tests for DSV4 request-scoped compress-state cleanup.

PD-disaggregated DeepSeek-V4 keeps per-request compress state in rings
addressed by ``req_pool_idx``: the offline C128 ring and the ratio-2
pending-pair state (request-scoped pools), plus the unified C4 ring.
On disaggregated decode a failed/aborted/timed-out transfer frees the req
slot without a compress-state free; the safety invariant is that the *next*
user of that slot always starts from cleared rings:

- recv path: ``DeepSeekV4TokenToKVPool.clear_request_scoped_state`` is called
  with the newly pre-allocated slot before request-state payloads are written
  (disaggregation/decode.py),
- reuse path: ``clear_c4_req_states`` fires from the req-to-token pool's
  on-alloc hook for the unified C4 ring.

Test gap pinned here: clearing must (a) reset ``clear()`` semantics (kv=0,
score=-inf) for every request-scoped pool and only the addressed slot,
(b) cover the online-C128 3*head_dim row layout, and (c) tolerate
pre-allocated slots beyond ``max_running_requests`` (DecodeReqToTokenPool),
and the DSV4 pool must be constructed with ``num_req_slots`` equal to the
real req-to-token row count at every construction site.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.decode import DecodeReqToTokenPool
from sglang.srt.mem_cache.deepseek_v4_compress_state import KVAndScore
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _filled(rows, width, value):
    return KVAndScore(torch.full((rows, width), value))


def _offline_ring_pool(ratio, ring_size, num_slots, value):
    """Request-scoped ring pool (offline c128 or ratio-2 pair), pre-dirtied."""
    return SimpleNamespace(
        ratio=ratio,
        ring_size=ring_size,
        request_scoped=True,
        kv_score_buffer=_filled(num_slots * ring_size + ring_size + 1, 8, value),
    )


def _token_pool(pools):
    token_pool = object.__new__(DeepSeekV4TokenToKVPool)
    token_pool.compress_state_pools = pools
    return token_pool


class TestClearRequestScopedState(CustomTestCase):
    def test_resets_offline_c128_ring_only_for_target_slot(self):
        ring_size = 8
        pool = _offline_ring_pool(128, ring_size, num_slots=4, value=5.0)
        token_pool = _token_pool([pool, None])

        state = pool.kv_score_buffer.kv_score
        half = state.shape[-1] // 2
        with patch(
            "sglang.srt.mem_cache.deepseek_v4_memory_pool.ONLINE_C128", False
        ):
            token_pool.clear_request_scoped_state(2)

        rows = slice(2 * ring_size, 3 * ring_size)
        self.assertTrue((state[rows, :half] == 0).all())
        self.assertTrue(torch.isneginf(state[rows, half:]).all())
        # Every other slot, and the padding rows, stay dirty.
        others = torch.cat([state[: 2 * ring_size], state[3 * ring_size :]])
        self.assertTrue((others == 5).all())

    def test_resets_ratio2_pair_state(self):
        pool = _offline_ring_pool(2, 1, num_slots=4, value=5.0)
        token_pool = _token_pool([pool])

        token_pool.clear_request_scoped_state(3)

        state = pool.kv_score_buffer.kv_score
        half = state.shape[-1] // 2
        self.assertTrue((state[3, :half] == 0).all())
        self.assertTrue(torch.isneginf(state[3, half:]).all())
        self.assertTrue((state[:3] == 5).all())

    def test_clears_all_request_scoped_pools_but_not_page_scoped(self):
        ring_pool = _offline_ring_pool(128, 8, num_slots=4, value=5.0)
        pair_pool = _offline_ring_pool(2, 1, num_slots=4, value=6.0)
        page_scoped = _offline_ring_pool(4, 8, num_slots=8, value=7.0)
        page_scoped.request_scoped = False
        token_pool = _token_pool([ring_pool, pair_pool, page_scoped])

        with patch(
            "sglang.srt.mem_cache.deepseek_v4_memory_pool.ONLINE_C128", False
        ):
            token_pool.clear_request_scoped_state(1)

        self.assertFalse((ring_pool.kv_score_buffer.kv_score[8:16] == 5).all().item())
        self.assertFalse((pair_pool.kv_score_buffer.kv_score[1] == 6).all().item())
        self.assertTrue((page_scoped.kv_score_buffer.kv_score == 7).all())

    def test_online_c128_row_layout_uses_third_split(self):
        """Online C128 rows are 3*head_dim wide: [score | kv | rope] — the
        clear must fill the score third with -inf and zero the rest."""
        head_dim = 4
        pool = SimpleNamespace(
            ratio=128,
            ring_size=1,
            request_scoped=True,
            kv_score_buffer=KVAndScore(torch.full((6, 3 * head_dim), 2.0)),
        )
        token_pool = _token_pool([pool])

        with patch(
            "sglang.srt.mem_cache.deepseek_v4_memory_pool.ONLINE_C128", True
        ):
            token_pool.clear_request_scoped_state(4)

        row = pool.kv_score_buffer.kv_score[4]
        self.assertTrue(torch.isneginf(row[:head_dim]).all())
        self.assertTrue((row[head_dim:] == 0).all())
        others = torch.cat(
            [pool.kv_score_buffer.kv_score[:4], pool.kv_score_buffer.kv_score[5:]]
        )
        self.assertTrue((others == 2).all())

    def test_prealloc_slot_beyond_max_running_requests(self):
        """A decode-side prealloc slot can exceed max_running_requests; the
        clear must address it through the real pool capacity."""
        num_slots = 6  # == req_to_token rows, > max_running_requests
        pool = _offline_ring_pool(128, 8, num_slots=num_slots, value=5.0)
        token_pool = _token_pool([pool])
        with patch(
            "sglang.srt.mem_cache.deepseek_v4_memory_pool.ONLINE_C128", False
        ):
            token_pool.clear_request_scoped_state(4)  # prealloc slot
        state = pool.kv_score_buffer.kv_score
        half = state.shape[-1] // 2
        self.assertTrue((state[32:40, :half] == 0).all())
        self.assertTrue(torch.isneginf(state[32:40, half:]).all())


class TestDecodeAbortRecycleInvariant(CustomTestCase):
    """The abort/timeout path frees the slot without clearing compress state;
    the next owner must observe cleared rings through the same machinery the
    recv path uses."""

    def test_decode_pool_alloc_hook_recovers_aborted_slot(self):
        # Register the DSV4 C4-recovery hook the way the configurator does.
        req_pool = DecodeReqToTokenPool(
            2, 16, "cpu", enable_memory_saver=False, pre_alloc_size=2
        )
        ring_size = 8
        c4 = _offline_ring_pool(4, ring_size, num_slots=5, value=7.0)
        token_pool = object.__new__(DeepSeekV4TokenToKVPool)
        token_pool._unified_kv = True
        token_pool.compress_state_pools = [c4]
        token_pool.indexer_compress_state_pools = []
        token_pool.get_ring_size = MagicMock(return_value=ring_size)
        req_pool.register_on_alloc_rows(token_pool.clear_c4_req_states)

        def grab():
            req = SimpleNamespace(
                kv=SimpleNamespace(
                    req_pool_idx=None,
                    kv_committed_len=0,
                    kv_allocated_len=0,
                    holds_kv=False,
                ),
                inflight_middle_chunks=0,
            )
            (idx,) = req_pool.alloc([req])
            return idx

        first = grab()
        # Poison the slot the way a partially-received transfer would leave it
        # before an abort: garbage rows in the request's C4 ring.
        state = c4.kv_score_buffer.kv_score
        half = state.shape[-1] // 2
        state[first * ring_size : (first + 1) * ring_size] = 3.0
        # Abort frees the slot WITHOUT a compress-state clear (that is the
        # known deferred-clear design: deferred release handles KV pages; the
        # request rings wait for the next owner). DecodeReqToTokenPool frees to
        # the tail and allocs from the front, so the recycled slot surfaces
        # after the intervening slots are taken.
        req_pool.free(
            SimpleNamespace(
                kv=SimpleNamespace(req_pool_idx=first, holds_kv=True)
            )
        )

        seen = set()
        recycled = None
        for _ in range(5):
            idx = grab()
            seen.add(idx)
            if idx == first:
                recycled = idx
                break
        self.assertEqual(recycled, first)
        rows = slice(recycled * ring_size, (recycled + 1) * ring_size)
        self.assertTrue((state[rows, :half] == 0).all())
        self.assertTrue(torch.isneginf(state[rows, half:]).all())


class TestPoolNumReqSlotsContract(CustomTestCase):
    """Every DSV4 request-ring consumer is sized by the req-to-token row
    capacity (which includes PD decode prealloc slots). Pins the wiring, not
    just the math: if a construction site regresses to max_num_reqs+1 sizing,
    transfer-index and clear calls overflow the rings (PD decode prealloc
    hands out slots beyond max_running_requests)."""

    def test_dsv4_pool_ctor_passes_real_req_capacity(self):
        import sglang.srt.mem_cache.kv_cache_configurator as kcc

        source = open(kcc.__file__).read()
        # All DeepSeekV4TokenToKVPool construction happens once in
        # kv_cache_configurator; it must feed the req-to-token row count.
        self.assertIn(
            "num_req_slots=req_to_token_pool.req_to_token.shape[0]", source
        )

    def test_dsa_tail_pool_ctor_passes_real_req_capacity(self):
        import sglang.srt.mem_cache.kv_cache_configurator as kcc
        import sglang.srt.mem_cache.memory_pool as mp

        source = open(kcc.__file__).read()
        # DSATokenToKVPool._compress_tail_k is indexed by raw req_pool_idx;
        # every max_running_requests argument to the DSA pool builders must be
        # the req-to-token row count, not the admission cap.
        self.assertGreaterEqual(
            source.count(
                "max_running_requests=req_to_token_pool.req_to_token.shape[0]"
            )
            + source.count(
                "max_running_requests=(req_to_token_pool.req_to_token.shape[0])"
            ),
            1,
        )
        # And the pool must not silently fall back to a smaller row count
        # than the buffers it writes: the tail buffer dim-0 comes from the
        # constructor argument.
        pool_source = open(mp.__file__).read()
        self.assertIn(
            "req_pool_size = max_running_requests + 1", pool_source
        )

    def test_ring_rows_cover_every_addressable_slot(self):
        ring_size = 8
        max_running = 4
        prealloc = 3
        req_pool = DecodeReqToTokenPool(
            max_running, 16, "cpu", enable_memory_saver=False, pre_alloc_size=prealloc
        )
        num_req_slots = req_pool.req_to_token.shape[0]
        self.assertEqual(num_req_slots, max_running + prealloc + 1)
        # Every addressable slot's ring region fits within the padded ring
        # allocation (rows + ring_size + 1, padded to a ratio multiple).
        ring_rows = num_req_slots * ring_size
        for req_pool_idx in range(num_req_slots):
            start = req_pool_idx * ring_size
            self.assertLess(start + ring_size - 1, ring_rows + ring_size)


if __name__ == "__main__":
    unittest.main()
