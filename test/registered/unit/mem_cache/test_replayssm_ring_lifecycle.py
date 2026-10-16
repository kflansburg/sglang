"""ReplaySSM ring + extra_buffer ping-pong slot lifecycle regression (CPU).

Prod config: Kimi K3 hybrid linear attn with --mamba-radix-cache-strategy
extra_buffer and --enable-linear-replayssm-spec, PD-disaggregated decode with
cpu_tensor retraction backup. This test drives the REAL HybridReqToTokenPool
alloc / free_mamba_cache state machine (stubbing only the KV row table and the
mamba state store) and pins three invariants that a retract/resume cycle must
not break:

1. FRESH-ROW RING RESET: spec rings (replayssm_spec_write_pos / cache_base /
   is_flush, keyed by req_pool_idx) are zeroed for every freshly allocated row,
   including a row REUSED after a retract (generation bump must not carry stale
   cursors).
2. DECODE RING RESET: replayssm_write_pos (keyed by mamba slot) is zeroed when a
   fresh mamba slot is assigned (memory_pool.py:1618-1619 behaviour), i.e. a
   resumed request starts ring-empty against its restored checkpoint.
3. SINGLE OWNERSHIP across donate/boundary/finish/abort: one mamba slot id must
   never be owned twice (req pool_idx ∪ ping-pong buffer ∪ allocator free
   list). This mirrors the audit's harness-slot-lifecycle oracle
   (audit_mamba2/harness_slot_lifecycle.py) against the real code paths.
"""

import importlib.util
import os

import pytest
import torch

try:
    from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.test.ci.ci_register import register_cpu_ci

    register_cpu_ci(est_time=11, suite="base-a-test-cpu")
except ImportError:  # thin queue/agent venv: load real modules with stubbed deps
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _spec = importlib.util.spec_from_file_location(
        "_mamba_ring_loader", os.path.join(_HERE, "_mamba_ring_loader.py")
    )
    _loader = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_loader)
    _alloc_mod, _mempool_mod = _loader.load()
    MambaSlotAllocator = _alloc_mod.MambaSlotAllocator
    HybridReqToTokenPool = _mempool_mod.HybridReqToTokenPool

    class register_cpu_ci:  # noqa: N801 - the CI helper's name
        def __init__(self, *a, **k):
            pass

NUM_SLOTS = 16  # mamba slots (slot 0 is the reserved dummy)
NUM_ROWS = 4  # req_pool rows (row 0 is the reserved dummy)


class _SparseMambaPool:
    """Stand-in for MambaPool: only carries the ring-cursor tensors alloc/free
    touch on the replayssm-spec + replayssm paths."""

    def __init__(self):
        self.replayssm_write_pos = torch.zeros(NUM_SLOTS + 1, dtype=torch.int32)
        self.replayssm_spec_write_pos = torch.zeros(NUM_ROWS + 1, dtype=torch.int32)
        self.replayssm_cache_base = torch.zeros(NUM_ROWS + 1, dtype=torch.int32)
        self.replayssm_is_flush = torch.zeros(NUM_ROWS + 1, dtype=torch.int8)


class _ReqKV:
    """Minimal req.kv stand-in with the field set alloc/free_mamba_cache touch."""

    def __init__(self, req_pool_idx=None):
        self.req_pool_idx = req_pool_idx
        self.mamba_pool_idx = None
        self.mamba_needs_clear = False
        self.mamba_ping_pong_track_buffer = None
        self.mamba_next_track_idx = None
        self.mamba_last_track_idx = None
        self.mamba_last_track_seqlen = None
        self.mamba_prev_track_seqlen = None
        self.mamba_cow_src_index = None

    @property
    def holds_kv(self):
        return self.req_pool_idx is not None

    @property
    def holds_mamba(self):
        return self.mamba_pool_idx is not None


class _Req:
    def __init__(self, rid, req_pool_idx=None):
        self.rid = rid
        self.kv = _ReqKV(req_pool_idx)
        # ensure_mamba_capacity / logs read these; alloc itself doesn't but keep
        # the shape close to a real Req for future-proofing.
        self.origin_input_ids = [1, 2]
        self.output_ids = []


def _make_pool(extra_buffer_lazy=False):
    pool = object.__new__(HybridReqToTokenPool)
    # ReqToTokenPool fields that super().alloc() needs:
    pool.free_slots = list(range(1, NUM_ROWS + 1))
    pool.req_generation = torch.zeros(NUM_ROWS + 1, dtype=torch.int64)
    pool._on_alloc_rows = None
    pool._aux_cache = None
    # HybridReqToTokenPool.alloc fields:
    pool.mamba_pool = _SparseMambaPool()
    pool.device = "cpu"
    pool.enable_mamba_extra_buffer = True
    pool.enable_mamba_extra_buffer_lazy = extra_buffer_lazy
    pool.mamba_ping_pong_track_buffer_size = 2
    pool.mamba_initial_tracking_slots = 1 if extra_buffer_lazy else 2
    pool.mamba_allocator = MambaSlotAllocator(size=NUM_SLOTS, device="cpu")
    pool.req_index_to_mamba_index_mapping = torch.zeros(
        NUM_ROWS + 1, dtype=torch.int32
    )
    pool.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.zeros(
        (NUM_ROWS + 1, pool.mamba_ping_pong_track_buffer_size), dtype=torch.int64
    )
    return pool


def _live_slots(pool, reqs):
    owned = {}
    for r in reqs:
        owned.setdefault(int(r.kv.mamba_pool_idx), set()).add(f"{r.rid}.pool_idx")
        for s in r.kv.mamba_ping_pong_track_buffer.tolist():
            if s != -1:
                owned.setdefault(s, set()).add(f"{r.rid}.buf")
    return owned


def _assert_single_ownership(pool, reqs, tree):
    owned = _live_slots(pool, reqs)
    for s in tree:
        owned.setdefault(s, set()).add("tree")
    free = pool.mamba_allocator.free_slots.tolist()
    for s, owners in owned.items():
        assert len(owners) == 1, f"slot {s} owned by {owners}"
        assert s not in free, f"slot {s} owned by {owners} but also on free list"
    assert len(free) == len(set(free)), f"double free: {free}"
    assert all(0 < s <= NUM_SLOTS for s in free), f"bad free list: {free}"


_HRTP = HybridReqToTokenPool


class TestReplaySSMRingLifecycle:
    def test_spec_rings_zeroed_on_fresh_rows(self):
        pool = _make_pool()
        reqs = [_Req("a"), _Req("b")]
        got = _HRTP.alloc(pool, reqs)
        assert got == [r.kv.req_pool_idx for r in reqs]
        rows = [r.kv.req_pool_idx for r in reqs]
        assert (
            pool.mamba_pool.replayssm_spec_write_pos[rows] == 0
        ).all()
        assert (pool.mamba_pool.replayssm_cache_base[rows] == 0).all()
        assert (pool.mamba_pool.replayssm_is_flush[rows] == 0).all()
        assert (pool.mamba_pool.replayssm_write_pos[
            [int(r.kv.mamba_pool_idx) for r in reqs]
        ] == 0).all()

    def test_row_reuse_after_retract_resets_rings(self):
        pool = _make_pool()
        req = _Req("a")
        _HRTP.alloc(pool, [req])
        row = req.kv.req_pool_idx
        # Simulate a decode phase that advanced the spec ring cursors.
        pool.mamba_pool.replayssm_spec_write_pos[row] = 5
        pool.mamba_pool.replayssm_cache_base[row] = 3
        pool.mamba_pool.replayssm_is_flush[row] = 1
        pool.mamba_pool.replayssm_write_pos[int(req.kv.mamba_pool_idx)] = 7
        # Retract: free all mamba state + release the row (mirrors
        # free_mamba_cache + disagg decode free()).
        pool.free_mamba_cache(req)
        pool.free_slots.append(row)
        req.kv.req_pool_idx = None
        # A different request takes the row first (row reuse under pressure)...
        other = _Req("b")
        _HRTP.alloc(pool, [other])
        assert pool.mamba_pool.replayssm_spec_write_pos[other.kv.req_pool_idx] == 0
        assert pool.mamba_pool.replayssm_cache_base[other.kv.req_pool_idx] == 0
        assert pool.mamba_pool.replayssm_is_flush[other.kv.req_pool_idx] == 0
        # ...then the original request resumes and also gets clean rings.
        pool.free_mamba_cache(other)
        pool.free_slots.append(other.kv.req_pool_idx)
        other.kv.req_pool_idx = None
        _HRTP.alloc(pool, [req])
        assert pool.mamba_pool.replayssm_spec_write_pos[req.kv.req_pool_idx] == 0
        assert pool.mamba_pool.replayssm_write_pos[int(req.kv.mamba_pool_idx)] == 0

    def test_ping_pong_single_ownership_across_donate_finish_abort(self):
        # Non-lazy extra_buffer: both slots live for the request's lifetime.
        for lazy in (True, False):
            # The radix tree is 1:1 with its pool; reset it per pool instance.
            tree = []
            pool = _make_pool(extra_buffer_lazy=lazy)
            reqs = [_Req("a"), _Req("b"), _Req("c")]
            _HRTP.alloc(pool, reqs)
            _assert_single_ownership(pool, reqs, tree)
            # Donate a's keep slot to the tree (mirrors
            # donate_mamba_ping_pong_slot): replace buf[keep] with a fresh slot.
            a = reqs[0]
            keep = pool.get_mamba_ping_pong_keep_idx(a)
            new_slot = pool.mamba_allocator.alloc(1)
            donated = pool.donate_mamba_ping_pong_slot(a, new_slot)
            assert len(donated) == 1 and int(donated[0]) != -1
            tree.append(int(donated[0]))
            _assert_single_ownership(pool, reqs, tree)
            # Finish b (donate keep, free everything else).
            b = reqs[1]
            keep_b = pool.get_mamba_ping_pong_keep_idx(b)
            tree.append(int(b.kv.mamba_ping_pong_track_buffer[keep_b]))
            pool.free_mamba_cache(b, mamba_ping_pong_track_buffer_to_keep=keep_b)
            reqs.remove(b)
            _assert_single_ownership(pool, reqs, tree)
            # Abort c (free everything).
            pool.free_mamba_cache(reqs[1])
            reqs.pop(1)
            _assert_single_ownership(pool, reqs, tree)
            # Every slot except tree + a's holdings returns to the free list.
            expected_free = (
                set(range(1, NUM_SLOTS + 1)) - set(tree) - set(_live_slots(pool, reqs))
            )
            assert set(pool.mamba_allocator.free_slots.tolist()) == expected_free

    def test_fresh_slot_generation_bump_does_not_preserve_cursors(self):
        # Two sequential reqs on the SAME row: generation bumps, rings reset.
        pool = _make_pool()
        r1 = _Req("r1")
        _HRTP.alloc(pool, [r1])
        row = r1.kv.req_pool_idx
        g0 = int(pool.req_generation[row])
        pool.free_mamba_cache(r1)
        pool.free_slots.append(row)
        r1.kv.req_pool_idx = None
        r2 = _Req("r2")
        _HRTP.alloc(pool, [r2])
        assert r2.kv.req_pool_idx == row
        assert int(pool.req_generation[row]) == g0 + 1
        assert pool.mamba_pool.replayssm_spec_write_pos[row] == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
