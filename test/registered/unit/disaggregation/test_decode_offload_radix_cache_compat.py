"""Unit tests for DecodeKVCacheOffloadManager._release_finished_req's
interaction with --disaggregation-decode-enable-radix-cache.

These tests cover three regressions introduced when the decode radix cache
flag (PR #24257) was added to a code path that the offload manager
(decode_kvcache_offload_manager.py, unchanged since ae15fca19) had been
written under ChunkCache assumptions:

  1. Use-after-free: the offload manager freed slots in [0:prefill_len]
     directly, but with radix cache enabled those slots are owned by
     TreeNode.value (cache_protected_len boundary) and concurrent prefix
     matches would hand the recycled slots to another request.
  2. Radix lock leak: the offload manager mutated
     tree_cache.protected_size_ directly instead of calling dec_lock_ref
     on req.last_node, so the lock_ref on the matched prefix node
     accumulated forever and the tree could never evict the node.
  3. Re-entry AssertionError: _release_finished_req was callable twice
     for the same request when an in-flight offload ack arrived after
     finalize_release_on_finish had already popped the committed KV,
     causing pop_committed_kv_cache to assert and crash the scheduler.

The tests construct fake tree_cache / token_to_kv_pool_allocator /
req_to_token_pool objects that record method calls, then drive
_release_finished_req with carefully constructed Req fixtures to verify
the three behaviors.
"""

import unittest
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


# --- Fixtures ----------------------------------------------------------


class _FakeNode:
    """Minimal stand-in for radix_cache.TreeNode."""

    def __init__(self, key_len: int = 4):
        self.lock_ref = 1
        self.key_len = key_len


class _FakeAllocator:
    """Records every allocator.free(indices) call."""

    page_size = 1

    def __init__(self):
        self.freed: List[List[int]] = []

    def free(self, indices: torch.Tensor) -> None:
        if isinstance(indices, torch.Tensor):
            self.freed.append(indices.tolist())
        else:
            self.freed.append(list(indices))


class _FakeReqPool:
    """Stand-in for ReqToTokenPool that supports req_to_token row access
    and free(req)."""

    def __init__(self, rows):
        # rows: dict[req_pool_idx -> torch.Tensor] (the slot indices)
        self._rows = {idx: torch.tensor(row) for idx, row in rows.items()}
        self.freed_reqs: List[int] = []

    @property
    def req_to_token(self):
        return self

    def __getitem__(self, key):
        idx, sl = key
        return self._rows[idx][sl]

    def free(self, req) -> None:
        self.freed_reqs.append(req.req_pool_idx)
        req.req_pool_idx = -1


class _ChunkCacheStub:
    """Stand-in for ChunkCache: dec_lock_ref is a no-op, protected_size_ stays 0."""

    def __init__(self):
        self.protected_size_ = 0
        self.dec_lock_ref_calls: List[_FakeNode] = []

    def dec_lock_ref(self, node):
        # ChunkCache.dec_lock_ref is a no-op; record the call for assertions.
        self.dec_lock_ref_calls.append(node)


class _RadixCacheStub:
    """Stand-in for RadixCache: dec_lock_ref walks the path, subtracts
    len(node.key) from protected_size_."""

    def __init__(self):
        self.protected_size_ = 100  # arbitrary positive starting value
        self.dec_lock_ref_calls: List[_FakeNode] = []

    def dec_lock_ref(self, node):
        self.dec_lock_ref_calls.append(node)
        # Symmetric to RadixCache.inc_lock_ref / dec_lock_ref: subtract the
        # locked path's key lengths from protected_size_ (here the test
        # uses a single-node path for simplicity).
        if node is not None:
            self.protected_size_ -= node.key_len


@dataclass
class _FakeReq:
    rid: str
    req_pool_idx: int
    cache_protected_len: int = 0
    prefix_indices: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    last_node: Optional[_FakeNode] = None
    _kv_committed_len: int = 0
    kv_committed_freed: bool = False
    _overalloc: Tuple[int, int] = (0, 0)

    def pop_committed_kv_cache(self) -> int:
        assert (
            not self.kv_committed_freed
        ), "pop_committed_kv_cache called twice for the same req"
        self.kv_committed_freed = True
        return self._kv_committed_len

    def pop_overallocated_kv_cache(self) -> Tuple[int, int]:
        return self._overalloc


def _make_manager(allocator, req_pool, tree_cache, *, page_size: int = 1):
    """Construct a DecodeKVCacheOffloadManager-shaped object with only the
    fields _release_finished_req needs. We avoid the heavy real constructor
    so the unit test stays a pure-Python checker."""

    from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
        DecodeKVCacheOffloadManager,
        OffloadedState,
    )

    mgr = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
    mgr.token_to_kv_pool_allocator = allocator
    mgr.req_to_token_pool = req_pool
    mgr.tree_cache = tree_cache
    mgr.offloaded_state = {}
    mgr.page_size = page_size
    return mgr, OffloadedState


# --- Tests --------------------------------------------------------------


class TestReleaseFinishedReqChunkCacheCompat(unittest.TestCase):
    """ChunkCache mode (radix disabled): existing behavior must be preserved
    — frees [start_offset:kv_committed_len], dec_lock_ref is a no-op."""

    def test_release_frees_request_private_tail(self):
        allocator = _FakeAllocator()
        req_pool = _FakeReqPool({0: list(range(20))})
        tree_cache = _ChunkCacheStub()
        mgr, _ = _make_manager(allocator, req_pool, tree_cache)

        req = _FakeReq(
            rid="r-chunk",
            req_pool_idx=0,
            cache_protected_len=0,  # chunk-cache mode
            prefix_indices=torch.empty(0),
            last_node=_FakeNode(key_len=0),
            _kv_committed_len=16,
        )

        mgr._release_finished_req(req, start_offset=4)

        # The free should cover [4:16], i.e. slots 4..15.
        self.assertEqual(len(allocator.freed), 1)
        self.assertEqual(allocator.freed[0], list(range(4, 16)))
        # req_to_token_pool.free(req) was called
        self.assertEqual(req_pool.freed_reqs, [0])
        # dec_lock_ref(req.last_node) was called once (no-op for chunk cache)
        self.assertEqual(len(tree_cache.dec_lock_ref_calls), 1)
        self.assertIs(tree_cache.dec_lock_ref_calls[0], req.last_node)
        # protected_size_ is unchanged (ChunkCache.dec_lock_ref is a no-op)
        self.assertEqual(tree_cache.protected_size_, 0)


class TestReleaseFinishedReqRadixCacheCompat(unittest.TestCase):
    """Radix mode (--disaggregation-decode-enable-radix-cache): the
    [0:cache_protected_len] prefix region must be left to the radix tree."""

    def test_radix_skips_tree_owned_prefix(self):
        """Frees only [max(start_offset, cache_protected_len):kv_committed_len]
        — the tree-owned prefix region is never freed via allocator.free."""
        allocator = _FakeAllocator()
        req_pool = _FakeReqPool({1: list(range(100, 132))})
        tree_cache = _RadixCacheStub()
        mgr, _ = _make_manager(allocator, req_pool, tree_cache)

        # Request committed 32 tokens with first 16 in the radix tree
        # (cache_protected_len=16). state.prefill_len + state.inc_len = 8
        # (the offload watermark, which is BEFORE the tree boundary).
        node = _FakeNode(key_len=16)
        req = _FakeReq(
            rid="r-radix",
            req_pool_idx=1,
            cache_protected_len=16,
            prefix_indices=torch.arange(16),
            last_node=node,
            _kv_committed_len=32,
        )

        mgr._release_finished_req(req, start_offset=8)

        # The free should cover [16:32], NOT [8:32] — the tree-owned region
        # [0:16] is skipped.
        self.assertEqual(len(allocator.freed), 1)
        self.assertEqual(allocator.freed[0], list(range(116, 132)))
        # dec_lock_ref(req.last_node) was called — releasing the lock the
        # tree held on the matched prefix node.
        self.assertEqual(len(tree_cache.dec_lock_ref_calls), 1)
        self.assertIs(tree_cache.dec_lock_ref_calls[0], node)
        # protected_size_ decreased by len(node.key_len) — proper accounting
        # rather than the buggy len(req.prefix_indices) subtract.
        self.assertEqual(tree_cache.protected_size_, 100 - 16)

    def test_radix_no_free_when_start_offset_beyond_committed(self):
        """If start_offset >= kv_committed_len (e.g. all incremental KV was
        offloaded already), no incremental free fires."""
        allocator = _FakeAllocator()
        req_pool = _FakeReqPool({2: list(range(50, 64))})
        tree_cache = _RadixCacheStub()
        mgr, _ = _make_manager(allocator, req_pool, tree_cache)

        node = _FakeNode(key_len=8)
        req = _FakeReq(
            rid="r-no-tail",
            req_pool_idx=2,
            cache_protected_len=8,
            prefix_indices=torch.arange(8),
            last_node=node,
            _kv_committed_len=14,
        )

        # start_offset already past committed (e.g. offload finished
        # everything and is calling us redundantly): the start = max(14, 8)
        # == 14 == end, so the free range is empty.
        mgr._release_finished_req(req, start_offset=14)

        self.assertEqual(allocator.freed, [])
        self.assertEqual(len(tree_cache.dec_lock_ref_calls), 1)


class TestReleaseFinishedReqDoubleCallGuard(unittest.TestCase):
    """Re-entry into _release_finished_req for the same req must be a
    no-op (the previous implementation re-popped committed KV and
    triggered AssertionError, crashing the scheduler)."""

    def test_second_call_is_noop(self):
        allocator = _FakeAllocator()
        req_pool = _FakeReqPool({3: list(range(0, 16))})
        tree_cache = _RadixCacheStub()
        mgr, _ = _make_manager(allocator, req_pool, tree_cache)

        node = _FakeNode(key_len=4)
        req = _FakeReq(
            rid="r-double",
            req_pool_idx=3,
            cache_protected_len=4,
            prefix_indices=torch.arange(4),
            last_node=node,
            _kv_committed_len=12,
        )

        mgr._release_finished_req(req, start_offset=4)
        # First call: frees [4:12], 1 dec_lock_ref, 1 req_pool free
        self.assertEqual(len(allocator.freed), 1)
        self.assertEqual(len(tree_cache.dec_lock_ref_calls), 1)
        self.assertEqual(req_pool.freed_reqs, [3])
        self.assertTrue(req.kv_committed_freed)

        # Second call: must be a complete no-op (no extra allocator.free,
        # no extra dec_lock_ref, no extra req_pool.free, no
        # pop_committed_kv_cache AssertionError).
        mgr._release_finished_req(req, start_offset=4)
        self.assertEqual(len(allocator.freed), 1)
        self.assertEqual(len(tree_cache.dec_lock_ref_calls), 1)
        self.assertEqual(req_pool.freed_reqs, [3])


class TestReleaseFinishedReqOverallocFree(unittest.TestCase):
    """Speculative-decoding v2 over-allocation slots are freed in addition
    to the [start_offset:kv_committed_len] range; the page_size alignment
    still applies."""

    def test_overalloc_freed_alongside_tail(self):
        allocator = _FakeAllocator()
        req_pool = _FakeReqPool({4: list(range(200, 220))})
        tree_cache = _ChunkCacheStub()
        mgr, _ = _make_manager(allocator, req_pool, tree_cache, page_size=1)

        req = _FakeReq(
            rid="r-overalloc",
            req_pool_idx=4,
            cache_protected_len=0,  # chunk-cache mode
            prefix_indices=torch.empty(0),
            last_node=_FakeNode(key_len=0),
            _kv_committed_len=12,
            _overalloc=(12, 16),  # 4 extra over-allocated slots
        )

        mgr._release_finished_req(req, start_offset=0)

        # First call freed [0:12] (committed), then [12:16] (overalloc).
        self.assertEqual(len(allocator.freed), 2)
        self.assertEqual(allocator.freed[0], list(range(200, 212)))
        self.assertEqual(allocator.freed[1], list(range(212, 216)))


if __name__ == "__main__":
    unittest.main()
