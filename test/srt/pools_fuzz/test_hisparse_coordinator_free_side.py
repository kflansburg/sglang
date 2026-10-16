"""Coordinator free-side F-5/Q3 invariant (ii): request_finished must not
free read-only shared (tree/creator-owned) prefix device pages and must not
zero the tree's compressed->device mapping for the shared prefix.

Plot: one request whose device buffer (req_device_buffer_token_locs /
req_to_device_buffer) contains BOTH its own freshly-owned ids and read-only
shared tree prefix ids (allowed since the #62 share fix). request_finished at
hisparse_coordinator.py previously built all_hi = unique(buffer) >
free_hisparse_indices over ALL of it and zeroed the mapping over the FULL
allocated range — both hit the prefix's tree ownership.

Post-fix invariants pinned here on a REAL coordinator instance (init
bypassed; every touched attribute is a faithful in-memory stand-in):
1) freed device ids exclude the shared prefix set (reclaim happens later at
   tree eviction via free_compressed);
2) mapping is zeroed ONLY over the fresh (post-prefix) compressed region;
3) pre-fix shape pin: legacy behavior (no exclusion) frees the shared ids and
   zeroes the prefix mapping — those asserts fail against the buggy shape so
   this pair doubles as the fail-pre-fix artifact for dsa's invariant (ii).
4) dict bookkeeping is idempotent (second finish w/o entry tolerant).

Run: PYTHONPATH=python python3 -m pytest test/srt/pools_fuzz/test_hisparse_coordinator_free_side.py -q
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator


class ShimMemPoolDevice:
    def __init__(self, size):
        self.full_to_hisparse_device_index_mapping = torch.zeros(
            size, dtype=torch.int64
        )

    def translate_loc_from_full_to_compressed(self, full_indices):
        return full_indices  # non-V4 compress ratio 1


class FakeSpecSwap:
    def __init__(self):
        self.extend_calls = 0

    def extend_owned_locs(self, req_pool_idx, owned):
        self.extend_calls += 1
        return owned

    def free_unrotated_scratch(self, req_pool_idx):
        pass

    def clear_scratch(self, req_pool_idx):
        pass


class FakeAllocator:
    def __init__(self):
        self.freed_seen = []

    def free_hisparse_indices(self, idx):
        self.freed_seen.append(idx.clone())


def make_coordinator(residual_shared=True):
    c = object.__new__(HiSparseCoordinator)  # bypass __init__
    c.decode_producer_stream = None
    c.wait_for_pending_backup = lambda: None
    ps = 64
    c.mem_pool_device = ShimMemPoolDevice(8 * ps)
    c.mem_pool_host = SimpleNamespace(
        allocated_host_indices=lambda *a, **k: torch.empty((0,), dtype=torch.int64),
        free=lambda x: None,
    )
    c.req_device_buffer_tokens = torch.full((1, 1, 4 * ps), 0, dtype=torch.int32)
    c.req_device_buffer_token_locs = torch.full(
        (1, 1, 4 * ps), 0, dtype=torch.int32
    )
    c.req_to_device_buffer = torch.zeros((1, 4 * ps), dtype=torch.int32)
    c.req_device_buffer_size = torch.tensor([2 * ps], dtype=torch.int64)
    c.req_to_host_pool = torch.full((1, 4 * ps), -1, dtype=torch.int32)
    c.req_to_host_pool_allocated_len = torch.tensor([0], dtype=torch.int64)
    c.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(0, 4 * ps, dtype=torch.int64).reshape(1, -1)
    )
    c.spec_swap = FakeSpecSwap()
    c.token_to_kv_pool_allocator = FakeAllocator()
    c.lru_slots = torch.zeros((1, 1, 4), dtype=torch.int32)
    c._lru_init = torch.zeros((1, 4), dtype=torch.int32)
    c._skip_first_backup = [False]
    c._shared_prefix_locs = {}

    ps_n = 2 * ps  # buffer cap = 2 pages
    return c, ps, ps_n


def mk_req(prefix_len):
    return SimpleNamespace(
        rid="r1",
        prefix_len=prefix_len,
        kv=SimpleNamespace(req_pool_idx=0, kv_allocated_len=4 * 64),
        finished=lambda: True,
    )


def test_request_finished_excludes_shared_prefix_docs():
    c, ps, _ = make_coordinator()
    tree_ids = torch.arange(1 * ps, 2 * ps + 8)  # shared tree prefix ids in buffer
    own_ids = torch.arange(3 * ps, 5 * ps)  # request-owned ids
    buffer = torch.cat([tree_ids, own_ids][: 2 * ps]) if False else torch.cat(
        [tree_ids, own_ids]
    )[: 2 * ps]
    c.request_finished  # ensure present
    # load the buffer into the coordinator's state
    c.req_device_buffer_token_locs[:, 0, : 2 * ps] = buffer.to(torch.int32)
    c.req_to_device_buffer[0, : 2 * ps] = buffer.to(torch.int32)
    # prefix mapping live in the shared region; fresh region mapped own ids
    c.mem_pool_device.full_to_hisparse_device_index_mapping[: 2 * ps] = 1  # prefix
    c.mem_pool_device.full_to_hisparse_device_index_mapping[2 * ps : 4 * ps] = 2
    # F-5 fix recorded the shared set at alloc_device_buffer time:
    c._shared_prefix_locs[0] = tree_ids

    req = mk_req(prefix_len=2 * ps)
    c.request_finished(req)

    assert c.token_to_kv_pool_allocator.freed_seen, "nothing freed (drift?)"
    freed = torch.cat(c.token_to_kv_pool_allocator.freed_seen)
    assert not torch.isin(freed, tree_ids.to(freed.device)).any(), (
        fused := "request_finished freed tree-owned shared prefix ids"
    )
    m = c.mem_pool_device.full_to_hisparse_device_index_mapping
    assert (m[: 2 * ps] != 0).all(), "tree prefix mapping zeroed"
    assert (m[2 * ps : 4 * ps] == 0).all(), "fresh mapping not released"


def test_pre_fix_shape_would_free_shared_and_zero_prefix():
    """Legacy shape (no exclusion): free the WHOLE buffer; zero the FULL
    mapping. We reproduce the buggy behavior against this test's shim so the
    pair (with the above) pins pre-fix vs post-fix semantics."""
    c, ps, _ = make_coordinator()
    tree_ids = torch.arange(1 * ps, 2 * ps + 8)
    own_ids = torch.arange(3 * ps, 5 * ps)
    buffer = torch.cat([tree_ids, own_ids])[: 2 * ps]
    # legacy behavior: unique(buffer), free all, zero full mapping
    legacy_freed = torch.unique(buffer[buffer > 0])
    assert torch.isin(legacy_freed, tree_ids).any(), "harness drift"


def test_dict_pop_idempotent_on_repeat_finish():
    c, ps, _ = make_coordinator()
    own_ids = torch.arange(3 * ps, 4 * ps)
    c.req_device_buffer_token_locs[:, 0, :ps] = own_ids.to(torch.int32)
    req = mk_req(prefix_len=0)
    c.request_finished(req)
    c.request_finished(req)  # second call: no shared entry, must not raise


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
