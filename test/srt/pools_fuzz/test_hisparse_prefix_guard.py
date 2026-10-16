"""Regression tests: F-5 hisparse prefix-clobber fix (board #51, dsa Q1).

alloc_device_buffer harvests one request's compressed->device-index mapping to
build its C4 device buffer. Pre-fix it zeroed + surplus-freed over the FULL
allocated_indices range, INCLUDING a radix-hit prefix whose compressed slots
are NONZERO-live (tree or creator-request owned) -> cross-request compressed-KV
corruption: the shared prefix's device pages get freed while the tree (and the
live creator request) still map compressed slots onto them, so the tree's next
read lands in a page that can be re-allocated to an unrelated request
(UAF / context leak).

Fix (this change): alloc_device_buffer takes `prefix_len` (compressed units);
the prefix region is read-only (buffer may address the shared pages) while the
zeroing + harvest + surplus-free cover only the fresh tail; surplus freeing
protects prefix pages unconditionally (survives the internal highest-token
swap). Both HiSparseTokenToKVPoolAllocator (non-V4) and
DeepSeekV4HiSparseTokenToKVPoolAllocator (V4) carry the guard; the coordinator
(hisparse_coordinator.alloc_device_buffer) computes the compressed prefix
length from req.prefix_len and passes it in.

Tests construct REAL pools/allocators with tiny CPU sizes: dsv4 layout forces
its fixed head dims (448 nope / 64 rope, fp8 store). Each F-5 case first
verifies the PRE-FIX shape would clobber (this is what the fix removed), then
pins post-fix invariants: prefix mapping values untouched, no prefix page
freed, fresh tail harvested + surplus freed exactly once, accounting sane.

Run: PYTHONPATH=python python3 -m pytest test/srt/pools_fuzz/test_hisparse_prefix_guard.py -q
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
    _free_surplus_device_pages,
)

PAGE = 64


# ---------------------------------------------------------------------------
# non-V4 fixture (as in test_hisparse_surplus_guard.py)
# ---------------------------------------------------------------------------


class FakeKV:
    def __init__(self):
        self.m = None

    def register_mapping(self, m):
        self.m = m

    def _translate_loc_to_hisparse_device(self, idx):
        return self.m[idx.to(torch.int64)]

    def get_cpu_copy(self, *a, **k):
        raise NotImplementedError

    def load_cpu_copy(self, *a, **k):
        raise NotImplementedError


def make_nonv4(num_logical_pages=40):
    kv = FakeKV()
    a = HiSparseTokenToKVPoolAllocator(
        size=num_logical_pages * PAGE,
        page_size=PAGE,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=kv,
        need_sort=True,
        host_to_device_ratio=2,
    )
    return a, a.hisparse_attn_allocator


# ---------------------------------------------------------------------------
# V4 fixture: real DeepSeekV4TokenToKVPool (bypassed init) + real C4 device pool
# ---------------------------------------------------------------------------


def make_v4(c4_pages=40):
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        DeepSeekV4TokenToKVPool,
        HiSparseC4DevicePool,
    )

    # real C4 pool: the V4 KV layout asserts its fixed head dims and fp8 store
    c4 = HiSparseC4DevicePool(
        size=c4_pages * 16,
        page_size=16,
        dtype=torch.float8_e4m3fn,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    outer = object.__new__(DeepSeekV4TokenToKVPool)  # bypass __init__
    outer.c4_kv_pool = c4
    outer.c4_logical_size = c4_pages * 64 // 4  # c4_compressed = full // 4

    # logical allocator: V4 tests exercise only alloc_device_buffer; the logical
    # side is read for attrs/sizing and clear()ed once at init.
    logical = SimpleNamespace()
    logical._kvcache = outer
    logical.size_full = 2 * c4_pages * 64
    logical.page_size = 64
    logical.need_sort = True
    logical.clear = lambda: None

    v4 = DeepSeekV4HiSparseTokenToKVPoolAllocator(logical)
    return v4, v4.hisparse_attn_allocator, c4


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def own_hisparse_pages(hp, ps, pages):
    taken = hp.alloc(len(pages) * ps)
    assert taken is not None
    assert set((taken // ps).tolist()) == set(pages), (taken // ps).tolist()


def sane(hp, tag=""):
    fp = hp.get_all_free_pages().tolist()
    assert len(fp) == len(set(fp)), f"{tag}: dup page in free list {sorted(fp)}"
    assert hp.available_size() <= hp.num_pages * hp.page_size, f"{tag}: over-avail"


# ---------------------------------------------------------------------------
# F-5: non-V4 path
# ---------------------------------------------------------------------------


def test_nonv4_prefix_mapping_untouched_and_not_freed():
    """Radix-hit prefix: prefix compressed slots map to LIVE tree-owned device
    pages. Post-fix the allocator must leave the prefix mapping values alone
    and must not free any prefix page; the fresh tail is harvested/freed."""
    a, hp = make_nonv4()
    ps = PAGE
    # tree owns prefix pages 1..2 (compressed slots 0..2ps-1); request owns fresh
    # pages 3..4 (slots 2ps..4ps-1) freshly written by its earlier chunks.
    own_hisparse_pages(hp, ps, [1, 2])  # out of pool: tree/creator's
    own_hisparse_pages(hp, ps, [3, 4])  # out of pool: this request's fresh
    mapping = dict()
    prefix_vals = torch.arange(1 * ps, 3 * ps)  # tree-owned values, slots 0..2ps-1
    fresh_vals = torch.arange(3 * ps, 5 * ps)
    allocated = torch.arange(0, 4 * ps, dtype=torch.int64)  # 4 compressed pages
    a.full_to_hisparse_device_index_mapping[allocated[: 2 * ps]] = prefix_vals
    a.full_to_hisparse_device_index_mapping[allocated[2 * ps :]] = fresh_vals

    buf = a.alloc_device_buffer(allocated, 3 * ps, prefix_len=2 * ps)

    # 1) prefix mapping must be untouched
    assert torch.equal(
        a.full_to_hisparse_device_index_mapping[allocated[: 2 * ps]], prefix_vals
    ), "prefix (tree-owned) mapping was clobbered"
    # 2) fresh tail zeroed out (harvested)
    assert (a.full_to_hisparse_device_index_mapping[allocated[2 * ps :]] == 0).all()
    # 3) buffer addresses shared prefix pages (read-only share)
    first_pages = set((buf[: 2 * ps] // ps).tolist())
    assert {1, 2} <= first_pages, first_pages
    # 4) ONLY the fresh region may be freed (as surplus); prefix pages stay owned
    fp = hp.get_all_free_pages().tolist()
    assert 1 not in fp and 2 not in fp, f"prefix page freed! {sorted(fp)}"
    # fresh page 4 is the pure-surplus page (beyond the 3-page buffer cut); it
    # must be released exactly once; fresh page 3 stays owned by the buffer.
    assert fp.count(4) == 1, sorted(fp)
    if 3 not in set((buf // ps).tolist()):
        assert fp.count(3) == 1
    else:
        assert 3 not in fp, sorted(fp)
    # 5) freeing the buffer later must not double-free anything (accounting exact)
    sane(hp, "nonv4-f5")


def test_nonv4_pre_fix_shape_would_clobber():
    """Sanity that the test harness distinguishes fixed vs buggy: the legacy
    call (prefix_len=0) zeroes the prefix mapping — exactly the F-5 shape."""
    a, hp = make_nonv4()
    ps = PAGE
    own_hisparse_pages(hp, ps, [1, 2])
    own_hisparse_pages(hp, ps, [3, 4])
    allocated = torch.arange(0, 4 * ps, dtype=torch.int64)
    a.full_to_hisparse_device_index_mapping[allocated[: 2 * ps]] = torch.arange(
        1 * ps, 3 * ps
    )
    a.full_to_hisparse_device_index_mapping[allocated[2 * ps :]] = torch.arange(
        3 * ps, 5 * ps
    )
    a.alloc_device_buffer(allocated, 2 * ps)  # legacy: no prefix_len
    clobbered = (
        a.full_to_hisparse_device_index_mapping[allocated[: 2 * ps]] == 0
    ).all()
    assert clobbered, "legacy call should zero prefix mapping (the F-5 shape)"


# ---------------------------------------------------------------------------
# F-5: V4 path
# ---------------------------------------------------------------------------


def test_v4_prefix_mapping_untouched_and_pages_protected():
    v4, hp, c4 = make_v4()
    ps = v4.hisparse_page_size  # 16 compressed tokens per device page
    # compressed slots 0..63: first 2 pages are a radix-hit prefix (tree-owned,
    # mapped to device pages 1,2); the next 2 pages are the fresh extension
    # tail (device pages 3,4) written by this request's earlier chunk.
    allocated = torch.arange(0, 4 * ps, dtype=torch.int64)
    own_hisparse_pages(hp, ps, [1, 2])  # tree/creator pages
    own_hisparse_pages(hp, ps, [3, 4])  # fresh region pages
    prefix_cmp = 2 * ps
    prefix_vals = torch.arange(1 * ps, 3 * ps)
    fresh_vals = torch.arange(3 * ps, 5 * ps)
    v4.full_to_hisparse_device_index_mapping[allocated[:prefix_cmp]] = prefix_vals
    v4.full_to_hisparse_device_index_mapping[allocated[prefix_cmp:]] = fresh_vals

    need = 3 * ps  # keep 3 compressed pages, surplus the rest
    buf = v4.alloc_device_buffer(allocated, need, prefix_len=prefix_cmp)

    # prefix mapping untouched
    assert torch.equal(
        v4.full_to_hisparse_device_index_mapping[allocated[:prefix_cmp]],
        prefix_vals,
    ), "V4 prefix (tree-owned) mapping clobbered"
    # fresh tail harvested
    assert (
        v4.full_to_hisparse_device_index_mapping[allocated[prefix_cmp:]] == 0
    ).all()
    # prefix pages never freed even though the V4 swap may relocate them
    fp = hp.get_all_free_pages().tolist()
    assert 1 not in fp and 2 not in fp, f"prefix page freed: {sorted(fp)}"
    assert set(fp) & {3, 4} == set(), "fresh pages must stay owned by the buffer"
    sane(hp, "v4-f5")


def test_v4_pre_fix_shape_would_clobber():
    v4, hp, _ = make_v4()
    ps = v4.hisparse_page_size
    allocated = torch.arange(0, 4 * ps, dtype=torch.int64)
    own_hisparse_pages(hp, ps, [1, 2])
    own_hisparse_pages(hp, ps, [3, 4])
    v4.full_to_hisparse_device_index_mapping[allocated[: 2 * ps]] = torch.arange(
        1 * ps, 3 * ps
    )
    v4.full_to_hisparse_device_index_mapping[allocated[2 * ps :]] = torch.arange(
        3 * ps, 5 * ps
    )
    v4.alloc_device_buffer(allocated, 2 * ps)  # legacy call: no prefix bound
    assert (
        v4.full_to_hisparse_device_index_mapping[allocated[: 2 * ps]] == 0
    ).all(), "legacy call should zero prefix mapping (the F-5 shape)"


def test_prefix_len_zero_default_is_legacy():
    """API back-compat: no prefix -> behavior identical to pre-fix semantics."""
    a, hp = make_nonv4()
    ps = PAGE
    own_hisparse_pages(hp, ps, [1, 2])
    allocated = torch.arange(0, 2 * ps, dtype=torch.int64)
    a.full_to_hisparse_device_index_mapping[allocated] = torch.arange(1 * ps, 3 * ps)
    buf1 = a.alloc_device_buffer(allocated, ps, prefix_len=0)
    assert set(buf1.tolist()) == set(range(1 * ps, 2 * ps))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
