"""Regression tests: hisparse alloc_device_buffer surplus release under
NON-DENSE device indices within a run (dsa-auditor Q2 port).

Ports the V4 pure_surplus-page guard to the non-V4 path and unifies both via
_free_surplus_device_pages: a page split between the kept buffer and the
surplus must stay with the buffer; only pure-surplus pages are released.

BUG SHAPE FIXED: non-V4 alloc_device_buffer used to release
``hisparse_indices[need_size:]`` at TOKEN granularity
(free -> unique(//page)): if one page held both kept and surplus tokens (the
request's device indices are not dense within a run — e.g. retract-resume
reallocating a different last partial page), the whole page was released from
under the live buffer -> silent use-after-free / cross-request C4 KV takeover.

Run: PYTHONPATH=python python3 -m pytest test/srt/pools_fuzz/test_hisparse_surplus_guard.py -q
"""

import pytest
import torch

from sglang.srt.mem_cache.allocator.hisparse import (
    HiSparseTokenToKVPoolAllocator,
    _free_surplus_device_pages,
)

PAGE = 64


class FakeKV:
    """Minimal kvcache protocol used by the hissparse allocator paths."""

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


def make_alloc(num_logical_pages=40, ratio=2, need_sort=True):
    kv = FakeKV()
    a = HiSparseTokenToKVPoolAllocator(
        size=num_logical_pages * PAGE,
        page_size=PAGE,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=kv,
        need_sort=need_sort,
        host_to_device_ratio=ratio,
    )
    return a, a.hisparse_attn_allocator


def hisparse_free_pages(hp):
    return set(hp.get_all_free_pages().tolist())


def _accounting_sane(hp, tag=""):
    fp = hp.get_all_free_pages().tolist()
    assert len(fp) == len(set(fp)), f"{tag}: dup page in hisparse free list {sorted(fp)}"
    assert hp.available_size() <= hp.num_pages * PAGE, f"{tag}: over-availability"


def own_hisparse_pages(hp, pages):
    """Take whole hisparse pages out of the free list as a previous alloc would."""
    taken = hp.alloc(len(pages) * PAGE)
    assert taken is not None
    assert set((taken // PAGE).tolist()) == set(pages), (
        set((taken // PAGE).tolist()), pages
    )
    return taken


@pytest.mark.parametrize("need_sort", [True, False])
def test_pure_surplus_released_dense_case(need_sort):
    """Dense indices (classical case): surplus pages free exactly once."""
    a, hp = make_alloc(need_sort=need_sort)
    own_hisparse_pages(hp, [1, 2, 3, 4])  # request's private device pages
    logical = torch.arange(1, 4 * PAGE + 1, dtype=torch.int64)  # 4 logical pages
    his = torch.arange(1 * PAGE, 5 * PAGE, dtype=torch.int64)   # maps to dev pages 1..4
    a.full_to_hisparse_device_index_mapping[logical] = his
    buf = a.alloc_device_buffer(logical, 2 * PAGE)  # need 2 pages; surplus pages 3,4
    assert set(buf.tolist()) == set(range(1 * PAGE, 3 * PAGE))
    fp = hp.get_all_free_pages().tolist()
    assert 3 in fp and 4 in fp
    assert fp.count(3) == 1 and fp.count(4) == 1
    _accounting_sane(hp, "dense")


@pytest.mark.parametrize("need_sort", [True, False])
def test_non_dense_shared_page_not_freed(need_sort):
    """NON-DENSE within a run: one device page holds both a kept token and a
    surplus token. The page must NOT be freed; only the pure-surplus page goes."""
    a, hp = make_alloc(need_sort=need_sort)
    ps = PAGE
    # device indices interleaving page runs at the buffer cut:
    #   kept:  p1[0..ps-2] + [p2_first_token]  (one whole page of tokens, two pages)
    #   surplus: [p1_last_token] + p3[0..ps-1]
    # p1 is SHARED kept/surplus at the cut; p3 is pure surplus; p2 stays owned by
    # the buffer through its first token.
    own_hisparse_pages(hp, [1, 2, 3])
    p1 = torch.arange(1 * ps, 2 * ps)
    p2_first = torch.tensor([2 * ps])
    p1_last = torch.tensor([2 * ps - 1])
    p3 = torch.arange(3 * ps, 4 * ps)
    kept = torch.cat([p1[:-1], p2_first])      # len ps
    surplus = torch.cat([p1_last, p3[:-1]])     # len 1 + (ps - 1) = ps
    allidx = torch.cat([kept, surplus])
    logical = torch.arange(1, 2 * ps + 1, dtype=torch.int64)
    a.full_to_hisparse_device_index_mapping[logical] = allidx
    buf = a.alloc_device_buffer(logical, 1 * ps)
    assert set(buf.tolist()) == set(kept.tolist())
    fp = hp.get_all_free_pages().tolist()
    assert 1 not in fp, "shared page 1 was freed from under the buffer!"
    assert 3 in fp and fp.count(3) == 1, "pure surplus page 3 not released exactly once"
    assert 2 not in fp, "page 2 was never owned by this request"
    _accounting_sane(hp, "non-dense")


@pytest.mark.parametrize("need_sort", [True, False])
def test_token_granular_old_behavior_would_corrupt(need_sort):
    """Pin the fixed behavior: the old token-granular free would have released
    shared page 1 in the non-dense case; the unified guard keeps it."""
    _, hp = make_alloc(need_sort=need_sort)
    ps = PAGE
    own_hisparse_pages(hp, [1])
    kept = torch.arange(1 * ps, 2 * ps - 1)  # p1[0..ps-2]
    surplus = torch.tensor([2 * ps - 1])     # p1's last token
    before = hp.available_size()
    _free_surplus_device_pages(hp, kept, surplus, ps)
    after = hp.available_size()
    assert after == before, "shared-only surplus must not free any page"
    assert 1 not in set(hp.get_all_free_pages().tolist())


@pytest.mark.parametrize("need_sort", [True, False])
def test_surplus_then_buffer_free_no_double_free(need_sort):
    """Free-order safety: surplus released first, then the buffer's own free
    must not double-free any page and accounting stays exact."""
    a, hp = make_alloc(need_sort=need_sort)
    ps = PAGE
    own_hisparse_pages(hp, [1, 2, 3, 4])
    logical = torch.arange(1, 4 * ps + 1)
    a.full_to_hisparse_device_index_mapping[logical] = torch.arange(
        1 * ps, 5 * ps, dtype=torch.int64)
    buf = a.alloc_device_buffer(logical, 2 * ps)
    avail_after_surplus = hp.available_size()
    # now the buffer owner releases its device tokens (its own free path)
    a.free_hisparse_indices(buf)
    avail_after_full = hp.available_size()
    assert avail_after_full == (avail_after_surplus + 2 * ps), (
        avail_after_surplus, avail_after_full
    )
    _accounting_sane(hp, "post-free")


def test_unit_helper_empty_surplus():
    _, hp = make_alloc()
    avail = hp.available_size()
    _free_surplus_device_pages(hp, torch.arange(0, PAGE), torch.empty((0,), dtype=torch.int64), PAGE)
    assert hp.available_size() == avail


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
