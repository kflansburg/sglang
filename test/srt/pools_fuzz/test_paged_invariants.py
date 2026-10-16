"""Invariant checks for PagedTokenToKVPoolAllocator (real code, CPU).

1. free_pages storage invariant (PSA): `free_pages` / `staged_pages` must only
   ever hold PAGE IDs. Values or duplicates here propagate straight into KV
   indices. Regression for the PSA part of audit note pool_findings_alloc BUG-ALLOC-6.

2. BUG-ALLOC-7 / BUG-ALLOC-8: `free_page_ids` / `_release_page_ids` trust their
   input ("Free exactly these pages; no page twice, no dedup") — an out-of-range
   or duplicated page id lands in the free list and is materialized by alloc() as
   KV indices for fabricated / doubly-owned pages. These are internal API seams
   (no production caller found at HEAD that can supply bad ids without a caller
   bug upstream); the tests lock the seam's behavior.
"""

import sys

import torch

sys.path.insert(0, "python")
from sglang.srt.mem_cache.allocator.paged import (  # noqa: E402
    PagedTokenToKVPoolAllocator,
)


def build(num_pages=8, ps=4, need_sort=False):
    return PagedTokenToKVPoolAllocator(
        size=num_pages * ps,
        page_size=ps,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=None,
        need_sort=need_sort,
    )


def test_storage_is_page_ids_under_valid_traffic():
    for need_sort in (True, False):
        a = build(need_sort=need_sort)
        out = a.alloc(3 * a.page_size)
        assert out is not None
        a.free(out[:1])
        a.alloc_extend(
            torch.tensor([0]), torch.tensor([0]),
            torch.tensor([7]), torch.tensor([7]), torch.tensor([-1]), 7,
        )
        a.alloc_decode(torch.tensor([8]), torch.tensor([8]), torch.tensor([7]))
        a.merge_and_sort_free()
        vals = a.free_pages.tolist()
        staged = [x for t in getattr(a, "staged_pages", []) for x in t.tolist()]
        for v in vals + staged:
            assert 0 < v <= a.num_pages, f"non-page-id in storage: {v}"


def test_free_page_ids_out_of_range_poison(need_sort=False):
    """BUG-ALLOC-7: free_page_ids accepting num_pages+K fabricates KV slots."""
    a = build(num_pages=8, ps=4, need_sort=need_sort)
    a.free_page_ids(torch.tensor([9]))
    assert 9 in a.free_pages.tolist(), "out-of-range id rejected? (desired)"
    out = a.alloc(4)
    # if the poisoned id was kept, alloc materialized slots 36..39 for size-32 pool
    print("need_sort=False alloc after free_page_ids([9]):", out.tolist())


def test_free_page_ids_duplicate(need_sort=False):
    """BUG-ALLOC-8: duplicate ids in one free_page_ids call stay duplicated."""
    a = build(num_pages=8, ps=4, need_sort=need_sort)
    a.free_page_ids(torch.tensor([3, 3]))
    n3 = a.free_pages.tolist().count(3)
    assert n3 >= 1
    print(f"need_sort=False count(3) after [3,3]: {n3} (2 would double-alloc)")


def test_free_token_vector_out_of_range_and_dup(need_sort=True):
    """free() with a >size token index: int64 div floors, OOB index lands as page id."""
    a = build(num_pages=8, ps=4, need_sort=need_sort)
    a.free(torch.tensor([33, 34]))  # slot 33,34: page 8,8 in a 8-page pool (1..8): page 8 valid, try 40
    a.free(torch.tensor([400]))     # page 100 fabricated
    vals = a.free_pages.tolist() + [x for t in getattr(a, "staged_pages", []) for x in t.tolist()]
    print("need_sort=True storage after free([33,34,400]):", vals)
    assert 100 in vals, "OOB page id not yet visible (staged)"


if __name__ == "__main__":
    test_storage_is_page_ids_under_valid_traffic()
    test_free_page_ids_out_of_range_poison()
    test_free_page_ids_out_of_range_poison.__wrapped__ if False else None
    a = build()
    a.free_page_ids(torch.tensor([9]))
    out = a.alloc(4)
    print("BUG-ALLOC-7 (need_sort=False): free tail", a.free_pages[-4:].tolist(),
          "-> alloc values", out.tolist(), "pool slots", a.size)
    b = build()
    b.free_page_ids(torch.tensor([3, 3]))
    print("BUG-ALLOC-8 (need_sort=False): count(3) =", b.free_pages.tolist().count(3))
    c = build(need_sort=True)
    c.free(torch.tensor([400]))
    print("BUG-ALLOC-7 (need_sort=True): staged contains", [t.tolist() for t in c.staged_pages])
    test_storage_is_page_ids_under_valid_traffic()
    print("storage invariant OK")
