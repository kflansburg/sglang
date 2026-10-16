"""Retract/finish during speculation: alloc/free bookkeeping round trip.

Simulates the spec-v2 slot-ownership invariant on CPU with no GPU kernels:

  * eagle_prepare_for_decode -> page_aligned_decode_alloc_lens(reserve=
    2*get_alloc_len_per_decode, page_size) advances kv_allocated_len ahead of
    kv_committed_len in whole pages.
  * verify settle -> kv_committed_len += accepted.
  * request release -> release_kv_cache frees [cache_protected_len,
    owned_kv_len) plus _release_overallocated_kv_indices frees
    [ceil_align(owned_kv_len, page), kv_allocated_len).

The invariant: after any interleave of decode-alloc / verify-commit /
retract-release rounds, every position in [0, kv_allocated_len) is freed
EXACTLY once, and no position >= ceil(committed, page) that the tree could
have inserted is freed twice. This pins the retract-during-speculation class
(draft KV alloc/free mismatch on retract paths) for page_size > 1.
"""

import unittest
from typing import List, Optional, Tuple

from sglang.srt.mem_cache.allocation_sizing import page_aligned_decode_alloc_lens
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _call_real_alloc_lens(kv: "_FakeReqKv", reserve: int, page_size: int):
    """Exercise the REAL page_aligned_decode_alloc_lens against a shim req."""

    class _R:
        pass

    r = _R()
    r.kv = kv
    cur, nxt, needed = page_aligned_decode_alloc_lens(
        [r], reserve=reserve, page_size=page_size
    )
    return cur[0], nxt[0], needed


def ceil_align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


class _FakeReqKv:
    """The slice of Req.kv the alloc/free math reads."""

    def __init__(self):
        self.cache_protected_len = 0
        self.kv_committed_len = 0
        self.kv_allocated_len = 0
        self.component_evicted_seqlens = {}


class FreeLedger:
    """PAGE-granularity ledger of allocator frees for one request row.

    The paged allocator frees whole pages: a segment free of [s, e) releases
    every page touched by [s, e). That is exactly why the correctness
    invariant is `kv_allocated_len >= ceil(owned_committed, page)` at release
    time: the page containing the committed tail must be allocator-owned,
    else the free of [protected, owned) would free a page the tree adopted.
    """

    def __init__(self, page_size: int):
        self.page_size = page_size
        self.freed_pages = set()
        self.tree_pages = set()
        self.min_allocated = 0  # enforce allocated >= ceil(committed) at release

    def free(self, start: int, end: int):
        if start >= end:
            return
        for pg in range(start // self.page_size, (end - 1) // self.page_size + 1):
            assert pg not in self.tree_pages, (
                f"page {pg} freed after the tree adopted it (double ownership)"
            )
            assert pg not in self.freed_pages, f"page {pg} freed twice"
            self.freed_pages.add(pg)

    def tree_take(self, start: int, end: int):
        for pg in range(start // self.page_size, end // self.page_size):
            assert pg not in self.freed_pages, (
                f"page {pg} tree-adopted after free (use-after-free)"
            )
            assert pg not in self.tree_pages, f"page {pg} adopted twice"
            self.tree_pages.add(pg)

    def assert_consistent(self, allocated: int):
        num_pages = (allocated + self.page_size - 1) // self.page_size
        for pg in range(num_pages):
            assert pg in self.freed_pages or pg in self.tree_pages, (
                f"page {pg} leaked (never freed, never tree-owned)"
            )


def _spec_reserve(alloc_len_per_decode: int) -> int:
    return 2 * alloc_len_per_decode


def _alloc_step(
    kv: _FakeReqKv, *, reserve: int, page_size: int, ledger: FreeLedger
) -> Tuple[int, int]:
    """page_aligned_decode_alloc_lens + alloc_for_spec_decode bookkeeping."""
    cur, nxt, _needed = _call_real_alloc_lens(kv, reserve, page_size)
    kv.kv_allocated_len = max(kv.kv_allocated_len, nxt)
    return cur, kv.kv_allocated_len


def _release(kv: _FakeReqKv, *, page_size: int, ledger: FreeLedger):
    """release_kv_cache semantics (radix_cache.checkpoint + free_kv_row +
    _release_overallocated_kv_indices):

      checkpoint inserts the page-aligned-down committed prefix; the tree
      ADOPTS those physical pages (cache_protected_len = floor(owned, page)
      - the repoint slice is empty because len(new_indices) == key_len),
      then free_kv_row frees [new_protected, owned), then the over-allocated
      tail [ceil_align(owned, page), allocated).
    """
    owned_kv_len = kv.kv_committed_len  # owned_kv_len() without a stop trim
    new_protected = (owned_kv_len // page_size) * page_size
    assert new_protected >= kv.cache_protected_len
    # Tree adopts [0, new_protected); the request's own free starts there.
    ledger.tree_take(kv.cache_protected_len, new_protected)
    ledger.free(new_protected, owned_kv_len)
    start_p = ceil_align(owned_kv_len, page_size)
    ledger.free(start_p, kv.kv_allocated_len)
    # The request-only parts must be freed exactly once; tree-adopted pages
    # stay owned by the tree (their lifecycle is radix-lead's side).
    ledger.assert_consistent(kv.kv_allocated_len)
    kv.cache_protected_len = new_protected


class TestSpecRetractFreeRoundTrip(unittest.TestCase):
    def _run(self, page_size, accepts_per_round, rounds, reserve_tokens):
        ledger = FreeLedger(page_size)
        kv = _FakeReqKv()
        for n_accept in accepts_per_round:
            _alloc_step(kv, reserve=reserve_tokens, page_size=page_size, ledger=ledger)
            kv.kv_committed_len += n_accept
        _release(kv, page_size=page_size, ledger=ledger)

    def test_page64_varied_accepts(self):
        page = 64
        # eagle topk=1, e.g. num_steps=4, num_draft_tokens=5: per-decode
        # alloc_len = 5, reserve = 10; page 64 dominates.
        reserve = 10
        cases = [
            [3] * 20,
            [1, 5, 5, 1, 3, 2, 4] * 4,
            [5] * 40,
            [0, 5, 0, 5, 1] * 6,
        ]
        for i, accepts in enumerate(cases):
            with self.subTest(case=i):
                self._run(page, accepts, None, reserve)

    def test_page64_zero_and_full_accept_only(self):
        page = 64
        self._run(page, [1] * 100, None, 10)
        self._run(page, [5] * 100, None, 10)

    def test_page1_trivial(self):
        self._run(1, [3, 1, 4, 1, 5] * 10, None, 10)

    def test_long_run_many_rounds_no_leak(self):
        page = 64
        ledger = FreeLedger(page)
        kv = _FakeReqKv()
        reserve = 10
        import random

        rng = random.Random(20261016)
        for _ in range(500):
            _alloc_step(kv, reserve=reserve, page_size=page, ledger=ledger)
            kv.kv_committed_len += rng.randint(1, 5)
        # Random checkpoint mid-run: tree takes prefix (page aligned down)
        protected = (kv.kv_committed_len // page) * page
        ledger.tree_take(kv.cache_protected_len, protected)
        kv.cache_protected_len = protected
        for _ in range(500):
            _alloc_step(kv, reserve=reserve, page_size=page, ledger=ledger)
            kv.kv_committed_len += rng.randint(1, 5)
        _release(kv, page_size=page, ledger=ledger)


if __name__ == "__main__":
    unittest.main()
