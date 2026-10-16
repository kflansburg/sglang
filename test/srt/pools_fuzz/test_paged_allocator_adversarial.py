"""Adversarial probes for PagedTokenToKVPoolAllocator (real code, CPU).

Extends test_paged_allocator_fuzz.py (lead's base harness) with the op shapes
pools-lead asked to probe; does not rewrite the base model.

Covers:
 * free_segment / free_segments valid-use conservation (fixed-shape whole-page
   frees used by the SWA/radix paths).
 * Adversarial free_group: double release of the same pages inside ONE group.
   - via token-vector free() twice: torch.unique at group_end dedups ACROSS the
     deferred piles -> SAFE (regression-pinned).
   - via free_page_ids twice: no dedup on that path -> duplicate page ids enter
     the free list and later allocs hand the same page to two owners. This is a
     caller-contract violation ("no page twice"): proven corruption primitive,
     internal seam only (no in-tree production caller of free_page_ids outside
     the allocator layer at HEAD); caught under SGLANG_DEBUG_MEMORY_POOL.
   - mixed free()+free_page_ids same page in one group: same corruption.
 * boundary ops: extends with prefix % page_size == 0 (production radix-match
   quantum), decode where the request's last page is exactly full.
"""

import os
import random

import pytest
import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

PAGE = 64


def make(num_pages=20, page_size=PAGE, need_sort=True, seed=0):
    a = PagedTokenToKVPoolAllocator(
        size=num_pages * page_size,
        page_size=page_size,
        dtype=torch.float16,
        device="cpu",
        kvcache=object(),
        need_sort=need_sort,
    )
    return a


def free_pages_of(a):
    return a.get_all_free_pages().tolist()


@pytest.mark.parametrize("need_sort", [True, False])
def test_within_group_double_free_via_token_vector_is_deduped(need_sort):
    a = make(need_sort=need_sort)
    out = a.alloc(3 * PAGE)
    a.free_group_begin()
    a.free(out)
    a.free(out)  # same token vector twice inside one group
    a.free_group_end()
    pages = sorted(set((out // PAGE).tolist()))
    fp = free_pages_of(a)
    for p in pages:
        assert fp.count(p) == 1, (
            f"token-vector double free inside one group duplicated page {p}"
        )


@pytest.mark.parametrize("need_sort", [True, False])
def test_within_group_double_free_via_page_ids_corrupts(need_sort):
    """Caller-contract violation ('no page twice'): free_page_ids path has NO
    dedup, so the same ids twice in one group duplicate the page in the free
    list, and subsequent allocs hand it to two owners."""
    a = make(need_sort=need_sort)
    out = a.alloc(3 * PAGE)
    pages = sorted(set((out // PAGE).tolist()))
    a.free_group_begin()
    a.free_page_ids(torch.tensor(pages))
    a.free_page_ids(torch.tensor(pages))  # caller bug: same ids twice
    a.free_group_end()
    fp = free_pages_of(a)
    assert fp.count(pages[0]) == 2  # corruption primitive holds at HEAD
    # the duplicate materializes as a double-alloc slot:
    seen = []
    while True:
        r = a.alloc(PAGE)
        if r is None:
            break
        seen.append(int(r[0]) // PAGE)
    assert seen.count(pages[0]) == 2, "page handed to two owners"


@pytest.mark.parametrize("need_sort", [True, False])
def test_within_group_mixed_free_paths_same_page_corrupts(need_sort):
    a = make(need_sort=need_sort)
    out = a.alloc(2 * PAGE)
    pages = sorted(set((out // PAGE).tolist()))
    a.free_group_begin()
    a.free(out)  # token vector (deduped path)
    a.free_page_ids(torch.tensor(pages))  # page ids (non-deduped path)
    a.free_group_end()
    fp = free_pages_of(a)
    # the token path uniques its own pile; the page-id path releases separately
    # -> the page is in the free list twice unless an allocator-level sweep sees it.
    assert fp.count(pages[0]) == 2


def test_debug_mode_catches_group_page_id_double_free():
    os.environ["SGLANG_DEBUG_MEMORY_POOL"] = "true"
    try:
        a = make()
        assert a.debug_mode
        out = a.alloc(3 * PAGE)
        pages = sorted(set((out // PAGE).tolist()))
        a.free_group_begin()
        a.free_page_ids(torch.tensor(pages))
        a.free_page_ids(torch.tensor(pages))
        with pytest.raises(AssertionError):
            a.free_group_end()
    finally:
        os.environ.pop("SGLANG_DEBUG_MEMORY_POOL", None)


# ---------------- free_segment / free_segments ------------------

class SegmentModel:
    """Drives valid free_segment/free_segments ops and cross-checks conservation."""

    def __init__(self, need_sort, seed, num_pages=20):
        self.page_size = PAGE
        self.num_pages = num_pages
        self.alloc = make(num_pages=num_pages, need_sort=need_sort)
        self.rng = random.Random(seed)
        self.owned_pages = set()

    def check(self, tag):
        free = set(free_pages_of(self.alloc))
        assert not (free & self.owned_pages), f"[{tag}] live page in free list"
        allx = free | self.owned_pages
        assert allx == set(range(1, self.num_pages + 1)), (
            f"[{tag}] conservation: missing {sorted(set(range(1, self.num_pages + 1)) - allx)}"
        )
        assert len(free_pages_of(self.alloc)) == len(free), f"[{tag}] dup in free list"

    def op_alloc(self):
        n = self.rng.randint(1, 4)
        r = self.alloc.alloc(n * PAGE)
        if r is None:
            return
        self.owned_pages |= set((r // PAGE).tolist())
        self.check("alloc")

    def op_free_segments(self):
        # choose owned pages, free them as contiguous page-aligned segments
        if not self.owned_pages:
            return
        pages = sorted(self.rng.sample(
            sorted(self.owned_pages), min(len(self.owned_pages), self.rng.randint(1, 4))))
        # build segments of consecutive pages (free_index contiguous slots per run)
        runs = []
        cur = [pages[0]]
        for p in pages[1:]:
            if p == cur[-1] + 1:
                cur.append(p)
            else:
                runs.append(cur)
                cur = [p]
        runs.append(cur)
        segs = []
        for run in runs:
            start_tok = run[0] * PAGE
            n = len(run) * PAGE
            segs.append((torch.arange(start_tok, start_tok + n), start_tok))
        if self.rng.random() < 0.5:
            for fi, sp in segs:
                self.alloc.free_segment(fi, start_pos=sp)
        else:
            self.alloc.free_segments(segs)
        for p in pages:
            self.owned_pages.discard(p)
        self.check("free_segments")

    def op_grouped_free_segments(self):
        if not self.owned_pages:
            return
        pages = sorted(self.owned_pages)
        self.alloc.free_group_begin()
        for p in pages:
            self.alloc.free_segment(
                torch.arange(p * PAGE, (p + 1) * PAGE), start_pos=p * PAGE)
        self.alloc.free_group_end()
        self.owned_pages.clear()
        self.check("grouped_free_segments")


@pytest.mark.parametrize("need_sort", [True, False])
@pytest.mark.parametrize("seed", range(8))
def test_free_segment_ops_conserve(need_sort, seed):
    m = SegmentModel(need_sort, seed)
    for _ in range(300):
        x = m.rng.random()
        if x < 0.45:
            m.op_alloc()
        elif x < 0.85:
            m.op_free_segments()
        else:
            m.op_grouped_free_segments()


# ---------------- boundary ops ------------------

@pytest.mark.parametrize("need_sort", [True, False])
@pytest.mark.parametrize("seed", range(8))
def test_page_multiple_prefix_extends(need_sort, seed):
    """prefix %% page_size == 0 (radix match quantum) — extend across exact
    boundaries, then decode until the partial page is exactly full, then one more."""
    rng = random.Random(seed)
    a = make(need_sort=need_sort)
    seq = 0
    last_loc = -1
    owned = set()
    for i in range(60):
        prefix = seq
        delta = rng.choice([PAGE, 2 * PAGE, PAGE - 1, 1, PAGE + 1])
        out = a.alloc_extend(
            torch.tensor([prefix]), torch.tensor([prefix]),
            torch.tensor([prefix + delta]), torch.tensor([prefix + delta]),
            torch.tensor([last_loc]), delta,
        )
        if out is None:
            break
        seq = prefix + delta  # advance only on success: seq tracks owned slots
        owned |= {int(x) // PAGE for x in out.tolist()}
        last_loc = int(out[-1])
        # decode a few steps
        for _ in range(rng.randint(0, PAGE + 1)):
            d = a.alloc_decode(
                torch.tensor([seq + 1]), torch.tensor([seq + 1]),
                torch.tensor([last_loc]))
            if d is None:
                break
            seq += 1
            owned.add(int(d.item()) // PAGE)
            last_loc = int(d.item())
    # free whole owned pages; conservation must hold exactly
    free_before = set(free_pages_of(a))
    assert not (free_before & owned)
    a.free(torch.tensor(sorted(owned)) * PAGE)
    free_after = set(free_pages_of(a))
    assert free_after | set() == free_before | owned
    # available_size never exceeded the page count. NOTE: the pool's slot
    # convention is pages 1..num_pages (slot 0 reserved), so the slots can reach
    # num_pages*ps = size+ps: KV buffers are allocated size+page_size to cover
    # exactly that (memory_pool.py:2269). available_size() itself is bounded by
    # num_pages pages.
    assert a.available_size() <= a.num_pages * PAGE
    assert set(free_pages_of(a)) <= set(range(1, a.num_pages + 1))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
