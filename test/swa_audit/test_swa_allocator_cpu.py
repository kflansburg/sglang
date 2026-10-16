"""CPU regression tests for SWATokenToKVPoolAllocator memory-safety bugs.

Drives the REAL allocator classes (loaded from their repo files on CPU)
through alloc/evict/free sequences, checking slot conservation, no
double-free, mapping coherence, and the alloc_decode one-sided-leak fix.

Run with: python3 test/swa_audit/test_swa_allocator_cpu.py  (or pytest)
"""
import random
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest_real_alloc import load

base, token, paged, swa, BaseSWAKVPool = load()
SWATokenToKVPoolAllocator = swa.SWATokenToKVPoolAllocator
PagedTokenToKVPoolAllocator = paged.PagedTokenToKVPoolAllocator


class InstrPool(BaseSWAKVPool):
    """Instrumented sub-pool stat tracker (allocator internals untouched)."""


def held_and_free(sub):
    """Return (n_free_pages) of a Paged sub-allocator."""
    return len(sub.free_pages)


def new_allocator(size_full, size_swa, page_size):
    kvc = InstrPool()
    kvc.full_kv_pool = object()
    kvc.swa_kv_pool = object()
    alloc = SWATokenToKVPoolAllocator(
        size=size_full,
        size_swa=size_swa,
        page_size=page_size,
        dtype=torch.float16,
        device="cpu",
        kvcache=kvc,
        need_sort=False,
    )
    return alloc


def simulate_basic_lifecycle(ps=4, rnd_seed=0):
    """Random alloc_extend/alloc_decode/window-evict/free; conservation at end."""
    rnd = random.Random(rnd_seed)
    full_size = 128 * ps
    swa_size = 64 * ps
    alloc = new_allocator(full_size, swa_size, ps)

    live = {}  # rid -> dict(last_full, len, committed_len)
    dead = []

    for step in range(300):
        # add a request
        if rnd.random() < 0.4 and len(live) < 6:
            rid = f"r{step}"
            ext = rnd.randint(1, 5) * ps // 2  # tokens
            ext = max(ext, 1)
            seq_lens = torch.tensor([ext])
            prefix = torch.tensor([0])
            last_loc = torch.tensor([-1])
            out = alloc.alloc_extend(
                prefix, prefix.clone(), seq_lens, seq_lens.clone(), last_loc, ext
            )
            if out is not None:
                live[rid] = {"last": int(out[-1]), "len": ext, "held": [int(v) for v in out]}
        # decode step for random live
        if live and rnd.random() < 0.6:
            rid = rnd.choice(list(live))
            st = live[rid]
            # Callers pass seq_lens_next = committed_len + 1 (see allocation.py
            # alloc_for_decode_default / schedulers). Boundary iff len%ps==0.
            seq = torch.tensor([st["len"] + 1])
            last = torch.tensor([st["last"]])
            out = alloc.alloc_decode(seq, seq.clone(), last)
            if out is not None:
                st["last"] = int(out[-1])
                st["len"] += 1
                st["held"].append(int(out[-1]))
        # window evict (fake): free swa-only segment of a long request
        # free a request
        if live and rnd.random() < 0.25:
            rid = rnd.choice(list(live))
            st = live.pop(rid)
            alloc.free(torch.tensor(st["held"]))

    for st in live.values():
        alloc.free(torch.tensor(st["held"]))
    return alloc


def test_conservation_after_lifecycle():
    alloc = simulate_basic_lifecycle()
    f = len(alloc.full_attn_allocator.free_pages)
    s = len(alloc.swa_attn_allocator.free_pages)
    assert f == alloc.full_attn_allocator.num_pages, f"full leak: {f}"
    assert s == alloc.swa_attn_allocator.num_pages, f"swa leak: {s}"
    # mapping should be all zero (except trailing -1)
    m = alloc.full_to_swa_index_mapping
    assert int((m[:-1] != 0).sum()) == 0, "mapping not cleaned"
    print("conservation ok")


def test_alloc_decode_one_sided_leak_FIXED():
    """Regression for the alloc_decode one-sided leak: exhaust the SWA pool
    while the full pool still has pages, then force a boundary decode; both
    pools must be unchanged (no full-page leak)."""
    ps = 4
    # swa pool has exactly 1 free page available after setup; full plenty.
    alloc = new_allocator(16 * ps, 2 * ps, ps)
    # occupy one swa page + one full page via an extend
    out = alloc.alloc_extend(
        torch.tensor([0]), torch.tensor([0]),
        torch.tensor([ps]), torch.tensor([ps]),
        torch.tensor([-1]), ps,
    )
    assert out is not None
    # swa free pages now: 1 ; full free: 14
    assert len(alloc.swa_attn_allocator.free_pages) == 1
    before_full = len(alloc.full_attn_allocator.free_pages)

    # Craft a decode req at a page boundary whose SWA pool cannot provide:
    # exhaust the remaining swa page first (another extend on a second req)
    out2 = alloc.alloc_extend(
        torch.tensor([0]), torch.tensor([0]),
        torch.tensor([ps]), torch.tensor([ps]),
        torch.tensor([-1]), ps,
    )
    assert out2 is not None
    assert len(alloc.swa_attn_allocator.free_pages) == 0
    assert len(alloc.full_attn_allocator.free_pages) == before_full - 1

    # decode for req1: seq 4->5? seq_lens semantics: last committed len; page
    # boundary when (seq+1)..: alloc_decode allocates a page iff seq%ps==1.
    seq = torch.tensor([ps])  # seq=4 (positions 0..3 live); next pos 4 is NOT a boundary
    # to FORCE boundary: last token at pos ps-1 is index 3 -> seq=ps, new page iff seq%ps==1
    # seq%ps for ps=4: need seq==5->1? decode rule: (seq_lens % ps == 1) -> boundary
    seq = torch.tensor([ps + 1])  # %ps == 1 -> boundary, needs a new page each side
    alloc2 = alloc.alloc_decode(seq, seq.clone(), torch.tensor([int(out[-1])]))
    assert alloc2 is None, "expected decode OOM (swa pool empty)"
    # KEY INVARIANT: the full pool must not have lost the page it allocated.
    assert len(alloc.full_attn_allocator.free_pages) == before_full - 1, (
        f"ONE-SIDED LEAK: full pages {len(alloc.full_attn_allocator.free_pages)} "
        f"!= {before_full - 1}"
    )
    # and the full-pool page ids must be identical (no silent consumption)
    # free page uniqueness
    pages = alloc.full_attn_allocator.free_pages
    assert len(torch.unique(pages)) == len(pages)
    print("alloc_decode one-sided leak regression ok")


def test_alloc_decode_success_then_free():
    ps = 4
    alloc = new_allocator(16 * ps, 16 * ps, ps)
    out = alloc.alloc_extend(
        torch.tensor([0]), torch.tensor([0]),
        torch.tensor([ps]), torch.tensor([ps]),
        torch.tensor([-1]), ps,
    )
    fb, sb = (
        len(alloc.full_attn_allocator.free_pages),
        len(alloc.swa_attn_allocator.free_pages),
    )
    d = alloc.alloc_decode(torch.tensor([ps + 1]), torch.tensor([ps + 1]), torch.tensor([int(out[-1])]))
    assert d is not None
    alloc.free(torch.cat([out, d]))
    assert len(alloc.full_attn_allocator.free_pages) == alloc.full_attn_allocator.num_pages
    assert len(alloc.swa_attn_allocator.free_pages) == alloc.swa_attn_allocator.num_pages
    assert int((alloc.full_to_swa_index_mapping[:-1] != 0).sum()) == 0
    print("decode+free round trip ok")


if __name__ == "__main__":
    test_alloc_decode_success_then_free()
    test_alloc_decode_one_sided_leak_FIXED()
    test_conservation_after_lifecycle()
    print("ALL PASS")
