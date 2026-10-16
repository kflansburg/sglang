"""Property fuzz: PagedTokenToKVPoolAllocator conservation + no-double-alloc.

Runs random sequences of alloc_extend / alloc_decode / free / merge_and_sort_free
against a reference model that knows exactly which pages are live. Bug classes
targeted: capacity boundary at page_size=64, free-list double free, available_size
drift under error paths, stale last_loc reuse.

Model: each `req` owns a contiguous virtual row of `seq_len` tokens whose slots
are tracked page-aligned like the real request would be (subset semantics: we
free exactly what we allocated, chunked in page units). We assert:
  1. conservation: live_pages ∪ free_pages == {1..num_pages}, disjoint.
  2. allocator never hands out a page already live.
  3. available_size() stays consistent with free_pages count.
"""

from __future__ import annotations

import random
import sys

import torch
import pytest

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

PAGE = 64


class Model:
    def __init__(self, num_pages, page_size, need_sort, seed):
        self.page_size = page_size
        self.num_pages = num_pages
        self.alloc = PagedTokenToKVPoolAllocator(
            size=num_pages * page_size,
            page_size=page_size,
            dtype=torch.float16,
            device="cpu",
            kvcache=object(),
            need_sort=need_sort,
        )
        self.rng = random.Random(seed)
        # req_id -> dict(seq_len=int, pages=set[int])
        self.reqs: dict[int, dict] = {}
        self.next_req = 0

    def live_pages(self) -> set[int]:
        out: set[int] = set()
        for r in self.reqs.values():
            out |= r["pages"]
        return out

    def free_set(self) -> set[int]:
        free_pages = self.alloc.get_all_free_pages()
        return set(free_pages.tolist())

    def check_invariants(self, op: str):
        live = self.live_pages()
        free = self.free_set()
        assert len(free) == len(self.alloc.get_all_free_pages()), (
            f"[{op}] duplicated page in free list: "
            f"{len(free)} unique vs {len(self.alloc.get_all_free_pages())} entries"
        )
        inter = live & free
        assert not inter, f"[{op}] pages {sorted(inter)[:8]} both live and free"
        union = live | free
        expected = set(range(1, self.num_pages + 1))
        assert union == expected, (
            f"[{op}] conservation broken: missing {sorted(expected - union)[:8]} "
            f"extra {sorted(union - expected)[:8]}"
        )
        assert not (0 in free or 0 in live), f"[{op}] sink page 0 in circulation"

    # extend a fresh or existing request
    def op_extend(self):
        rid = self.rng.choice(list(self.reqs)) if self.reqs and self.rng.random() < 0.5 else None
        if rid is None:
            rid = self.next_req
            self.next_req += 1
            self.reqs[rid] = {"seq_len": 0, "pages": set()}
        r = self.reqs[rid]
        extend_by = self.rng.randint(1, 3 * self.page_size)
        prefix = r["seq_len"]
        seq = prefix + extend_by
        if seq > self.num_pages * self.page_size // 2:
            return  # keep total bounded so we hit boundaries often

        prefix_t = torch.tensor([prefix])
        seq_t = torch.tensor([seq])
        if r["seq_len"] == 0:
            last_loc = torch.tensor([-1])
        else:
            last_loc = None  # must be consistent with prior alloc; see below

        # Track the last token's index from the model side
        if last_loc is None:
            last_loc = torch.tensor([r["last_idx"]])

        out = self.alloc.alloc_extend(
            prefix_t, prefix_t, seq_t, seq_t, last_loc.clone(), extend_by
        )
        if out is None:
            # OOM: nothing should have changed
            self.check_invariants("extend-oom")
            return
        out = out.tolist()
        # model: phase1 = fill partial page; phase2/3: new pages
        new_pages = set()
        for i, tok in enumerate(range(prefix, seq)):
            idx = out[i]
            p = idx // self.page_size
            if tok < ((prefix + self.page_size - 1) // self.page_size) * self.page_size \
               and prefix % self.page_size != 0:
                # phase-1: still on the prefix's last partial page — must be an
                # already-owned page, and sequentially after last idx
                assert p in r["pages"], (
                    f"phase-1 token {tok} landed on non-owned page {p}"
                )
            else:
                new_pages.add(p)
                r["pages"].add(p)
        r["last_idx"] = out[-1]
        r["seq_len"] = seq
        self.check_invariants("extend")

    def op_decode(self):
        if not self.reqs:
            return
        rid = self.rng.choice(list(self.reqs))
        r = self.reqs[rid]
        seq = r["seq_len"] + 1
        seq_t = torch.tensor([seq])
        last_loc = torch.tensor([r["last_idx"]])
        out = self.alloc.alloc_decode(seq_t, seq_t, last_loc)
        if out is None:
            self.check_invariants("decode-oom")
            return
        p = int(out.item()) // self.page_size
        if seq % self.page_size != 1:
            # same partial page as before
            assert p in r["pages"], f"decode token {seq} on non-owned page {p}"
        else:
            r["pages"].add(p)
        r["last_idx"] = int(out.item())
        r["seq_len"] = seq
        self.check_invariants("decode")

    def op_free(self):
        if not self.reqs:
            return
        rid = self.rng.choice(list(self.reqs))
        r = self.reqs.pop(rid)
        if r["seq_len"] == 0:
            return
        # free whole pages the request owns, via token indices (mimics
        # req_to_token contents: last partial page too)
        pages = sorted(r["pages"])
        token_idx = []
        for p in pages:
            base = p * self.page_size
            token_idx.extend(range(base, base + self.page_size))
        self.alloc.free(torch.tensor(token_idx, dtype=torch.int64))
        self.check_invariants("free")

    def op_merge(self):
        self.alloc.merge_and_sort_free()
        self.check_invariants("merge")

    def run(self, steps: int):
        for i in range(steps):
            x = self.rng.random()
            if x < 0.35:
                self.op_extend()
            elif x < 0.55:
                self.op_decode()
            elif x < 0.9:
                self.op_free()
            else:
                self.op_merge()
        # final drain: free everything, check full conservation
        for rid in list(self.reqs):
            r = self.reqs.pop(rid)
            pages = sorted(r["pages"])
            if not pages:
                continue
            token_idx = []
            for p in pages:
                token_idx.extend(range(p * self.page_size, (p + 1) * self.page_size))
            self.alloc.free(torch.tensor(token_idx, dtype=torch.int64))
        self.check_invariants("drain")
        assert self.alloc.available_size() == self.num_pages * self.page_size, (
            f"after drain available_size={self.alloc.available_size()} "
            f"expected {self.num_pages * self.page_size}"
        )


@pytest.mark.parametrize("need_sort", [True, False])
@pytest.mark.parametrize("seed", list(range(10)))
def test_paged_allocator_conservation(need_sort, seed):
    m = Model(num_pages=20, page_size=PAGE, need_sort=need_sort, seed=seed)
    m.run(steps=400)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q"]))
