"""CPU fuzz harness for SGLang KV-cache allocator free-list logic.

Drives random alloc_extend / alloc_decode / free / free_group / merge sequences
against a faithful model of the free-list implementations in
  python/sglang/srt/mem_cache/allocator/{token.py,paged.py}
  python/sglang/kernels/ops/memory/allocator.py (alloc_*_naive reference math,
    which the Triton kernels mirror; the naive math is inlined in PagedModel).

Harness realism (learned the hard way): a request's slots come from pages it
owns. A 191-token prefix does NOT occupy slots 0..190; it occupies slots on
pages drawn from the free list over its life. last_loc is the slot of its last
token. The harness therefore keeps a real req_to_token map per request:
slots[rid] = [slot per token], and last_loc = slots[rid][-1]. Phase-1 slots of
an extend land on the request's own partial page (never a free page), so no
false positive double-allocation is reported.

Checked invariants after every op:
  * ownership: every page in [1, num_pages] is exactly one of free-list |
    staged-release | owned-by-exactly-one-request (no leak, no double-free).
  * no slot collision: the multiset of all live slots (across requests) has no
    duplicates => every token gets a distinct physical slot.
  * allocator.available_size() == (free+staged pages) * page_size.
  * a free vector's footprint pages are all owned by its freeing request (a
    violation = freeing a live/unowned page = the UAF primitive).

Runs on CPU torch only; sglang is not imported, so the shared sandbox runs it.
"""

import random
import sys

import torch


class PagedModel:
    """Mirrors PagedTokenToKVPoolAllocator free-list logic (naive kernels, CPU)."""

    def __init__(self, size: int, page_size: int, need_sort: bool):
        self.size = size
        self.page_size = page_size
        self.num_pages = size // page_size
        self.need_sort = need_sort
        self.clear()

    def clear(self):
        self.free_pages = torch.arange(1, self.num_pages + 1, dtype=torch.int64)
        self.staged_pages = []
        self.num_staged_pages = 0
        self.free_group = None
        self.free_page_ids_group = []

    def available_size(self):
        return (len(self.free_pages) + self.num_staged_pages) * self.page_size

    def merge_and_sort_free(self):
        if not self.staged_pages:
            return
        self.free_pages, _ = torch.sort(torch.cat([self.free_pages, *self.staged_pages]))
        self.staged_pages = []
        self.num_staged_pages = 0

    def alloc_extend(self, prefix_lens, seq_lens, last_loc, extend_num_tokens):
        ps = self.page_size
        bs = len(prefix_lens)
        max_new_pages = extend_num_tokens // ps + bs + 1
        if max_new_pages > len(self.free_pages):
            self.merge_and_sort_free()
        num_new_pages = int(
            ((seq_lens + ps - 1) // ps - (prefix_lens + ps - 1) // ps).sum()
        )
        if max_new_pages > len(self.free_pages) and num_new_pages > len(self.free_pages):
            return None
        out = torch.empty((extend_num_tokens,), dtype=torch.int64)
        el = seq_lens - prefix_lens
        ep = torch.cumsum(el, 0)
        sp = ep - el
        j = torch.arange(extend_num_tokens)
        owner = torch.searchsorted(ep, j, right=True)
        local = j - sp[owner]
        llg = last_loc[owner]
        cp = (prefix_lens + ps - 1) // ps * ps
        fs = seq_lens // ps * ps
        if self.free_pages.numel() == 0:
            cs = (seq_lens + ps - 1) // ps * ps
            assert torch.all(cs == cp)
            out.copy_(llg + 1 + local)
        else:
            n1 = torch.clamp(seq_lens, max=cp) - prefix_lens
            d1 = (prefix_lens + n1) == seq_lens
            n2 = torch.where(d1, torch.zeros_like(n1), fs - cp)
            n3 = torch.where(d1, torch.zeros_like(n1), seq_lens - fs)
            ppr = n2 // ps + (n3 > 0)
            enp = torch.cumsum(ppr, 0)
            snp = enp - ppr
            n1g = n1[owner]; n2g = n2[owner]; snpg = snp[owner]; enpg = enp[owner]
            is1 = local < n1g
            is2 = (~is1) & (local < n1g + n2g)
            v1 = llg + 1 + local
            r2 = torch.clamp(local - n1g, min=0)
            pi2 = torch.clamp(snpg + r2 // ps, min=0, max=self.free_pages.numel() - 1)
            v2 = self.free_pages[pi2] * ps + r2 % ps
            r3 = torch.clamp(local - n1g - n2g, min=0)
            pi3 = torch.clamp(enpg - 1, min=0, max=self.free_pages.numel() - 1)
            v3 = self.free_pages[pi3] * ps + r3
            out.copy_(torch.where(is1, v1, torch.where(is2, v2, v3)))
        popped = self.free_pages[:num_new_pages].clone()
        self.free_pages = self.free_pages[num_new_pages:]
        return out, popped

    def alloc_decode(self, seq_lens, last_loc):
        ps = self.page_size
        bs = len(seq_lens)
        if bs > len(self.free_pages):
            self.merge_and_sort_free()
        num_new_pages = int(((seq_lens % ps) == 1).sum())
        if num_new_pages > len(self.free_pages):
            return None
        out = torch.empty((bs,), dtype=torch.int64)
        if num_new_pages == 0:
            out.copy_(last_loc + 1)
        else:
            need_new = (((seq_lens - 1) % ps) == 0).to(torch.int64)
            rank = (torch.cumsum(need_new, 0) - need_new).clamp_(max=num_new_pages - 1)
            out.copy_(torch.where(need_new.bool(), self.free_pages[rank] * ps, last_loc + 1))
        popped = self.free_pages[:num_new_pages].clone()
        self.free_pages = self.free_pages[num_new_pages:]
        return out, popped

    def free(self, free_index):
        if free_index.numel() == 0:
            return
        if self.free_group is None:
            self._release(torch.unique(free_index // self.page_size))
        else:
            self.free_group.append(free_index.clone())

    def free_page_ids(self, page_ids):
        if page_ids.numel() == 0:
            return
        if self.free_group is None:
            self._release(page_ids)
        else:
            self.free_page_ids_group.append(page_ids.clone())

    def _release(self, *ids):
        if self.need_sort:
            self.staged_pages.extend(ids)
            self.num_staged_pages += sum(i.numel() for i in ids)
        else:
            self.free_pages = torch.cat([*ids, self.free_pages])

    def free_group_begin(self):
        assert self.free_group is None
        self.free_group = []
        self.free_page_ids_group = []

    def free_group_end(self):
        pending, self.free_group = self.free_group, None
        if pending:
            self.free(torch.cat(pending))
        if self.free_page_ids_group:
            self._release(torch.cat(self.free_page_ids_group))
            self.free_page_ids_group = []


class Harness:
    """Ownership oracle + op driver around PagedModel, with real req_to_token maps."""

    def __init__(self, seed, size=64 * 48, page_size=64, need_sort=True):
        self.rng = random.Random(seed)
        self.ps = page_size
        self.m = PagedModel(size, page_size, need_sort)
        self.owner = {}        # page -> rid
        self.slots = {}        # rid -> list of slots per token (a real req_to_token)
        self.next_rid = 0
        self.group_open = False
        self.deferred_release = []  # pages freed inside an open group
        self.live_slot_set = set()
        self.history = []

    # ---- ownership ----
    def _claim(self, pages, rid):
        for p in pages:
            p = int(p)
            assert self.owner.get(p) is None, f"double-alloc page {p} held by r{self.owner.get(p)}"
            self.owner[p] = rid

    def _check_conservation(self):
        free_set = set(int(x) for x in self.m.free_pages.tolist())
        staged_set = set()
        for t in self.m.staged_pages:
            staged_set.update(int(x) for x in t.tolist())
        assert not (free_set & staged_set), f"page in free and staged: {free_set & staged_set}"
        # pages sitting in an open free_group are still owned (release deferred)
        in_group = set(self.deferred_release)
        if self.m.free_group is not None:
            seen = set()
            for t in self.m.free_group:
                seen.update(int(x) // self.ps for x in t.tolist())
            for t in self.m.free_page_ids_group:
                seen.update(int(x) for x in t.tolist())
            assert seen == in_group, (
                f"harness deferred {sorted(in_group)} vs model group pages {sorted(seen)}"
            )
        rec = free_set | staged_set
        for p in range(1, self.m.num_pages + 1):
            owned = self.owner.get(p) is not None or p in in_group
            # pages queued in the group must still be claimed in the owner table
            if p in in_group:
                assert self.owner.get(p) is not None, (
                    f"page {p} released into free_group but not owned"
                )
            if p in in_group:
                continue
            assert owned != (p in rec), (
                f"page {p}: owned={owned} recoverable={p in rec} "
                f"(leak or free-of-live)"
            )
        assert self.m.available_size() == (len(free_set) + len(staged_set)) * self.ps
        # no slot collision across all live tokens
        allslots = []
        for rid, sl in self.slots.items():
            allslots.extend(sl)
        assert len(set(allslots)) == len(allslots), "two live tokens share a physical slot"

    # ---- ops ----
    def op_extend(self):
        r = self.rng
        # sometimes extend an existing request (radix-cache prefix match style)
        if self.slots and r.random() < 0.3:
            rid = r.choice(list(self.slots))
            pre = r.randint(len(self.slots[rid]), len(self.slots[rid]))
        else:
            rid = None
            pre = 0
        if rid is None or pre == len(self.slots[rid]):
            pre_t_len = pre
            choices = [c for c in [1, self.ps - 1, self.ps, self.ps + 1, 2 * self.ps,
                                   2 * self.ps + 3, r.randint(1, 3 * self.ps)] if c > 0]
            delta = r.choice(choices)
            ln = pre_t_len + delta
            last = self.slots[rid][pre_t_len - 1] if (rid is not None and pre_t_len > 0) else -1
        else:
            raise AssertionError
        pre_t = torch.tensor([pre_t_len]); seq_t = torch.tensor([ln])
        last_t = torch.tensor([last])
        res = self.m.alloc_extend(pre_t, seq_t, last_t, ln - pre_t_len)
        if res is None:
            return f"extend-none(p{pre_t_len}->{ln})", None
        out, popped = res
        if rid is None:
            rid = self.next_rid; self.next_rid += 1
            self.slots[rid] = []
        self._claim(popped.tolist(), rid)
        self.slots[rid].extend(int(x) for x in out.tolist())
        assert len(self.slots[rid]) == ln
        self._check_conservation()
        return f"extend(r{rid} {pre_t_len}->{ln})", rid

    def op_decode(self):
        r = self.rng
        cands = list(self.slots)
        if not cands:
            return "skip", None
        rid = r.choice(cands)
        ln = len(self.slots[rid]) + 1
        last = self.slots[rid][-1]
        if self.ps == 1:
            # token (page_size==1) pools do not use the paged alloc_decode
            # kernel; decode pops one slot via TokenToKVPoolAllocator.alloc(1).
            # alloc_decode_naive's `last_loc + 1` fill is only sound when the
            # slot is within an already-owned page (ps > 1).
            if len(self.m.free_pages) < 1:
                if self.m.staged_pages:
                    self.m.merge_and_sort_free()
            if len(self.m.free_pages) < 1:
                return f"decode-none(r{rid}->{ln})", None
            slot = int(self.m.free_pages[0])
            popped = [slot]
            self.m.free_pages = self.m.free_pages[1:]
            self._claim(popped, rid)
            self.slots[rid].append(slot)
            self._check_conservation()
            return f"decode-alloc1(r{rid}->{ln})", rid
        res = self.m.alloc_decode(torch.tensor([ln]), torch.tensor([last]))
        if res is None:
            return f"decode-none(r{rid}->{ln})", None
        out, popped = res
        self._claim(popped.tolist(), rid)
        self.slots[rid].append(int(out[0]))
        self._check_conservation()
        return f"decode(r{rid}->{ln})", rid

    def op_free(self):
        r = self.rng
        # production-like: free whole trailing pages of a request (whole-page granularity),
        # or the full request (tree owner releases pages whose full range is covered).
        cands = list(self.slots)
        if not cands:
            return "skip", None
        rid = r.choice(cands)
        k = len(self.slots[rid])
        # free ALL pages owned by rid (whole-request teardown/abort): the
        # contract-correct whole-page free. Partial frees are not modeled (the
        # tree never releases a subset of a request's pages outside retract,
        # which hands the suffix back through req_to_token truncation -- a
        # suffix free is exactly a smaller whole; here the whole is used).
        pages = sorted(p for p, o in self.owner.items() if o == rid)
        if not pages:
            return "skip", None
        take = len(pages)
        pages = pages[:take]
        for p in pages:
            # free vector: full page coverage (tree frees whole pages)
            idx = torch.arange(p * self.ps, (p + 1) * self.ps, dtype=torch.int64)
            assert self.owner[p] == rid
            self.m.free(idx)
            if self.m.free_group is not None:
                # release deferred: keep the owner entry until free_group_end
                self.deferred_release.append(p)
            else:
                self.owner[p] = None
        # whole-request free only (the tree abort/teardown case): every page
        # rid owns is released, so the request disappears. free() handles both
        # the immediate (group closed) and deferred (group open) paths.
        del self.slots[rid]
        self._check_conservation()
        return f"free(r{rid}, {len(pages)}p)", rid

    def op_merge(self):
        self.m.merge_and_sort_free()
        self._check_conservation()
        return "merge", None

    def op_group(self, end):
        if end and self.group_open:
            self.m.free_group_end(); self.group_open = False
            for p in self.deferred_release:
                self.owner[p] = None
            self.deferred_release = []
        elif not end and not self.group_open:
            self.m.free_group_begin(); self.group_open = True
        self._check_conservation()
        return "grp_end" if end else "grp_begin", None


def run(seed, steps=3000, need_sort=True, page_size=64, verbose=False):
    torch.manual_seed(seed)
    h = Harness(seed, need_sort=need_sort, page_size=page_size)
    ops = (
        [h.op_extend] * 4 + [h.op_decode] * 3 + [h.op_free] * 5
        + [h.op_merge] + [lambda: h.op_group(False)] + [lambda: h.op_group(True)]
    )
    for i in range(steps):
        op = h.rng.choice(ops)
        try:
            tag, _ = op()
        except AssertionError as e:
            print(f"[seed={seed} step={i}] INVARIANT VIOLATION: {e}")
            return False
        if verbose:
            print(f"  step {i}: {tag}")
    live = sum(1 for o in h.owner.values() if o is not None)
    print(f"[seed={seed} sort={int(need_sort)} ps={page_size}] {steps} steps OK, "
          f"live_pages={live}, avail={h.m.available_size()}")
    return True


if __name__ == "__main__":
    ok = True
    for seed in range(10):
        ok &= run(seed, steps=3000, need_sort=True, page_size=64)
    for seed in range(10, 14):
        ok &= run(seed, steps=3000, need_sort=False, page_size=64)
    for seed in range(14, 18):
        ok &= run(seed, steps=3000, need_sort=True, page_size=1)
    print("ALL OK" if ok else "FAILURES FOUND")
    sys.exit(0 if ok else 1)
