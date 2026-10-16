#!/usr/bin/env python3
"""Pure-python slot-ownership harness for the mamba extra_buffer donate/boundary
lifecycle (board #8 items 1-2). State machines transcribed from:
  boundary_prealloc  schedule_batch.py:3503-3532 (lazy) / 3123-3130 (non-lazy)
  boundary_post      batch_result_processor.py:1646-1658
  donate_running     components/mamba.py:624-629 + memory_pool.py:1802-1822
  finish             components/mamba.py:590-601 + memory_pool.py:1825-1885
  donate insert-fail components/mamba.py:681-684
  abort              free_mamba_cache (memory_pool.py:1845-1863)
Oracle invariant: no live slot id appears twice across {req pool_idx, req buf,
tree, allocator free list}; free list has no -1, no dups, no out-of-range.
"""
import random, itertools

class Oracle:
    def __init__(self, size=32):
        self.size = size
        self.free = list(range(1, size + 1))
    def alloc(self):
        return self.free.pop(0) if self.free else None
    def free_slot(self, s):
        self.free.append(s)  # unguarded MambaSlotAllocator.free semantics
def check(orc, reqs, tree, label):
    assert -1 not in orc.free, f"{label}: -1 in free list"
    assert all(0 < s <= orc.size for s in orc.free), f"{label}: OOR slot in free list"
    dups = [s for s in set(orc.free) if orc.free.count(s) > 1]
    assert not dups, f"{label}: DOUBLE FREE {dups} free={orc.free}"
    seen = {}
    for rid, r in reqs.items():
        real = [x for x in r["buf"] if x != -1]
        assert len(set(real)) == len(real), f"{label}: req{rid} buf dup {r['buf']}"
        for s in [r["pool_idx"]] + real:
            assert s not in seen, f"{label}: slot {s} aliased by {seen.get(s)} and req{rid}"
            assert s not in orc.free, f"{label}: req{rid} slot {s} also free"
            seen[s] = f"req{rid}"
    for s in tree:
        assert s not in seen, f"{label}: tree slot {s} aliases {seen[s]}"
        assert s not in orc.free, f"{label}: tree slot {s} also free"
        seen[s] = "tree"

def run(seed, lazy, steps=600):
    rnd = random.Random(seed)
    orc = Oracle()
    reqs, tree = {}, {}
    rid_seq = itertools.count()
    def new_req():
        rid = next(rid_seq)
        a, b = orc.alloc(), orc.alloc()
        if a is None or b is None:
            for x in (a, b):
                if x is not None: orc.free_slot(x)
            return None
        if lazy:
            reqs[rid] = dict(pool_idx=a, buf=[b, -1], next=0, last=0, prev_sl=None)
        else:
            c = orc.alloc()
            if c is None:
                orc.free_slot(a); orc.free_slot(b); return None
            reqs[rid] = dict(pool_idx=a, buf=[b, c], next=0, last=1, prev_sl=None)
        return rid
    def donate(rid):
        r = reqs[rid]
        keep = r["last"]
        new = orc.alloc()
        if new is None:
            if tree:  # _alloc_mamba_slot evict-assist: frees one tree slot then re-allocs
                orc.free_slot(tree.pop(rnd.choice(list(tree.keys()))))
            new = orc.alloc(); assert new is not None
        donated = r["buf"][keep]
        assert donated != -1, f"donated -1: rid={rid} buf={r['buf']} last={keep}"
        r["buf"][keep] = new  # set_mamba_ping_pong_slot(keep, new)
        if rnd.random() < 0.3:
            orc.free_slot(donated)  # insert fail: _free_mamba_value
        else:
            tree[(rid, rnd.random())] = donated
    def finish(rid, overshoot=False):
        r = reqs[rid]
        keep = r["last"]
        # _select_finished_checkpoint: overshoot fallback to OTHER slot when
        # prev_seqlen exists (non-lazy only; lazy keeps prev=None)
        if overshoot and not lazy and r["prev_sl"] is not None:
            keep = 1 - keep
        elif overshoot:
            keep = None  # donate nothing; free everything
        if keep is None or r["buf"][keep] == -2:
            orc.free_slot(r["pool_idx"])
            for s in r["buf"]:
                if s != -1: orc.free_slot(s)
        else:
            tree[(rid, "fin")] = r["buf"][keep]
            orc.free_slot(r["pool_idx"])
            other = 1 - keep
            if lazy or r["buf"][other] != r["buf"][keep]:
                if r["buf"][other] != -1: orc.free_slot(r["buf"][other])
        del reqs[rid]
    def abort(rid):
        r = reqs[rid]
        orc.free_slot(r["pool_idx"])
        for s in r["buf"]:
            if s != -1: orc.free_slot(s)
        del reqs[rid]
    def boundary(rid):
        r = reqs[rid]
        if lazy:
            other = 1 - r["next"]
            if r["buf"][other] == -1:
                s = orc.alloc()
                if s is None: return  # degraded
                r["buf"][other] = s; r["next"] = other
            track = r["next"]; r["last"] = track; r["next"] = track
            o = 1 - track
            if r["buf"][o] != -1:
                orc.free_slot(r["buf"][o]); r["buf"][o] = -1
            r["prev_sl"] = None  # lazy never names prev
        else:
            r["prev_sl"] = 1  # non-lazy names prev on boundary
            r["last"] = r["next"]; r["next"] = 1 - r["next"]
    n = 0
    for step in range(steps):
        ev = rnd.choice(["new", "boundary", "donate", "finish", "abort", "finish_over"])
        if ev == "new" and len(reqs) < 5: new_req()
        elif ev == "boundary" and reqs: boundary(rnd.choice(list(reqs)))
        elif ev == "donate" and reqs: donate(rnd.choice(list(reqs)))
        elif ev == "finish" and reqs: finish(rnd.choice(list(reqs)))
        elif ev == "finish_over" and reqs: finish(rnd.choice(list(reqs)), overshoot=True)
        elif ev == "abort" and reqs: abort(rnd.choice(list(reqs)))
        check(orc, reqs, tree, f"lazy={lazy} seed={seed} step={step} ev={ev}")
        n += 1
    return n

if __name__ == "__main__":
    t = 0
    for lazy in (True, False):
        for seed in range(150):
            t += run(seed, lazy)
    print(f"OK: {t} transitions, ownership invariant held (lazy + non-lazy)")
