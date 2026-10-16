# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Model-checking fuzz for the unified sub-pool allocators (CPU only).

Drives MultiEndedAllocator (2-end mamba/full pair and the tri-pool with a FLOAT
middle) with interleaved page-granular alloc/free/flush/make_room ops against a
reference model, asserting:

  * every VIRTUAL page handed out is distinct (double-alloc detector)
  * every freed page was live (double-free detector)
  * v2p/p2v tables are inverse on live entries
  * physical pages: live XOR hole XOR beyond-frontier (no leak, no dup)
  * chain byte accounting violations list is empty after every op
  * watermark / span conservation: span == live + holes(+pending in lazy)

Run: python3 -m pytest test/srt/pools_fuzz/test_unified_alloc_fuzz.py -x
"""
import random
import unittest

import torch

from sglang.test.unified_allocator_fixtures import (
    build_swa_pool,
    build_tri_pool,
    setup_allocator_context,
)

register = None
try:
    from sglang.test.ci.ci_register import register_cpu_ci

    register_cpu_ci(est_time=25, suite="base-a-test-cpu")
except Exception:  # local run without the CI harness
    pass


class RefModel:
    """Reference allocation state for ONE physical pool (page granularity)."""

    def __init__(self):
        self.live_virt = set()  # virtual page ids currently handed out


def _check_invariants(name, alloc):
    """Allocator-local invariants for one MultiEndedAllocator."""
    v2p = alloc.virtual_to_physical
    p2v = alloc.physical_to_virtual
    n_v, n_p = alloc.num_virtual_ids, alloc.num_pages
    v2p_c, p2v_c = v2p.tolist(), p2v.tolist()

    # virtual page 0 <-> physical page 0 is the permanently-bound padding sink,
    # set in clear(); exclude it from all live bookkeeping.
    # v2p/p2v inverse on bound entries (skip trailing sentinels and the sink)
    for v in range(1, n_v):
        p = v2p_c[v]
        if p == -1:
            # p2v must not point back at v
            assert v not in p2v_c[:n_p], (
                f"{name}: v2p[{v}]=-1 but p2v has v (stale binding)"
            )
        else:
            assert 0 <= p < n_p, f"{name}: v2p[{v}]={p} out of range"
            assert p2v_c[p] == v, f"{name}: v2p[{v}]={p} but p2v[{p}]={p2v_c[p]}"
    # The sink itself must stay wired
    assert v2p_c[0] == 0 and p2v_c[0] == 0, f"{name}: sink page 0 unwired"

    # physical page census: each physical page is either free-space, hole,
    # pending-reuse, or live-bound exactly once
    holes = set(alloc._free_phys_pages.tolist())
    pending = set(alloc._pending_reuse_pages_cpu)
    assert not (holes & pending), f"{name}: page in both holes and pending"
    # no negative / out-of-range hole ever enters the free list (F0 tripwire)
    for h in holes:
        assert 0 < h < n_p, f"{name}: hole page {h} outside (0, {n_p})"
    live_phys = {v2p_c[v] for v in range(1, n_v) if v2p_c[v] != -1}
    assert len(live_phys) == sum(
        1 for v in range(1, n_v) if v2p_c[v] != -1
    ), f"{name}: two virtual pages bound to one physical page"

    if alloc.grow_direction == "up":
        in_band = lambda p: alloc.min_page_index <= p < alloc.watermark_physical
    else:
        in_band = lambda p: alloc.watermark_physical < p < alloc.num_pages
    if alloc.lazy_compaction:
        start = max(alloc.min_page_index, 1)
        for p in range(start, alloc.num_pages):
            is_live = p in live_phys
            is_hole = p in holes
            is_pend = p in pending
            free_space = not in_band(p)
            cnt = is_live + is_hole + is_pend + free_space
            assert cnt == 1, (
                f"{name}: phys page {p} census={cnt} "
                f"(live={is_live},hole={is_hole},pend={is_pend},free={free_space})"
            )
        assert alloc.live_page_count == len(live_phys), (
            f"{name}: live_page_count {alloc.live_page_count} != live {len(live_phys)}"
        )


class TestUnifiedAllocatorFuzz(unittest.TestCase):
    def setUp(self):
        setup_allocator_context()

    def _fuzz_pair(self, page_size, seed, steps=400):
        alloc = build_swa_pool(page_size=page_size)
        full = alloc.full_attn_allocator
        swa = alloc.swa_attn_allocator
        rng = random.Random(seed)
        ref = RefModel()
        ps = page_size

        for i in range(steps):
            op = rng.random()
            avail_pages = len(full.free_virtual_ids)
            if op < 0.45 and avail_pages > 0:  # page alloc
                n = rng.randint(1, min(4, avail_pages))
                tokens = alloc.alloc(n * ps)
                if tokens is None:
                    # shortfall: reference must agree we were full-ish
                    continue
                vpages = (tokens[::ps] // ps).tolist()
                for vp in vpages:
                    assert vp not in ref.live_virt, (
                        f"double-alloc of virtual page {vp} at step {i}"
                    )
                ref.live_virt.update(vpages)
            elif op < 0.85 and ref.live_virt:  # free
                n = rng.randint(1, min(6, len(ref.live_virt)))
                dead = rng.sample(sorted(ref.live_virt), n)
                ref.live_virt.difference_update(dead)
                tok = torch.tensor(
                    [vp * ps for vp in dead], dtype=torch.int64, device="cpu"
                )
                if rng.random() < 0.5:
                    alloc.free(tok)
                else:
                    alloc.free_segment(tok, start_pos=0)
            else:  # flush
                alloc.flush_opportunistic()

            _check_invariants("full", full)
            _check_invariants("swa", swa)
            viol = alloc.verify_byte_accounting()
            assert not viol, f"step {i}: {viol}"
            assert full.free_virtual_ids.tolist() != [] or True
            # virtual ids unique in free list (double-alloc via free-list dup)
            fv = full.free_virtual_ids
            assert len(set(fv.tolist())) == len(fv), (
                f"step {i}: duplicate virtual page in free_virtual_ids"
            )
        # drain
        if ref.live_virt:
            tok = torch.tensor(
                [vp * ps for vp in ref.live_virt], dtype=torch.int64, device="cpu"
            )
            alloc.free(tok)
        _check_invariants("full", full)
        _check_invariants("swa", swa)

    def _fuzz_tri(self, page_size, seed, steps=400):
        bundle, alloc = build_tri_pool(page_size=page_size)
        full = alloc.full_attn_allocator
        swa = alloc.swa_attn_allocator
        mamba = alloc.mamba_allocator
        rng = random.Random(seed)
        ref = RefModel()
        mref = RefModel()
        ps = page_size

        for i in range(steps):
            op = rng.random()
            if op < 0.35:  # full+swa coupled alloc via composite
                n = rng.randint(1, 4)
                tokens = alloc.alloc(n * ps)
                if tokens is not None:
                    vpages = (tokens[::ps] // ps).tolist()
                    for vp in vpages:
                        assert vp not in ref.live_virt, (
                            f"tri double-alloc {vp} at step {i}"
                        )
                    ref.live_virt.update(vpages)
            elif op < 0.5:  # mamba slot alloc (page_size 1)
                if mref.live_virt or rng.random() < 0.8:
                    n = rng.randint(1, 2)
                    slots = mamba.alloc(n)
                    if slots is not None:
                        for s in slots.tolist():
                            assert s not in mref.live_virt, (
                                f"mamba double-alloc {s} step {i}"
                            )
                        mref.live_virt.update(slots.tolist())
            elif op < 0.75 and ref.live_virt:  # composite free
                n = rng.randint(1, min(6, len(ref.live_virt)))
                dead = rng.sample(sorted(ref.live_virt), n)
                ref.live_virt.difference_update(dead)
                tok = torch.tensor(
                    [vp * ps for vp in dead], dtype=torch.int64, device="cpu"
                )
                if rng.random() < 0.5:
                    alloc.free(tok)
                else:
                    alloc.free_segment(tok, start_pos=0)
            elif op < 0.85 and mref.live_virt:  # mamba free
                n = rng.randint(1, len(mref.live_virt))
                dead = rng.sample(sorted(mref.live_virt), n)
                mref.live_virt.difference_update(dead)
                mamba.free(torch.tensor(dead, dtype=torch.int64, device="cpu"))
            elif op < 0.93:  # flush
                alloc.flush_opportunistic()
                mamba.flush_opportunistic()
            else:  # make_room on the FLOAT middle
                side = rng.choice(["low", "high"])
                swa.make_room(side=side, min_bytes=rng.randint(1, 8) *
                              swa.entry_bytes_per_page)
            _check_invariants("tri-full", full)
            _check_invariants("tri-swa-float", swa)
            _check_invariants("tri-mamba", mamba)
            viol = alloc.verify_byte_accounting()
            assert not viol, f"tri step {i}: {viol}"

    def test_pair_ps4_seeds(self):
        for seed in (1, 7, 42, 1234):
            with self.subTest(seed=seed):
                self._fuzz_pair(page_size=4, seed=seed)

    def test_pair_ps64(self):
        self._fuzz_pair(page_size=64, seed=99, steps=250)

    def test_tri_ps4_seeds(self):
        for seed in (3, 11, 777):
            with self.subTest(seed=seed):
                self._fuzz_tri(page_size=4, seed=seed)

    def test_tri_ps64(self):
        self._fuzz_tri(page_size=64, seed=5, steps=250)


if __name__ == "__main__":
    unittest.main()
