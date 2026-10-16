# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""F0 hardening regressions: a stale (double/never-allocated) free must not
write out of bounds, must fail loud in debug mode, and the idle verifier must
report the corrupted state afterwards.

Covers: lazy END pool _free_lazy path, FLOAT pool free path, free-list value
checks in _byte_accounting_violations, CPU-reference kernel store guards, and
the host-sync-free contract of the production free paths.
"""
import os
import unittest
from unittest import mock

import torch

from sglang.test.unified_allocator_fixtures import (
    build_swa_pool,
    build_tri_pool,
    setup_allocator_context,
)

try:
    from sglang.test.ci.ci_register import register_cpu_ci

    register_cpu_ci(est_time=10, suite="base-a-test-cpu")
except Exception:
    pass

_DEBUG_ENV = {"SGLANG_DEBUG_MEMORY_POOL": "1"}


def _debug_env():
    # SGLANG_DEBUG_MEMORY_POOL is read through envs lazily hot-paths; patch os.environ
    # so the env var reads True during the block.
    return mock.patch.dict(os.environ, _DEBUG_ENV)


class TestStaleFreeHardening(unittest.TestCase):
    def setUp(self):
        setup_allocator_context()

    def test_debug_mode_stale_free_raises_end_pool(self):
        alloc = build_swa_pool(page_size=4)
        full = alloc.full_attn_allocator
        tok = alloc.alloc(16)
        alloc.free(tok.clone())
        with _debug_env():
            with self.assertRaises(AssertionError):
                alloc.free(tok)  # caller bug: same ids twice
        assert full._stats_n_stale_frees == 4
        # The filter raises BEFORE mutating counters / free lists: state clean.
        holes = full._free_phys_pages.tolist()
        assert all(h >= 0 for h in holes) and len(set(holes)) == len(holes)
        assert full.live_page_count == 0
        fv = full.free_virtual_ids.tolist()
        assert len(set(fv)) == len(fv), "virtual ids must not be double-recycled"
        assert full._byte_accounting_violations() == []

    def test_debug_mode_stale_free_raises_float_pool(self):
        bundle, alloc = build_tri_pool(page_size=4)
        swa = alloc.swa_attn_allocator  # FLOAT middle
        tok = alloc.alloc(16)
        alloc.free(tok.clone())
        with _debug_env():
            with self.assertRaises(AssertionError):
                alloc.free(tok)

    def test_default_mode_stale_free_stays_sync_free_and_unfiltered(self):
        """Pin the production default-mode contract on _free_lazy:

        With no debug flags the free path must (a) perform NO host sync (the
        item/unique ban), and (b) NOT filter the stale -1 -- the guard layers
        downstream (kernel store guards, idle byte-accounting value checks)
        are the documented catches. A future "always filter" change would
        either re-introduce the host sync test_unified_free_no_host_sync bans,
        or change accounting semantics; this fails on both.
        """
        assert not os.getenv("SGLANG_DEBUG_MEMORY_POOL")
        assert not os.getenv("SGLANG_ENABLE_ASYNC_ASSERT")
        alloc = build_swa_pool(page_size=4)
        full = alloc.full_attn_allocator
        tok = alloc.alloc(16)
        # `.item()` is the sync must be disclosed on; `torch.unique` is allowed
        # on the token-free path (pre-existing page-rep dedup, see _free_lazy).
        # An "always filter" change NEEDS a value read of freed_p_pages -> .any()
        # -> item, so banning item fails exactly the regression we pin.
        with mock.patch.object(
            torch.Tensor, "item", side_effect=AssertionError("item = host sync")
        ):
            alloc.free(tok.clone())
            alloc.free(tok)  # caller bug: same ids twice
        holes = full._free_phys_pages.tolist()
        assert any(h == -1 for h in holes), "default mode must admit the stale -1"
        assert full.live_page_count < 0
        viol = full._byte_accounting_violations()
        assert any("out-of-range hole" in v for v in viol), viol

    def test_pre_radix38_checkpoint_shrink_shape_is_contained(self):
        """Historical caller class for F0: the pre-#38 RadixCache.checkpoint
        shrink made release_kv_cache free rows the TREE still owned, and the
        tree's later eviction freed the same pages again. Allocator-level
        shape: free(v) -> page reallocated to a new request -> stale free(v)
        arrives again. In default mode the second free must not write OOB,
        must not evict the NEW owner's binding, and the idle verifier must
        flag the residual stale hole."""
        assert not os.getenv("SGLANG_DEBUG_MEMORY_POOL")
        assert not os.getenv("SGLANG_ENABLE_ASYNC_ASSERT")
        alloc = build_swa_pool(page_size=4)
        full = alloc.full_attn_allocator
        v_page_full = alloc.full_attn_allocator.virtual_to_physical

        tok = alloc.alloc(16)
        v_page = int(tok[0]) // 4
        p_after_first_alloc = v_page_full[v_page].item()
        alloc.free(tok.clone())
        # Tree still believes it owns the page; the page goes to a new request.
        tok2 = alloc.alloc(16)
        assert tok2 is not None
        # Stale second free from the new-request's perspective-old request:
        alloc.free(tok)  # duplicate free of the ORIGINAL ids
        # The new owner's binding for the same PHYSICAL page is intact: the
        # stale free could only tombstone pages the ORIGINAL virtuals map,
        # which are already -1, so no live binding is touched.
        v2 = full.virtual_to_physical[int(tok2[0]) // 4].item()
        assert v2 >= 0, "new owner's binding must survive the stale free"
        # p_after_first_alloc came back exactly once through alloc2 or is in
        # holes; either the verifier or the new binding accounts for it.
        viol = full._byte_accounting_violations()
        assert any("out-of-range hole" in v for v in viol), viol
        holes = full._free_phys_pages.tolist()
        assert holes.count(p_after_first_alloc) <= 1, (
            "same physical page must not be double-recycled"
        )

    def test_idle_verifier_reports_injected_stale(self):
        # Simulate pre-patch state to prove the new value checks are not blind
        # where the old count identity was.
        alloc = build_swa_pool(page_size=4)
        full = alloc.full_attn_allocator
        alloc.alloc(16)
        full._free_phys_pages = torch.cat(
            [full._free_phys_pages, torch.tensor([-1, -1], dtype=torch.int64)]
        )
        full.live_page_count -= 2
        viol = full._byte_accounting_violations()
        assert viol, "verifier blind to -1 holes / negative live count"
        joined = " | ".join(viol)
        assert "out-of-range hole" in joined or "live_page_count" in joined, joined

    def test_idle_verifier_clean_on_normal_run(self):
        alloc = build_swa_pool(page_size=4)
        t1 = alloc.alloc(16)
        alloc.free(t1.clone())
        alloc.free(alloc.alloc(8))
        assert alloc.verify_byte_accounting() == []

    def test_bind_inplace_cpu_guard(self):
        from sglang.kernels.ops.memory.virtual_slot import bind_inplace

        v2p = torch.full((9,), -1, dtype=torch.int64)
        p2v = torch.full((9,), -1, dtype=torch.int64)
        v2p[0] = 0
        p2v[0] = 0
        bind_inplace(
            torch.tensor([3, 4], dtype=torch.int64),
            torch.tensor([7, -1], dtype=torch.int64),
            v2p,
            p2v,
        )
        assert v2p[3].item() == 7
        assert v2p[4].item() == -1, "stale bind must leave v2p unbound"
        assert p2v[-1].item() == -1, "wrapped write to last p2v slot"

    def test_free_unbind_inplace_cpu_guard(self):
        from sglang.kernels.ops.memory.virtual_slot import free_unbind_inplace

        v2p = torch.full((9,), -1, dtype=torch.int64)
        p2v = torch.full((9,), -1, dtype=torch.int64)
        v2p[0] = 0
        p2v[0] = 0
        v2p[5] = 6
        p2v[6] = 5
        freed = free_unbind_inplace(
            torch.tensor([5, 7], dtype=torch.int64), v2p, p2v
        )
        assert freed.tolist() == [6, -1]
        assert p2v[6].item() == -1
        assert p2v[-1].item() == -1, "stale free must not wrap to last p2v slot"
        assert v2p[5].item() == -1 and v2p[7].item() == -1


if __name__ == "__main__":
    unittest.main()
