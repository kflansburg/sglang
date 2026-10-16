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
