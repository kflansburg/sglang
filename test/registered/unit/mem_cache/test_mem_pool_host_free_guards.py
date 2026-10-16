"""Regression tests for HiCache host-pool free-list guards.

Pins two audit findings (hostpool-audit, board #25):

- F1: host-pool free() overrides (MambaPoolHost, DeepSeekV4PagedHostPool,
  LogicalHostPool) accepted double frees silently; a doubly-freed slot is
  later handed to TWO live owners -> cross-request host KV corruption.
- F2: no free() path rejected -1 sentinel / out-of-range indices; a -1
  wrapped to the LAST slot (slot_used[-1]), marking a live slot free, and
  was later issued as a valid slot id, aliasing the final host row.

The tests use __new__-style stubs (no host buffers) so they run in small
sandboxes; a loader fallback exercises the same source when the queue venv
cannot import the sglang package.
"""

import os
import threading
import unittest

import torch

_QUEUE_SAFE = False
try:
    from sglang.srt.mem_cache.memory_pool_host import (
        DeepSeekV4PagedHostPool,
        LogicalHostPool,
    )
    from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase

    register_cpu_ci(est_time=5, suite="base-a-test-cpu")
    _QUEUE_SAFE = True
except ImportError:  # thin merge-queue venv: load real methods from source
    import importlib.util

    _HERE = os.path.dirname(os.path.abspath(__file__))
    _spec = importlib.util.spec_from_file_location(
        "_pool_free_guard_loader", os.path.join(_HERE, "_pool_free_guard_loader.py")
    )
    _loader = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_loader)
    CustomTestCase = unittest.TestCase


if _QUEUE_SAFE:

    def _make_logical_pool():
        return LogicalHostPool(size=8, page_size=2)

    def _make_dsv4_pool():
        pool = DeepSeekV4PagedHostPool.__new__(DeepSeekV4PagedHostPool)
        pool.size = 8
        pool.slot_page_size = 2
        pool.lock = threading.RLock()
        pool.clear()
        return pool

    def _make_mamba_pool():
        pool = MambaPoolHost.__new__(MambaPoolHost)
        pool.size = 8
        pool.page_size = 1
        pool.device = "cpu"
        pool.lock = threading.RLock()
        pool.clear()
        return pool

    def _make_mha_stub_pool():
        pool = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
        pool.size = 8
        pool.dcp_size = 1
        pool.page_size = 2
        pool.device = "cpu"
        pool.lock = threading.RLock()
        pool.clear()
        return pool

else:
    _makers_ns = None

    def _loader_makers():
        global _makers_ns
        if _makers_ns is None:
            _makers_ns = _loader.load()
        return _makers_ns

    def _class(name):
        for cls, make in _loader_makers().items():
            if cls.__name__ == name:
                return make
        raise KeyError(name)

    def _make_logical_pool():
        return _class("LogicalHostPool")()

    def _make_dsv4_pool():
        return _class("DeepSeekV4PagedHostPool")()

    def _make_mamba_pool():
        return _class("MambaPoolHost")()

    def _make_mha_stub_pool():
        return _class("MHATokenToKVPoolHost")()


class TestHostPoolFreeGuards(CustomTestCase):
    """All four host-pool families must detect double frees and sentinels."""

    def _assert_double_free_detected(self, make_pool):
        pool = make_pool()
        allocated = pool.alloc(4)
        self.assertIsNotNone(allocated)
        pool.free(allocated[:2])
        # allocated[1] has already been freed above.
        with self.assertRaisesRegex(AssertionError, "Double-free"):
            pool.free(allocated[1:3])

    def _assert_free_rejects_sentinels(self, make_pool):
        pool = make_pool()
        allocated = pool.alloc(4)
        self.assertIsNotNone(allocated)
        bad = allocated.clone()
        bad[0] = -1
        with self.assertRaisesRegex(AssertionError, "negative"):
            pool.free(bad)
        too_big = allocated.clone()
        too_big[0] = pool.size
        with self.assertRaisesRegex(AssertionError, "out of range"):
            pool.free(too_big)

    def _assert_lazy_release_preserved(self, make_pool, pass_mha=False):
        """The guards must not change the lazy free-list merge behavior."""
        pool = make_pool()
        self.assertEqual(pool.free(torch.empty(0, dtype=torch.int64)), 0)
        allocated = pool.alloc(6)
        free_slots_before = pool.free_slots
        pool.free(allocated[:2])
        self.assertIs(pool.free_slots, free_slots_before)
        self.assertEqual(pool.num_release_slots, 2)
        self.assertEqual(pool.available_size(), 4)
        self.assertTrue(torch.equal(pool.alloc(2), torch.tensor([6, 7])))
        self.assertTrue(torch.equal(pool.alloc(2), torch.tensor([0, 1])))
        self.assertEqual(pool.num_release_slots, 0)

    def test_mamba_pool_free_guards(self):
        self._assert_double_free_detected(_make_mamba_pool)
        self._assert_free_rejects_sentinels(_make_mamba_pool)
        self._assert_lazy_release_preserved(_make_mamba_pool)

    def test_deepseek_v4_pool_free_guards(self):
        self._assert_double_free_detected(_make_dsv4_pool)
        self._assert_free_rejects_sentinels(_make_dsv4_pool)
        self._assert_lazy_release_preserved(_make_dsv4_pool)

    def test_logical_pool_free_guards(self):
        self._assert_double_free_detected(_make_logical_pool)
        self._assert_free_rejects_sentinels(_make_logical_pool)
        self._assert_lazy_release_preserved(_make_logical_pool)

    def test_base_mha_pool_free_guards(self):
        self._assert_double_free_detected(_make_mha_stub_pool)
        self._assert_free_rejects_sentinels(_make_mha_stub_pool)
        self._assert_lazy_release_preserved(_make_mha_stub_pool)


if __name__ == "__main__":
    unittest.main()
