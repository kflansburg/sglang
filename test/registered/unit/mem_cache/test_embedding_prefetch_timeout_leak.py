"""Regression PoC for candidate C1: a wedged Mooncake batch_get wedges the
EmbeddingCacheController's single _io_loop; its FILLING entries' pages are
never freed and are un-evictable, permanently shrinking the pool, and the
content hash becomes a permanent cache miss (dedup keeps the FILLING entry).

Current code FAILS the pool-recovery assertions; it passes once a cancel /
deadline path frees timed-out FILLING entries.
"""

import threading
import time
import unittest
from queue import Queue
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.mem_cache.embedding_cache_controller import (
    EmbeddingCacheController,
    EvictableLRU,
    EntryState,
)
from sglang.test.ci.ci_register import register_cpu_ci

from test_embedding_cache_controller import _make_controller

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestPrefetchWedgeLeak(unittest.TestCase):
    def test_wedged_get_leaks_filling_pages_and_hash(self):
        ctrl = _make_controller(num_pages=8, dim=4, page_size=2)
        gate = threading.Event()

        def wedged_batch_get(keys, ptrs, sizes):
            # Simulates a Mooncake store call that never returns.
            gate.wait(timeout=30)
            return [True] * len(keys)

        ctrl.embedding_store.batch_get_into_multi_buffers = wedged_batch_get
        ctrl.embedding_store.batch_put_from_multi_buffers = MagicMock(
            return_value=[True]
        )

        free0 = ctrl.vision_pool.allocator.free_pages

        # _io_loop in a thread, exactly like the controller starts it.
        io_thread = threading.Thread(target=ctrl._io_loop, daemon=True)
        io_thread.start()

        ctrl.prefetch("req1", ["h1"], [3], Modality.IMAGE)
        # Give the io loop a moment to pick the op and wedge inside batch_get.
        deadline = time.time() + 5
        while time.time() < deadline:
            with ctrl.lock:
                e = ctrl.entries.get("h1")
                gone_filling = e is not None and e.state == EntryState.FILLING
            if gone_filling and "req1" in ctrl.ongoing_prefetch:
                break
            time.sleep(0.01)

        with ctrl.lock:
            e1 = ctrl.entries.get("h1")
            assert e1 is not None and e1.state == EntryState.FILLING
            free_after_alloc = ctrl.vision_pool.allocator.free_pages
        assert (
            free_after_alloc < free0
        ), "prefetch must allocate pages; test setup broken"

        # Encoder-server timeout side (server.py::_wait_global_cache_prefetch):
        # req1 gives up on the prefetch. There is no cancel API; the op stays.
        # (If a cancel API is added by the fix, call it here instead.)
        cancel = getattr(ctrl, "cancel_prefetch", None)
        if cancel is not None:
            cancel("req1")

        # The same content hash can never be re-driven: prefetch and
        # store_to_pool_async both dedup onto the stuck FILLING entry.
        ctrl.prefetch("req2", ["h1"], [3], Modality.IMAGE)
        handles = ctrl.store_to_pool_async(
            ["h1"], [torch.zeros((3, 4), dtype=torch.float32)], Modality.IMAGE
        )
        self.assertEqual(handles, [], "FILLING dedup silently skips the store")
        self.assertFalse(ctrl.has_local_embedding("h1"))

        # The FILLING entry is not evictable: simulate pool pressure.
        with ctrl.lock:
            ctrl._evict_for_pool(ctrl.vision_pool, ctrl.vision_pool.num_pages)
            free_after_evict = ctrl.vision_pool.allocator.free_pages

        # Stop the io loop and check what the pool looks like afterwards.
        ctrl.stop_event.set()

        if cancel is None:
            # Current behavior: pages are never recovered while the io loop is
            # wedged -- assert the leak so the test documents it. After the fix
            # lands (cancel_prefetch or an _io_loop deadline), this branch is
            # unreachable.
            self.assertEqual(
                free_after_evict,
                free_after_alloc,
                "FILLING entry must not be evictable (leak if io loop wedges)",
            )
        else:
            self.assertEqual(
                free_after_evict,
                free0,
                "cancelling the wedged prefetch must free its pages",
            )


if __name__ == "__main__":
    unittest.main()
