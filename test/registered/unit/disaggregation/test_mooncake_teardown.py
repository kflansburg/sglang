"""Regression tests for mooncake teardown safety (F3).

teardown() used to join worker threads with timeout=3s and proceed to close
sockets, destroy the zmq context, and deregister memory even when a worker
was still alive inside a transfer engine call — deregistering memory while
the engine has in-flight RDMA on it. Now: a shared join deadline (default
45s, env-tunable) must exceed the engine's batch wall-clock timeout, and if
any thread is still alive the teardown skips deregistration (leak beats
use-after-free); the engine wrapper dedups re-registration so the next
KVManager does not fail on double registration.
"""

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _bare_manager():
    mgr = object.__new__(MooncakeKVManager)
    mgr._stopped = False
    mgr._worker_threads = []
    mgr._transfer_worker_threads = []
    mgr.transfer_queues = []
    mgr.executors = []
    mgr._socket_lock = threading.Lock()
    mgr._socket_cache = {}
    mgr._monitor_cache = {}
    mgr._zmq_ctx = Mock()
    mgr.server_socket = Mock()
    mgr.connection_pool = {}
    mgr.connection_lock = threading.Lock()
    mgr.disaggregation_mode = SimpleNamespace(value="prefill")
    mgr.kv_args = SimpleNamespace(
        kv_data_ptrs=[0x1000],
        kv_data_lens=[1024],
        aux_data_ptrs=[0x5000],
        aux_data_lens=[16],
        state_data_ptrs=[],
        state_data_lens=[],
    )
    mgr.engine = Mock()
    mgr.engine.batch_deregister = Mock(return_value=0)
    return mgr


class TestMooncakeTeardown(CustomTestCase):
    def test_clean_exit_deregisters(self):
        mgr = _bare_manager()
        done = threading.Event()

        def worker():
            done.wait(0.05)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        mgr._worker_threads = [t]
        with patch.object(
            envs.SGLANG_DISAGGREGATION_TEARDOWN_JOIN_TIMEOUT_S, "get",
            return_value=5.0,
        ):
            mgr.teardown()
        # Worker exited within budget; deregistration ran.
        mgr.engine.batch_deregister.assert_called_once_with([0x1000, 0x5000])

    def test_stuck_worker_skips_deregister(self):
        mgr = _bare_manager()
        release = threading.Event()

        def worker():
            release.wait(30.0)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        mgr._worker_threads = [t]
        with patch.object(
            envs.SGLANG_DISAGGREGATION_TEARDOWN_JOIN_TIMEOUT_S, "get",
            return_value=0.1,
        ):
            start = time.monotonic()
            mgr.teardown()
            elapsed = time.monotonic() - start
        # Gave up on the wedge quickly and did NOT deregister the regions.
        mgr.engine.batch_deregister.assert_not_called()
        self.assertGreaterEqual(elapsed, 0.1)
        self.assertLess(elapsed, 5.0)
        release.set()


class TestMooncakeEngineRegionTracking(CustomTestCase):
    def _engine(self):
        from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
            MooncakeTransferEngine,
        )

        engine = object.__new__(MooncakeTransferEngine)
        engine._registered_regions = set()
        engine._registered_regions_lock = threading.Lock()
        engine.engine = Mock()
        engine.engine.batch_register_memory = Mock(return_value=0)
        engine.engine.batch_unregister_memory = Mock(return_value=0)
        return engine

    def test_register_dedups_known_regions(self):
        engine = self._engine()
        self.assertEqual(engine.batch_register([0x1000], [1024]), 0)
        self.assertEqual(engine.batch_register([0x1000], [1024]), 0)
        # Second call hit the wrapper's dedup, not the engine.
        engine.engine.batch_register_memory.assert_called_once_with([0x1000], [1024])

    def test_deregister_then_register(self):
        engine = self._engine()
        engine.batch_register([0x1000], [1024])
        engine.batch_deregister([0x1000])
        engine.batch_register([0x1000], [1024])
        self.assertEqual(engine.engine.batch_register_memory.call_count, 2)
        engine.engine.batch_unregister_memory.assert_called_once_with([0x1000])

    def test_failed_register_not_deduped(self):
        engine = self._engine()
        engine.engine.batch_register_memory.return_value = -1
        engine.batch_register([0x1000], [1024])
        engine.batch_register([0x1000], [1024])
        self.assertEqual(engine.engine.batch_register_memory.call_count, 2)


if __name__ == "__main__":
    unittest.main()
