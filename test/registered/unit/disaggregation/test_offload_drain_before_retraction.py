"""
Unit tests for DecodeKVCacheOffloadManager.drain_before_retraction.

Regression coverage for the fix: previously, any pending offload D2H copy at
retraction time raised RuntimeError and crashed the decode scheduler
(OOM-retract racing a normal offload copy). The drain must now WAIT
(bounded) for pending copies to finish before refusing.

Requires: torch, sglang (CPU only).
"""

import time
import unittest
from unittest.mock import MagicMock

from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_manager(ack_write_queue, ongoing_offload, offload_inflight, tp_world_size=1):
    mgr = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
    mgr.cache_controller = MagicMock()
    mgr.cache_controller.ack_write_queue = ack_write_queue
    mgr.ongoing_offload = ongoing_offload
    mgr.offload_inflight = offload_inflight
    mgr.tp_world_size = tp_world_size
    # check_offload_progress() is exercised via its real code path; the mocked
    # cache controller exposes the queues it reads. ack pops are irrelevant for
    # these tests, so stub the worker parts of the progress check instead:
    mgr.check_offload_progress = MagicMock()
    return mgr


class TestDrainBeforeRetraction(unittest.TestCase):
    def test_no_pending_returns_immediately(self):
        mgr = _make_manager([], {}, {})
        # Must not raise.
        mgr.drain_before_retraction()

    def test_pending_copy_drains_then_returns(self):
        mgr = _make_manager([], {"ack-1": object()}, {})

        def clear_after_first_poll(*args, **kwargs):
            mgr.ongoing_offload.clear()

        # First progress check clears the copy, second loop iteration sees none.
        mgr.check_offload_progress.side_effect = clear_after_first_poll
        start = time.monotonic()
        mgr.drain_before_retraction()
        # One 1ms sleep at most: well under the default 30s timeout.
        assert time.monotonic() - start < 5.0

    def test_wedged_copy_raises_after_timeout(self):
        mgr = _make_manager([], {"ack-1": object()}, {})

        from sglang.srt.environ import envs

        # Shrink the wait so the test is fast.
        with envs.SGLANG_DECODE_OFFLOAD_DRAIN_TIMEOUT_S.override(0.2):
            with self.assertRaises(RuntimeError) as ctx:
                mgr.drain_before_retraction()
        assert "still pending" in str(ctx.exception)


if __name__ == "__main__":
    unittest.main()
