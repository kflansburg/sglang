"""Regression tests for V4-decode: device-destination deferred KV release.

Bug: resolve_deferred_releases freed device-destination pages at the
deferred_kv_release_timeout deadline even when the prefill drain ack had not
arrived ("releasing anyway"). A lost AbortNotification or a slow multi-chunk
transfer then let a still-writing prefill RDMA write land in pages already
reissued to a new request (cross-request KV corruption).

Fix: device destinations are held until drain, exactly like host
destinations; the deadline only rate-limits a re-log. The only early release
is the prefill instance being declared down (popped from
kv_mgr.prefill_info_table), because a dead engine has no in-flight writes.
"""
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

BOOTSTRAP_ADDR = "10.0.0.1:8998"


def _make_queue(prefill_alive=True):
    """Build a minimal DecodeTransferQueue with one armed deferred release."""
    req = SimpleNamespace(rid="aborted-mid-transfer", bootstrap_room=42)
    receiver = MagicMock()
    receiver.bootstrap_addr = BOOTSTRAP_ADDR
    receiver.kv_mgr.is_abort_release_safe.return_value = False
    receiver.kv_mgr.prefill_info_table = (
        {BOOTSTRAP_ADDR: object()} if prefill_alive else {}
    )
    decode_req = SimpleNamespace(
        req=req,
        kv_receiver=receiver,
        host_staged=False,
        metadata_buffer_index=3,
    )

    queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
    queue.enable_host_receive = False
    queue.enable_staging = False
    queue.deferred_kv_release_timeout = 30.0
    queue.gloo_group = MagicMock()
    queue.tree_cache = MagicMock()
    queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
    queue.req_to_metadata_buffer_idx_allocator = MagicMock()
    queue.scheduler = MagicMock()
    queue.scheduler.metrics_reporter.enable_metrics = False
    # Arm one deferred entry whose deadline is already in the past.
    queue._deferred_releases = [
        (decode_req, time.monotonic() - 31.0, time.monotonic() - 1.0, 3, 1)
    ]
    queue._release_request = MagicMock()
    return queue, decode_req, receiver


class TestDeferredReleaseTimeoutHold(CustomTestCase):
    def test_undrained_past_deadline_is_held_not_freed(self):
        queue, decode_req, receiver = _make_queue(prefill_alive=True)

        queue.resolve_deferred_releases()

        # Not released: pages stay out of the allocator.
        queue._release_request.assert_not_called()
        receiver.clear.assert_not_called()
        queue.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
        # Entry still armed; re-log deadline pushed into the future.
        self.assertEqual(len(queue._deferred_releases), 1)
        _, _, deadline, idx, acks = queue._deferred_releases[0]
        self.assertEqual((idx, acks), (3, 1))
        self.assertGreater(deadline, time.monotonic())

    def test_held_entry_released_once_drain_ack_lands(self):
        queue, decode_req, receiver = _make_queue(prefill_alive=True)

        queue.resolve_deferred_releases()
        self.assertEqual(len(queue._deferred_releases), 1)

        receiver.kv_mgr.is_abort_release_safe.return_value = True
        queue.resolve_deferred_releases()

        self.assertEqual(queue._deferred_releases, [])
        queue._release_request.assert_called_once_with(decode_req)
        receiver.clear.assert_called_once_with()
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)

    def test_undrained_released_when_prefill_instance_declared_down(self):
        queue, decode_req, receiver = _make_queue(prefill_alive=False)

        queue.resolve_deferred_releases()

        self.assertEqual(queue._deferred_releases, [])
        queue._release_request.assert_called_once_with(decode_req)
        receiver.clear.assert_called_once_with()
        receiver.kv_mgr.clear_deferred_abort_state.assert_called_once_with(42)

    def test_relog_is_rate_limited_until_next_deadline(self):
        queue, decode_req, receiver = _make_queue(prefill_alive=True)

        queue.resolve_deferred_releases()
        held_deadline = queue._deferred_releases[0][2]
        first_warn_deadline = held_deadline

        # Before the new deadline no re-log fires; the deadline is stable.
        queue.resolve_deferred_releases()
        self.assertEqual(queue._deferred_releases[0][2], first_warn_deadline)


if __name__ == "__main__":
    unittest.main()
