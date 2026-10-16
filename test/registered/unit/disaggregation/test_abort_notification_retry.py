"""Unit tests for abort-notification retry in srt/disaggregation/common/conn.py.

Regression for the F8+F9 compound: decode's abort notification to prefill ranks
was one-shot and per-rank failures were swallowed at ``logger.debug``. A lost
notification means the prefill rank keeps RDMA-writing KV pages that decode may
reissue to later requests (cross-request corruption). The fix retries failed
sends with bounded backoff and tracks per-rank delivery.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

# NOTE: keep the import surface minimal — sglang.test.test_utils pulls the
# benchmark/datasets chain, which is not part of the CPU test env.
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVReceiver


class _ConcreteReceiver(CommonKVReceiver):
    def failure_exception(self):
        raise NotImplementedError


def _receiver(num_ranks: int, fail_ranks: set):
    """Build a receiver whose abort-notification send fails for ``fail_ranks``."""
    receiver = object.__new__(_ConcreteReceiver)
    receiver.bootstrap_room = 7
    receiver.bootstrap_infos = [
        {"rank_ip": f"10.0.0.{i}", "rank_port": 1000 + i} for i in range(num_ranks)
    ]
    receiver.abort_notified = False
    receiver._abort_generation = 3
    receiver._abort_notify_failed_ranks = []
    receiver._abort_notify_attempts = 0
    receiver._abort_notify_next_retry_ts = 0.0
    receiver.init_time = 1.0
    receiver.conclude_state = None
    receiver._owns_bootstrap_room = True
    receiver.kv_mgr = SimpleNamespace(
        enable_deferred_decode_kv_release=True,
        local_ip="192.0.2.1",
        rank_port=4567,
        deferred_bootstrap=None,
        record_failure=Mock(),
        update_status=Mock(),
        check_status=Mock(return_value=KVPoll.Transferring),
        register_deferred_abort_room=Mock(return_value=3),
    )

    sent = []

    class _Sock:
        def __init__(self, idx):
            self.idx = idx

        def send_multipart(self, msg):
            sent.append(self.idx)

    class _Lock:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_connect(bootstrap_info):
        idx = int(bootstrap_info["rank_ip"].rsplit(".", 1)[1])
        if idx in fail_ranks:
            raise ConnectionError(f"rank {idx} unreachable")
        return _Sock(idx), _Lock()

    receiver._connect_to_bootstrap_server = fake_connect
    receiver._sent = sent
    return receiver


class TestAbortNotificationRetry(unittest.TestCase):
    def test_failed_sends_are_retried_and_eventually_succeed(self):
        # Rank 1 fails once, then recovers.
        receiver = _receiver(num_ranks=3, fail_ranks={1})
        receiver._send_abort_notification()
        self.assertEqual(sorted(receiver._sent), [0, 2])
        self.assertEqual(receiver._abort_notify_failed_ranks, [1])
        self.assertEqual(receiver._abort_notify_attempts, 1)

        # Fold the network recovery in by clearing the failure set is not
        # possible (closure) -- instead emulate recovery by swapping connect.
        def ok_connect(bootstrap_info):
            idx = int(bootstrap_info["rank_ip"].rsplit(".", 1)[1])
            return receiver._connect_to_bootstrap_server.__wrapped__(bootstrap_info)

        # Rebind to a no-failure connect: only the failed rank is retried.
        sent2 = []

        class _Sock2:
            def send_multipart(self, msg):
                sent2.append(msg)

        class _Lock2:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        receiver._connect_to_bootstrap_server = lambda info: (_Sock2(), _Lock2())
        # Not yet due.
        receiver._retry_abort_notifications(now=receiver._abort_notify_next_retry_ts - 0.1)
        self.assertEqual(sent2, [])
        # Due: retry hits only rank 1.
        receiver._retry_abort_notifications(now=receiver._abort_notify_next_retry_ts + 0.1)
        self.assertEqual(len(sent2), 1)
        self.assertEqual(receiver._abort_notify_failed_ranks, [])

    def test_persistent_failure_gives_up_loudly_without_plumbing(self):
        receiver = _receiver(num_ranks=1, fail_ranks={0})
        receiver._send_abort_notification()
        self.assertEqual(receiver._abort_notify_failed_ranks, [0])
        with self.assertLogs(level="ERROR") as logs:
            for _ in range(10):
                receiver._retry_abort_notifications(
                    now=receiver._abort_notify_next_retry_ts + 1.0
                )
        self.assertTrue(any("Giving up" in line for line in logs.output))
        self.assertEqual(receiver._abort_notify_failed_ranks, [])
        # Generation gate untouched: _abort_generation stays set, so ack
        # tracking still requires acks that never arrive -> pages held.
        self.assertEqual(receiver._abort_generation, 3)

    def test_retry_is_driven_from_poll_for_failed_rooms(self):
        receiver = _receiver(num_ranks=1, fail_ranks={0})
        receiver._send_abort_notification()
        retried = []
        receiver._retry_abort_notifications = lambda now=None: retried.append(now)
        receiver._poll()
        self.assertEqual(len(retried), 1)

    def test_success_path_has_no_retry_state(self):
        receiver = _receiver(num_ranks=2, fail_ranks=set())
        receiver._send_abort_notification()
        self.assertEqual(sorted(receiver._sent), [0, 1])
        self.assertEqual(receiver._abort_notify_failed_ranks, [])


if __name__ == "__main__":
    unittest.main()
