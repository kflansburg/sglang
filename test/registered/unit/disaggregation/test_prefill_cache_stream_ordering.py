import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPrefillCacheStreamOrdering(CustomTestCase):
    def test_overlap_loop_orders_cache_mutations_after_each_forward(self):
        first_batch = MagicMock(name="first_batch")
        second_batch = MagicMock(name="second_batch")
        first_result = MagicMock(name="first_result")
        second_result = MagicMock(name="second_result")
        calls = MagicMock()
        scheduler = SimpleNamespace(
            request_receiver=MagicMock(),
            process_input_requests=MagicMock(),
            _engine_paused=False,
            waiting_queue=[],
            disagg_prefill_bootstrap_queue=MagicMock(),
            _apply_war_barrier=MagicMock(),
            get_next_disagg_prefill_batch_to_run=MagicMock(),
            running_batch=None,
            last_batch=None,
            ngram_embedding_manager=MagicMock(),
            chunked_req=None,
            enable_staging=False,
            run_batch=calls.run_batch,
            schedule_stream=MagicMock(),
            forward_stream=MagicMock(name="forward_stream"),
            process_batch_result=calls.process_result,
            process_disagg_prefill_inflight_queue=MagicMock(),
            launch_batch_sample_if_needed=MagicMock(),
            on_idle=MagicMock(),
        )
        scheduler.request_receiver.recv_requests.side_effect = [[], [], StopIteration]
        scheduler.disagg_prefill_bootstrap_queue.pop_bootstrapped.return_value = []
        scheduler.get_next_disagg_prefill_batch_to_run.side_effect = [
            SimpleNamespace(batch_to_run=first_batch, running_batch=first_batch),
            SimpleNamespace(batch_to_run=second_batch, running_batch=second_batch),
        ]
        scheduler.ngram_embedding_manager.prepare_for_forward.side_effect = (
            lambda batch, **_: batch
        )
        scheduler.run_batch.side_effect = [first_result, second_result]
        scheduler.schedule_stream.wait_stream = calls.wait_stream

        with self.assertRaises(StopIteration):
            SchedulerDisaggregationPrefillMixin.event_loop_overlap_disagg_prefill(
                scheduler
            )

        self.assertEqual(
            calls.mock_calls,
            [
                unittest.mock.call.run_batch(first_batch),
                unittest.mock.call.wait_stream(scheduler.forward_stream),
                unittest.mock.call.run_batch(second_batch),
                unittest.mock.call.wait_stream(scheduler.forward_stream),
                unittest.mock.call.process_result(
                    first_batch.copy.return_value, first_result
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
