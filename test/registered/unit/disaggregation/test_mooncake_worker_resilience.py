"""Regression tests for Mooncake transfer-worker fault isolation (F4).

A failure while processing one chunk used to kill the (daemon) transfer
worker thread: nothing restarted it, its queue kept growing, and every room
hashing to that shard then hung until the 300s waiting timeout. The worker
must now fail that chunk's room and keep serving the rest of the shard, and
add_transfer_request must refuse to enqueue onto a dead shard.
"""

import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.utils import FastQueue, TransferKVChunk
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, TransferInfo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMooncakeWorkerResilience(CustomTestCase):
    def _make_manager(self):
        manager = object.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[0x1000],
            state_types=[],
            state_data_ptrs=[],
            state_item_lens=[],
            state_dim_per_tensor=[],
            state_layer_ids=[],
            aux_data_ptrs=[0x5000],
            aux_item_lens=[8],
        )
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.attn_tp_size = manager.attn_cp_size = manager.pp_size = 1
        manager.attn_tp_rank = manager.attn_cp_rank = manager.pp_rank = 0
        manager.enable_trace = manager.enable_staging = False
        manager.enable_custom_mem_pool = False
        manager.max_transfer_batch_indices = 0
        manager.bootstrap_port = 1234
        manager.session_lock = threading.Lock()
        manager.failure_lock = threading.Lock()
        manager.failed_sessions = set()
        manager.session_failures = defaultdict(int)
        manager.state_layout_rejections = {}
        manager.state_strides_validated = set()
        manager.failure_records = {}
        manager._staging_outstanding = defaultdict(int)
        manager.request_status = {}
        manager.req_to_decode_prefix_len = {}
        manager.transfer_infos = {}
        manager.decode_kv_args_table = {}
        manager.deferred_bootstrap = None
        manager._deferred_ack_targets = {}
        manager._deferred_ack_poisoned_rooms = set()
        manager._dcp_pack_buffers = None
        manager.enable_deferred_decode_kv_release = True
        manager.defer_decode_allocation = False
        manager._send_multipart_locked = Mock()
        from sglang.srt.disaggregation.utils import DisaggregationMode

        manager.disaggregation_mode = DisaggregationMode.PREFILL
        return manager

    def _make_room(self, manager, room):
        req = TransferInfo(
            room=room,
            endpoint="127.0.0.1",
            dst_port=1234,
            mooncake_session_id=f"decode:{room}",
            dst_kv_indices=np.array([20, 21, 22, 23], dtype=np.int32),
            dst_aux_index=5,
            dst_state_indices=[],
            required_dst_info_num=1,
            is_dummy=False,
        )
        manager.transfer_infos[room] = {req.mooncake_session_id: req}
        manager.decode_kv_args_table[req.mooncake_session_id] = SimpleNamespace(
            requires_dcp_relayout=False,
            dst_kv_ptrs=[0x2000],
            dst_kv_layer_ids=[0],
            dst_kv_item_len=64,
            dst_state_data_ptrs=[],
            dst_state_item_lens=[],
            dst_state_dim_per_tensor=[],
            dst_state_layer_ids=[],
            dst_attn_tp_size=1,
            dst_aux_ptrs=[0x6000],
        )
        manager.request_status[room] = KVPoll.Transferring
        manager.req_to_decode_prefix_len[room] = 0

    def _make_chunk(self, room):
        return TransferKVChunk(
            room=room,
            prefill_kv_indices=np.array([10, 11, 12, 13], dtype=np.int32),
            index_slice=slice(0, 4),
            is_last_chunk=True,
            prefill_aux_index=2,
            state_indices=None,
        )

    def test_worker_survives_chunk_error_and_serves_next_room(self):
        manager = self._make_manager()
        self._make_room(manager, 1)
        self._make_room(manager, 2)
        manager.send_aux = Mock(return_value=0)
        manager.send_kvcache = Mock(
            side_effect=lambda session, *args, **kwargs: (
                (_ for _ in ()).throw(ValueError("injected engine fault"))
                if session == "decode:1"
                else 0
            )
        )

        queue = FastQueue()
        queue.put(self._make_chunk(1))
        queue.put(self._make_chunk(2))
        queue.put(None)
        # Runs in the calling thread: if the old behavior regressed the first
        # chunk would raise RuntimeError out of transfer_worker.
        manager.transfer_worker(queue, executor=None, worker_index=0)

        # Room 1 failed loudly with the error recorded and drained accounting.
        self.assertEqual(manager.request_status.get(1), KVPoll.Failed)
        self.assertIn("injected engine fault", manager.failure_records[1])
        # Room 2, on the same shard, still completed successfully.
        self.assertEqual(manager.request_status.get(2), KVPoll.Success)
        # No outstanding-counting leak blocking later drain acks.
        self.assertFalse(manager._staging_outstanding)
        # The worker consumed every chunk (both rooms processed + sentinel).
        self.assertEqual(len(queue._buf), 0)
        self.assertEqual(manager.send_kvcache.call_count, 2)

    def test_add_transfer_request_refuses_dead_worker_shard(self):
        manager = self._make_manager()
        self._make_room(manager, 7)
        manager.transfer_queues = [FastQueue()]
        dead = Mock(spec=threading.Thread)
        dead.is_alive = Mock(return_value=False)
        manager._transfer_worker_threads = [dead]
        manager.send_kvcache = Mock(return_value=0)

        manager.add_transfer_request(
            bootstrap_room=7,
            kv_indices=np.array([10], dtype=np.int32),
            index_slice=slice(0, 1),
            is_last_chunk=False,
        )

        self.assertEqual(len(manager.transfer_queues[0]._buf), 0)
        self.assertEqual(manager.request_status.get(7), KVPoll.Failed)
        self.assertIn("dead", manager.failure_records[7])

    def test_add_transfer_request_enqueues_on_live_worker(self):
        manager = self._make_manager()
        self._make_room(manager, 8)
        manager.transfer_queues = [FastQueue()]
        live = Mock(spec=threading.Thread)
        live.is_alive = Mock(return_value=True)
        manager._transfer_worker_threads = [live]

        manager.add_transfer_request(
            bootstrap_room=8,
            kv_indices=np.array([10], dtype=np.int32),
            index_slice=slice(0, 1),
            is_last_chunk=False,
        )

        self.assertEqual(len(manager.transfer_queues[0]._buf), 1)
        self.assertEqual(manager.request_status.get(8), KVPoll.Transferring)


if __name__ == "__main__":
    unittest.main()
