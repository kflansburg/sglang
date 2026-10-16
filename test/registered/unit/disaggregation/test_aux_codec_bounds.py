"""Regression tests for:

- F12: AuxDataCodec.deserialize_data_to_buffer must bounds-check the
  wire-provided buffer_index / aux_index / payload length against the
  registered aux buffer capacity (aux_data_lens / aux_item_lens). Before the
  fix, a stale or corrupt AUX_DATA message wrote aux_index*item_len bytes from
  the buffer start with no validation -> OOB write into a metadata slot now
  owned by a different request (cross-request corruption of output ids /
  logprobs / hidden states).
- F2: DecodeStagingHandler must prune its watermark subscriber when the last
  room using that prefill session unregisters; before the fix the subscriber
  dict grew by one entry per staging request forever and every staging free
  broadcast ZMQ watermarks to dead peers.
"""

import ctypes
import unittest
import unittest.mock
from types import SimpleNamespace

from sglang.srt.disaggregation.common.utils import AuxDataCodec


def _make_kv_args(num_slots: int = 4, item_len: int = 32, num_bufs: int = 2):
    backing = [ctypes.create_string_buffer(num_slots * item_len) for _ in range(num_bufs)]
    kv_args = SimpleNamespace(
        aux_data_ptrs=[ctypes.addressof(b) for b in backing],
        aux_item_lens=[item_len] * num_bufs,
        aux_data_lens=[num_slots * item_len] * num_bufs,
    )
    return kv_args, backing


def _read_index(kv_args, buffer_index, aux_index, length):
    ptr = kv_args.aux_data_ptrs[buffer_index]
    item_len = kv_args.aux_item_lens[buffer_index]
    buf = (ctypes.c_byte * length).from_address(ptr + item_len * aux_index)
    return bytes(buf)


class TestAuxDataCodecBounds(unittest.TestCase):
    def test_valid_write_roundtrip(self):
        kv_args, _ = _make_kv_args()
        payload = bytes(range(32))
        AuxDataCodec.deserialize_data_to_buffer(kv_args, 1, 3, payload)
        self.assertEqual(_read_index(kv_args, 1, 3, 32), payload)

    def test_first_and_last_slot(self):
        kv_args, _ = _make_kv_args()
        AuxDataCodec.deserialize_data_to_buffer(kv_args, 0, 0, b"a" * 32)
        AuxDataCodec.deserialize_data_to_buffer(kv_args, 0, 3, b"b" * 32)
        self.assertEqual(_read_index(kv_args, 0, 0, 32), b"a" * 32)
        self.assertEqual(_read_index(kv_args, 0, 3, 32), b"b" * 32)

    def test_aux_index_past_capacity_rejected(self):
        kv_args, backing = _make_kv_args(num_slots=4)
        before = ctypes.string_at(ctypes.addressof(backing[0]), 4 * 32)
        with self.assertRaises(ValueError):
            # aux_index 4 * item_len 32 == capacity 128 -> out of bounds
            AuxDataCodec.deserialize_data_to_buffer(kv_args, 0, 4, b"x" * 32)
        with self.assertRaises(ValueError):
            AuxDataCodec.deserialize_data_to_buffer(kv_args, 0, 10_000, b"x")
        after = ctypes.string_at(ctypes.addressof(backing[0]), 4 * 32)
        self.assertEqual(before, after)

    def test_negative_aux_index_rejected(self):
        kv_args, _ = _make_kv_args()
        with self.assertRaises(ValueError):
            AuxDataCodec.deserialize_data_to_buffer(kv_args, 0, -1, b"x" * 32)

    def test_buffer_index_out_of_range_rejected(self):
        kv_args, _ = _make_kv_args()
        with self.assertRaises(ValueError):
            AuxDataCodec.deserialize_data_to_buffer(kv_args, 2, 0, b"x")
        with self.assertRaises(ValueError):
            AuxDataCodec.deserialize_data_to_buffer(kv_args, -1, 0, b"x")

    def test_payload_larger_than_item_rejected(self):
        kv_args, _ = _make_kv_args(item_len=32)
        with self.assertRaises(ValueError):
            AuxDataCodec.deserialize_data_to_buffer(kv_args, 0, 0, b"x" * 33)

    def test_missing_aux_data_lens_falls_back_to_single_slot(self):
        kv_args, _ = _make_kv_args(num_slots=4)
        del kv_args.aux_data_lens
        AuxDataCodec.deserialize_data_to_buffer(kv_args, 0, 0, b"z" * 32)
        with self.assertRaises(ValueError):
            AuxDataCodec.deserialize_data_to_buffer(kv_args, 0, 1, b"z" * 32)


class TestWatermarkSubscriberPruning(unittest.TestCase):
    def _make_handler(self):
        from sglang.srt.disaggregation.common.staging_handler import (
            DecodeStagingHandler,
        )

        kv_manager = SimpleNamespace(
            pp_size=1,
            _staging_ctx=SimpleNamespace(room_receivers={}, room_bootstrap={}),
            kv_args=SimpleNamespace(engine_rank=0, total_kv_head_num=4),
            attn_tp_size=2,
            kv_buffer_tensors={
                "k_buffers": [],
                "v_buffers": [],
                "page_size": 64,
            },
        )
        allocator = SimpleNamespace(
            _scatter_stream=None,
            free=unittest.mock.Mock(),
            get_watermark=lambda: (0, 0),
        )
        handler = DecodeStagingHandler(
            kv_manager=kv_manager,
            staging_allocator=allocator,
            kv_buffer_info=kv_manager.kv_buffer_tensors,
            decode_tp=2,
            total_kv_heads=4,
            tp_rank=0,
            scheduler=SimpleNamespace(),
        )
        return handler

    def _receiver(self):
        return SimpleNamespace(
            bootstrap_infos=[{"rank_ip": "1.2.3.4", "rank_port": 9000}],
            chunk_staging_infos=[],
            prefill_info=SimpleNamespace(attn_tp_size=2, pp_size=1),
        )

    def test_subscriber_pruned_on_last_room_unregister(self):
        handler = self._make_handler()
        receiver = self._receiver()
        handler._room_to_receiver[7] = receiver
        handler.register_wm_subscriber(7, receiver, "sess")
        self.assertEqual(len(handler._wm_subscribers), 1)
        handler.unregister_decode_req(7)
        self.assertEqual(len(handler._wm_subscribers), 0)

    def test_subscriber_kept_while_other_room_shares_session(self):
        handler = self._make_handler()
        receiver = self._receiver()
        receiver2 = self._receiver()  # same bootstrap => same key
        handler._room_to_receiver[7] = receiver
        handler._room_to_receiver[8] = receiver2
        handler.register_wm_subscriber(7, receiver, "sess")
        handler.register_wm_subscriber(8, receiver2, "sess")
        self.assertEqual(len(handler._wm_subscribers), 1)
        handler.unregister_decode_req(7)
        self.assertEqual(len(handler._wm_subscribers), 1)
        handler.unregister_decode_req(8)
        self.assertEqual(len(handler._wm_subscribers), 0)


if __name__ == "__main__":
    unittest.main()
