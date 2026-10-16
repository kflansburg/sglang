"""Regression test for F2: DecodeStagingHandler must prune its watermark
subscriber when the last room using that prefill session unregisters. Before
the fix, _wm_subscribers grew by one entry per staging request forever, and
every staging free broadcast ZMQ watermarks to long-dead peers.

(The AuxDataCodec bounds regression coverage for F12 lives with the mooncake
F12 fix — see board items #42/#44 and their test files.)
"""

import unittest
import unittest.mock
from types import SimpleNamespace

from sglang.srt.disaggregation.common.staging_handler import DecodeStagingHandler


class TestWatermarkSubscriberPruning(unittest.TestCase):
    def _make_handler(self):
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
        return DecodeStagingHandler(
            kv_manager=kv_manager,
            staging_allocator=allocator,
            kv_buffer_info=kv_manager.kv_buffer_tensors,
            decode_tp=2,
            total_kv_heads=4,
            tp_rank=0,
            scheduler=SimpleNamespace(),
        )

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

    def test_unregister_without_registration_is_safe(self):
        handler = self._make_handler()
        handler.unregister_decode_req(999)  # no room, no subscriber
        self.assertEqual(len(handler._wm_subscribers), 0)


if __name__ == "__main__":
    unittest.main()
