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


class TestStagingWireFieldGuards(unittest.TestCase):
    """F20/F21: staging wire-decoded fields must not alias or regress the ring."""

    def test_set_chunk_negative_index_raises(self):
        from sglang.srt.disaggregation.common.staging_handler import (
            StagingTransferInfo,
        )

        t = StagingTransferInfo()
        t.set_chunk(0, 100, 1, 200)
        t.set_chunk(1, 300, 1, 400)
        with self.assertRaises(ValueError):
            t.set_chunk(-1, 999, 9, 9999)
        # The legitimate entries are untouched.
        self.assertEqual(t.offsets, [100, 300])
        self.assertEqual(t.rounds, [1, 1])
        self.assertEqual(t.ends, [200, 400])

    def test_handle_staging_rsp_drops_negative_chunk(self):
        from types import SimpleNamespace

        from sglang.srt.disaggregation.common.staging_handler import (
            StagingTransferInfo,
            handle_staging_rsp,
        )

        tinfo = SimpleNamespace(staging=StagingTransferInfo())
        transfer_infos = {5: {"sess": tinfo}}
        msg = [b"STAGING_RSP", b"5", b"-1", b"100", b"1", b"200", b"sess"]
        handle_staging_rsp(msg, transfer_infos)
        # Nothing overwritten through the negative-index alias.
        self.assertEqual(tinfo.staging.offsets, [-1])

    def test_handle_staging_rsp_positive_still_applies(self):
        from types import SimpleNamespace

        from sglang.srt.disaggregation.common.staging_handler import (
            StagingTransferInfo,
            handle_staging_rsp,
        )

        tinfo = SimpleNamespace(staging=StagingTransferInfo())
        transfer_infos = {5: {"sess": tinfo}}
        msg = [b"STAGING_RSP", b"5", b"1", b"100", b"1", b"200", b"sess"]
        handle_staging_rsp(msg, transfer_infos)
        self.assertEqual(tinfo.staging.offsets, [-1, 100])

    def test_allocator_rejects_nonpositive_size(self):
        from sglang.srt.disaggregation.common.staging_buffer import (
            StagingAllocator,
        )

        for bad in (0, -1, -4096):
            with self.assertRaises(ValueError):
                StagingAllocator.assign(
                    # Bypass __init__ (touches CUDA); only the guarded branch
                    # runs before any CUDA or buffer state is used.
                    SimpleNamespace(),
                    bad,
                )

    def test_watermark_drops_out_of_range(self):
        from sglang.srt.disaggregation.common.staging_handler import (
            handle_watermark_msg,
        )

        ctx = SimpleNamespace(
            remote_watermarks={}, watermark_cv=__import__("threading").Condition()
        )
        handle_watermark_msg(ctx, [b"WATERMARK", b"-3", b"10", b"s"])
        handle_watermark_msg(ctx, [b"WATERMARK", b"3", b"-10", b"s"])
        self.assertEqual(ctx.remote_watermarks, {})
        handle_watermark_msg(ctx, [b"WATERMARK", b"3", b"10", b"s"])
        self.assertEqual(ctx.remote_watermarks["s"], (3, 10))


if __name__ == "__main__":
    unittest.main()
