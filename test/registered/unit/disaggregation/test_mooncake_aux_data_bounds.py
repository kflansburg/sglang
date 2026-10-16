"""F12 regression tests: AUX_DATA wire bounds (codec + handler contract).

The AUX_DATA wire message (mooncake backend, TCP aux path) carries its target
coordinates: room, buffer_index, aux_index, payload. Final design (union of
the two F12 halves): the codec (disagg-mooncake, board #42) raises ValueError
on any out-of-range coordinate or mismatched length, and
mooncake/conn.py:_handle_aux_data additionally drops messages for unknown or
concluded rooms before calling the codec (a room cleared from request_status
may have had its metadata slot re-issued to another request).

These tests pin the full current contract end-to-end and must be merged only
after the codec-side fix is on dev (board #42); on an older tree they fail.
"""
import ctypes
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.common.utils import AuxDataCodec
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

ROOM = 77
N_SLOTS = 3
ITEM_LEN = 8
N_BUFFERS = 2


def _aux_msg(room=ROOM, buffer_index=0, aux_index=0, payload=None):
    if payload is None:
        payload = b"\x01" * ITEM_LEN
    return [
        MooncakeKVManager.AUX_DATA_HEADER,
        str(room).encode("ascii"),
        str(buffer_index).encode("ascii"),
        str(aux_index).encode("ascii"),
        struct.pack(">I", len(payload)),
        payload,
    ]


def _buffers():
    """Real ctypes-backed aux buffers + kv_args pointing at them."""
    raw = [ctypes.create_string_buffer(N_SLOTS * ITEM_LEN) for _ in range(N_BUFFERS)]
    kv_args = SimpleNamespace(
        aux_data_ptrs=[ctypes.addressof(b) for b in raw],
        aux_data_lens=[N_SLOTS * ITEM_LEN] * N_BUFFERS,
        aux_item_lens=[ITEM_LEN] * N_BUFFERS,
    )
    return raw, kv_args


def _fake_manager():
    mgr = MooncakeKVManager.__new__(MooncakeKVManager)
    mgr.request_status = {ROOM: object()}
    _, mgr.kv_args = _buffers()
    return mgr


class TestAuxDataCodecBounds(unittest.TestCase):
    def test_valid_write_lands_in_the_right_slot(self):
        raw, kv_args = _buffers()
        payload = b"\xab" * ITEM_LEN
        AuxDataCodec.deserialize_data_to_buffer(kv_args, 1, 2, payload)
        base = 2 * ITEM_LEN
        self.assertEqual(bytes(raw[1][base : base + ITEM_LEN]), payload)

    def test_buffer_index_out_of_range_raises(self):
        _, kv_args = _buffers()
        for bad in (-1, N_BUFFERS, 99):
            with self.assertRaises(ValueError):
                AuxDataCodec.deserialize_data_to_buffer(
                    kv_args, bad, 0, b"\x01" * ITEM_LEN
                )

    def test_aux_index_out_of_range_raises(self):
        _, kv_args = _buffers()
        for bad in (-1, N_SLOTS, 100):
            with self.assertRaises(ValueError):
                AuxDataCodec.deserialize_data_to_buffer(
                    kv_args, 0, bad, b"\x01" * ITEM_LEN
                )

    def test_zero_geometry_raises(self):
        # Contributed by disagg-prefill (post-#42 fold): zero item/data lens
        # are degenerate geometry and must raise before address math.
        _, kv_args = _buffers()
        for zeros_at in ("item_len", "data_len", "both"):
            item_lens = [ITEM_LEN] * N_BUFFERS
            data_lens = [N_SLOTS * ITEM_LEN] * N_BUFFERS
            if zeros_at != "data_len":
                item_lens[0] = 0
            if zeros_at != "item_len":
                data_lens[0] = 0
            bad_args = SimpleNamespace(
                aux_data_ptrs=kv_args.aux_data_ptrs,
                aux_item_lens=item_lens,
                aux_data_lens=data_lens,
            )
            with self.assertRaises(ValueError, msg=zeros_at):
                AuxDataCodec.deserialize_data_to_buffer(bad_args, 0, 0, b"\x01" * ITEM_LEN)

    def test_length_mismatch_raises(self):
        _, kv_args = _buffers()
        for bad_len in (ITEM_LEN - 1, ITEM_LEN + 1, 0):
            with self.assertRaises(ValueError):
                AuxDataCodec.deserialize_data_to_buffer(
                    kv_args, 0, 0, b"\x01" * bad_len
                )

    def test_out_of_range_leaves_buffers_untouched(self):
        raw, kv_args = _buffers()
        with self.assertRaises(ValueError):
            AuxDataCodec.deserialize_data_to_buffer(
                kv_args, 0, N_SLOTS + 5, b"\xff" * ITEM_LEN
            )
        self.assertEqual(bytes(raw[0]), b"\x00" * (N_SLOTS * ITEM_LEN))


class TestHandleAuxDataRoomGate(unittest.TestCase):
    def test_live_room_written(self):
        raw, kv_args = _buffers()
        mgr = _fake_manager()
        mgr.kv_args = kv_args
        payload = b"\xcd" * ITEM_LEN
        mgr._handle_aux_data(_aux_msg(payload=payload))
        self.assertEqual(bytes(raw[0][:ITEM_LEN]), payload)

    def test_unknown_room_dropped(self):
        raw, kv_args = _buffers()
        mgr = _fake_manager()
        mgr.kv_args = kv_args
        mgr._handle_aux_data(_aux_msg(room=1234))
        self.assertEqual(bytes(raw[0]), b"\x00" * (N_SLOTS * ITEM_LEN))

    def test_concluded_room_dropped(self):
        # clear() pops request_status: a late aux must not hit a reused slot.
        raw, kv_args = _buffers()
        mgr = _fake_manager()
        mgr.kv_args = kv_args
        mgr.request_status = {}
        mgr._handle_aux_data(_aux_msg())
        self.assertEqual(bytes(raw[0]), b"\x00" * (N_SLOTS * ITEM_LEN))

    def test_out_of_bounds_write_refused_not_raised(self):
        # Bad wire coordinates must not kill the daemon decode thread.
        raw, kv_args = _buffers()
        mgr = _fake_manager()
        mgr.kv_args = kv_args
        mgr._handle_aux_data(_aux_msg(aux_index=N_SLOTS + 1))  # must not raise
        self.assertEqual(bytes(raw[0]), b"\x00" * (N_SLOTS * ITEM_LEN))

    def test_declared_length_mismatch_dropped(self):
        raw, kv_args = _buffers()
        mgr = _fake_manager()
        mgr.kv_args = kv_args
        msg = _aux_msg()
        msg[4] = struct.pack(">I", 999)
        mgr._handle_aux_data(msg)
        self.assertEqual(bytes(raw[0]), b"\x00" * (N_SLOTS * ITEM_LEN))


if __name__ == "__main__":
    unittest.main()
