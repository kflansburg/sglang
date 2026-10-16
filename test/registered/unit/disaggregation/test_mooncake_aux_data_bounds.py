"""F12 regression tests: AUX_DATA decode-side bounds validation.

The AUX_DATA wire message (mooncake backend, TCP aux path) carries its target
coordinates: room, buffer_index, aux_index, payload. Before the fix, all four
were trusted:

  * a concluded/unknown room raced a metadata slot reissued to ANOTHER
    request -> cross-request metadata overwrite;
  * buffer_index / aux_index out of range -> the ctypes from_address write in
    AuxDataCodec lands outside the buffer (arbitrary-offset write);
  * a payload longer than the item slot overwrites the following slot.

The fix validates all four before writing; anything off is dropped and logged.
"""
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

ROOM = 77


def _aux_msg(room=ROOM, buffer_index=0, aux_index=0, payload=b"\x01\x02\x03\x04"):
    return [
        MooncakeKVManager.AUX_DATA_HEADER,
        str(room).encode("ascii"),
        str(buffer_index).encode("ascii"),
        str(aux_index).encode("ascii"),
        struct.pack(">I", len(payload)),
        payload,
    ]


def _fake_manager(item_len=8, capacity=2, buffers=2):
    mgr = MooncakeKVManager.__new__(MooncakeKVManager)
    mgr.request_status = {ROOM: object()}
    mgr.kv_args = SimpleNamespace(
        aux_data_ptrs=[1000 + i for i in range(buffers)],
        aux_data_lens=[item_len * capacity] * buffers,
        aux_item_lens=[item_len] * buffers,
    )
    return mgr


class TestAuxDataBounds(unittest.TestCase):
    def _run(self, mgr, msg):
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.AuxDataCodec"
        ) as codec:
            mgr._handle_aux_data(msg)
        return codec

    def test_valid_message_written(self):
        mgr = _fake_manager()
        codec = self._run(mgr, _aux_msg())
        codec.deserialize_data_to_buffer.assert_called_once_with(
            mgr.kv_args, 0, 0, b"\x01\x02\x03\x04"
        )

    def test_unknown_room_dropped(self):
        mgr = _fake_manager()
        codec = self._run(mgr, _aux_msg(room=1234))
        codec.deserialize_data_to_buffer.assert_not_called()

    def test_concluded_room_dropped(self):
        # clear() pops request_status: a late aux must not hit a reused slot.
        mgr = _fake_manager()
        mgr.request_status = {}
        codec = self._run(mgr, _aux_msg())
        codec.deserialize_data_to_buffer.assert_not_called()

    def test_buffer_index_out_of_range_dropped(self):
        mgr = _fake_manager()
        for bad in (-1, 2, 10):
            codec = self._run(mgr, _aux_msg(buffer_index=bad))
            codec.deserialize_data_to_buffer.assert_not_called()

    def test_aux_index_out_of_range_dropped(self):
        # capacity=2 items -> aux_index 0 and 1 legal only.
        mgr = _fake_manager()
        for bad in (-1, 2, 100):
            codec = self._run(mgr, _aux_msg(aux_index=bad))
            codec.deserialize_data_to_buffer.assert_not_called()

    def test_aux_index_at_last_slot_written(self):
        mgr = _fake_manager()
        codec = self._run(mgr, _aux_msg(aux_index=1))
        codec.deserialize_data_to_buffer.assert_called_once()

    def test_oversized_payload_dropped(self):
        mgr = _fake_manager(item_len=8)
        codec = self._run(mgr, _aux_msg(payload=b"\x00" * 9))
        codec.deserialize_data_to_buffer.assert_not_called()

    def test_length_mismatch_dropped(self):
        mgr = _fake_manager()
        msg = _aux_msg()
        msg[4] = struct.pack(">I", 999)  # declared != actual
        codec = self._run(mgr, msg)
        codec.deserialize_data_to_buffer.assert_not_called()

    def test_zero_item_len_dropped(self):
        mgr = _fake_manager(item_len=0, capacity=0)
        codec = self._run(mgr, _aux_msg())
        codec.deserialize_data_to_buffer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
