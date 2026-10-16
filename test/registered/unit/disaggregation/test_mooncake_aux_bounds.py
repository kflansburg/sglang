"""Regression tests for AUX_DATA bounds validation (F12).

The aux codec writes to ``aux_ptr + item_len * aux_index`` with all three
values wire-controlled; before, any out-of-bounds index became a raw-pointer
write into neighboring metadata (nothing checked room liveness, buffer index,
aux index, or payload length). Now the codec raises and the mooncake handler
drops messages for unknown/concluded rooms or invalid geometry instead of
writing.
"""

import struct
import unittest
from ctypes import c_byte
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _kv_args(total_slots=8, item_len=16):
    backing = (c_byte * 4096)()
    return SimpleNamespace(
        aux_data_ptrs=[backing_ptr(backing)],
        aux_data_lens=[total_slots * item_len],
        aux_item_lens=[item_len],
    ), backing


def backing_ptr(backing):
    from ctypes import addressof

    return addressof(backing)


class TestAuxDataBounds(CustomTestCase):
    def _manager(self, live_room=7):
        manager = object.__new__(MooncakeKVManager)
        # Keep the ctypes backing buffer alive for the manager's lifetime;
        # its address is embedded in aux_data_ptrs.
        kv_args, manager._backing = _kv_args()
        manager.kv_args = kv_args
        manager.request_status = {live_room: KVPoll.Transferring}
        return manager

    def _msg(self, room, buffer_index, aux_index, data):
        return [
            MooncakeKVManager.AUX_DATA_HEADER,
            str(room).encode("ascii"),
            str(buffer_index).encode("ascii"),
            str(aux_index).encode("ascii"),
            struct.pack(">I", len(data)),
            data,
        ]

    def test_valid_aux_data_is_written(self):
        manager = self._manager(live_room=7)
        data = bytes(range(16))
        manager._handle_aux_data(self._msg(7, 0, 3, data))
        # slot 3 of buffer 0 holds the data (addr check via ctypes read).
        from sglang.srt.disaggregation.common.utils import AuxDataCodec

        self.assertEqual(
            AuxDataCodec.serialize_data_from_buffer(
                manager.kv_args.aux_data_ptrs[0] + 3 * 16, 16
            ),
            data,
        )

    def test_unknown_room_is_dropped(self):
        manager = self._manager(live_room=7)
        manager._handle_aux_data(self._msg(999, 0, 3, bytes(16)))
        from sglang.srt.disaggregation.common.utils import AuxDataCodec

        self.assertEqual(
            AuxDataCodec.serialize_data_from_buffer(
                manager.kv_args.aux_data_ptrs[0] + 3 * 16, 16
            ),
            bytes(16),
        )

    def test_concluded_room_is_dropped(self):
        manager = self._manager(live_room=7)
        manager.request_status = {}  # room concluded and cleared
        manager._handle_aux_data(self._msg(7, 0, 3, bytes(16)))
        from sglang.srt.disaggregation.common.utils import AuxDataCodec

        self.assertEqual(
            AuxDataCodec.serialize_data_from_buffer(
                manager.kv_args.aux_data_ptrs[0] + 3 * 16, 16
            ),
            bytes(16),
        )

    def test_out_of_range_aux_index_is_not_written(self):
        manager = self._manager(live_room=7)
        # 8 slots; index 8 would write one slot past the registered region.
        manager._handle_aux_data(self._msg(7, 0, 8, b"\xff" * 16))
        manager._handle_aux_data(self._msg(7, 0, -1, b"\xff" * 16))
        from sglang.srt.disaggregation.common.utils import AuxDataCodec

        self.assertEqual(
            AuxDataCodec.serialize_data_from_buffer(
                manager.kv_args.aux_data_ptrs[0], 8 * 16
            ),
            bytes(8 * 16),
        )

    def test_foreign_buffer_index_is_not_written(self):
        manager = self._manager(live_room=7)
        manager._handle_aux_data(self._msg(7, 1, 0, b"\xff" * 16))
        from sglang.srt.disaggregation.common.utils import AuxDataCodec

        self.assertEqual(
            AuxDataCodec.serialize_data_from_buffer(
                manager.kv_args.aux_data_ptrs[0], 8 * 16
            ),
            bytes(8 * 16),
        )

    def test_wrong_payload_length_is_not_written(self):
        manager = self._manager(live_room=7)
        manager._handle_aux_data(self._msg(7, 0, 3, bytes(32)))
        from sglang.srt.disaggregation.common.utils import AuxDataCodec

        self.assertEqual(
            AuxDataCodec.serialize_data_from_buffer(
                manager.kv_args.aux_data_ptrs[0] + 3 * 16, 16
            ),
            bytes(16),
        )

    # Contributed by disagg-prefill (#42 review): codec-level edge cases.
    def test_zero_geometry_raises(self):
        from sglang.srt.disaggregation.common.utils import AuxDataCodec

        kv_args = SimpleNamespace(
            aux_data_ptrs=[0], aux_data_lens=[0], aux_item_lens=[0]
        )
        with self.assertRaises(ValueError):
            AuxDataCodec.deserialize_data_to_buffer(kv_args, 0, 0, b"xx")

    def test_in_range_write_lands_in_the_right_row(self):
        from ctypes import c_byte, addressof

        from sglang.srt.disaggregation.common.utils import AuxDataCodec

        buf = (c_byte * 32)()  # 4 rows x 8 bytes
        kv_args = SimpleNamespace(
            aux_data_ptrs=[addressof(buf)],
            aux_data_lens=[32],
            aux_item_lens=[8],
        )
        AuxDataCodec.deserialize_data_to_buffer(kv_args, 0, 2, b"ABCDEFGH")
        self.assertEqual(bytes(buf[16:24]), b"ABCDEFGH")
        self.assertEqual(bytes(buf[:16]), bytes(16))
        self.assertEqual(bytes(buf[24:]), bytes(8))


if __name__ == "__main__":
    unittest.main()
