"""Unit tests for AuxDataCodec wire-index bounds (common/utils.py).

Regression for a cross-request metadata corruption: the aux slot index comes
off the wire from the peer; a stale/duplicated metadata message (e.g. after
metadata-buffer slot reuse) or a malformed peer could write past the
registered aux buffer into a slot owned by another request. The codec must
raise instead of writing out of bounds.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


import ctypes
import unittest
from types import SimpleNamespace

# NOTE: keep the import surface minimal — sglang.test.test_utils pulls the
# benchmark/datasets chain, which is not part of the CPU test env.
from sglang.srt.disaggregation.common.utils import AuxDataCodec


def _kv_args(rows: int = 4, item_len: int = 64):
    buf = (ctypes.c_byte * (rows * item_len))()
    return SimpleNamespace(
        aux_data_ptrs=[ctypes.addressof(buf)],
        aux_data_lens=[rows * item_len],
        aux_item_lens=[item_len],
    ), buf


class TestAuxDataCodecBounds(unittest.TestCase):
    def test_in_range_write_lands_in_the_right_row(self):
        kv_args, buf = _kv_args(rows=4, item_len=8)
        AuxDataCodec.deserialize_data_to_buffer(kv_args, 0, 2, b"ABCDEFGH")
        self.assertEqual(bytes(buf[16:24]), b"ABCDEFGH")

    def test_negative_index_raises(self):
        kv_args, _ = _kv_args()
        with self.assertRaises(ValueError):
            AuxDataCodec.deserialize_data_to_buffer(kv_args, 0, -1, b"xx")

    def test_index_past_last_row_raises(self):
        kv_args, _ = _kv_args(rows=4, item_len=64)
        with self.assertRaises(ValueError):
            AuxDataCodec.deserialize_data_to_buffer(kv_args, 0, 4, b"xx")

    def test_payload_longer_than_row_raises(self):
        kv_args, _ = _kv_args(rows=4, item_len=8)
        with self.assertRaises(ValueError):
            AuxDataCodec.deserialize_data_to_buffer(kv_args, 0, 0, b"9 bytes!!")

    def test_zero_geometry_raises(self):
        kv_args = SimpleNamespace(
            aux_data_ptrs=[0], aux_data_lens=[0], aux_item_lens=[0]
        )
        with self.assertRaises(ValueError):
            AuxDataCodec.deserialize_data_to_buffer(kv_args, 0, 0, b"xx")


if __name__ == "__main__":
    unittest.main()
