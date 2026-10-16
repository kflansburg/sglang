"""Regression tests for the paged SWA/DSA state index drift cap (F6).

maybe_send_extra used to silently truncate on ANY length drift between the
prefill and decode state page lists: unmatched pages were never written yet
the transfer still concluded Success, so attention ran over stale state rows
left by previous owners (cross-request context leak). Only exact-match types
(QSA/SWA_RING/DSV4_REQUEST_STATE) raised. Now paged types keep the documented
1-page accounting window and raise on drift > 1 page.
"""

import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, TransferInfo
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_manager(state_type):
    manager = object.__new__(MooncakeKVManager)
    manager.kv_args = SimpleNamespace(
        state_types=[state_type],
        state_data_ptrs=[[0x3000]],
        state_item_lens=[[16]],
        state_dim_per_tensor=[[]],
        state_layer_ids=[[]],
    )
    manager.pp_size = 1
    manager.attn_tp_size = manager.attn_cp_size = 1
    manager.state_tp_size = 1
    manager.attn_tp_rank = manager.attn_cp_rank = 0
    manager.is_mla_backend = True
    manager.is_hybrid_mla_backend = False
    manager.state_strides_validated = {"decode:x"}
    manager.request_status = {1: KVPoll.Transferring}
    manager.conclude_failure = Mock()
    manager.enable_custom_mem_pool = False
    manager.custom_mem_pool_type = None
    manager.max_transfer_batch_indices = 0
    manager._transfer_data = Mock(return_value=0)
    return manager


def _make_req(state_type):
    return TransferInfo(
        room=1,
        endpoint="127.0.0.1",
        dst_port=1234,
        mooncake_session_id="decode:x",
        dst_kv_indices=None,
        dst_aux_index=0,
        dst_state_indices=[[5, 6, 7]],
        required_dst_info_num=1,
        is_dummy=False,
    )


DST_KV_ARGS = SimpleNamespace(
    dst_state_data_ptrs=[[0x4000]],
    dst_state_item_lens=[[16]],
    dst_state_dim_per_tensor=[[]],
    dst_state_layer_ids=[[]],
    dst_attn_tp_size=1,
    dst_tp_rank=0,
)


class TestMooncakeStateDrift(CustomTestCase):
    def test_one_page_drift_is_tolerated_for_paged_swa(self):
        manager = _make_manager(StateType.SWA)
        req = _make_req(StateType.SWA)
        # prefill has 4 pages, decode 3: exactly the 1-page window.
        with get_context().override_server_args(enable_unified_memory=False):
            rc = manager.maybe_send_extra(req, [[2, 3, 4, 5]], None, DST_KV_ARGS)
        self.assertEqual(rc, 0)
        # Truncated to the decode list: three consecutive pages coalesce into
        # one block, aligned pairwise (prefill [2,3,4] -> decode [5,6,7]).
        blocks = manager._transfer_data.call_args.args[1]
        self.assertEqual(blocks, [(0x3000 + 2 * 16, 0x4000 + 5 * 16, 3 * 16)])

    def test_two_page_drift_raises_for_paged_swa(self):
        manager = _make_manager(StateType.SWA)
        req = _make_req(StateType.SWA)
        # prefill 5 vs decode 3: outside the window -> fail loud.
        with self.assertRaisesRegex(RuntimeError, "drift exceeds the 1-page"):
            manager.maybe_send_extra(
                req, [[2, 3, 4, 5, 6]], None, DST_KV_ARGS
            )

    def test_two_page_drift_raises_for_paged_dsa(self):
        manager = _make_manager(StateType.DSA)
        req = _make_req(StateType.DSA)
        req.dst_state_indices = [[5, 6, 7, 8]]
        with self.assertRaisesRegex(RuntimeError, "drift exceeds the 1-page"):
            manager.maybe_send_extra(req, [[2, 3]], None, DST_KV_ARGS)

    def test_exact_match_types_still_raise_on_any_drift(self):
        manager = _make_manager(StateType.SWA_RING)
        req = _make_req(StateType.SWA_RING)
        with self.assertRaisesRegex(RuntimeError, "state index length mismatch"):
            manager.maybe_send_extra(req, [[2, 3, 4, 5]], None, DST_KV_ARGS)


if __name__ == "__main__":
    unittest.main()
