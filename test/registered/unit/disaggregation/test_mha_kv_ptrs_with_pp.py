"""Unit tests for get_mha_kv_ptrs_with_pp pointer pairing (common/conn.py).

Regression for a silent-corruption bug: with a prefill running WITHOUT
speculative decoding and a decode running a draft model, the old pairing
selected branches via dst_total % num_kv_layers arithmetic. Whenever the peer's
draft block size made the modulo hit the "PP" else-branch (e.g. draft layers a
multiple of main layers), dst_v_ptrs resolved into the peer's draft_K buffers:
main V was written into draft K and decode's main V was never written, serving
stale KV pages of previous requests (cross-request leak), with matching list
lengths so no downstream check could catch it.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


import unittest
from types import SimpleNamespace

# NOTE: keep the import surface minimal — sglang.test.test_utils pulls the
# benchmark/datasets chain, which is not part of the CPU test env.
from sglang.srt.disaggregation.common.conn import CommonKVManager


def _mgr(num_draft_entries: int, start_layer: int = 0, end_layer=None, total_layers: int = 0):
    mgr = object.__new__(CommonKVManager)
    mgr.kv_args = SimpleNamespace(
        prefill_start_layer=start_layer,
        prefill_end_layer=end_layer,
        num_draft_entries=num_draft_entries,
        hidden_kv_layers=total_layers,
    )
    return mgr


def _ptrs(M: int, D: int, tag: str):
    """A pointer list laid out [K_main M, V_main M, dK D, dV D]."""
    return (
        [f"{tag}K{i}" for i in range(M)]
        + [f"{tag}V{i}" for i in range(M)]
        + [f"{tag}dK{i}" for i in range(D)]
        + [f"{tag}dV{i}" for i in range(D)]
    )


class TestGetMhaKvPtrsWithPP(unittest.TestCase):
    def test_prefill_no_draft_decode_draft_multiple_of_main(self):
        # THE bug shape: M=4 main, D=4 draft on decode only.
        src = _ptrs(4, 0, "s")
        dst = _ptrs(4, 4, "t")
        sk, sv, dk, dv, layers = _mgr(0, total_layers=4).get_mha_kv_ptrs_with_pp(src, dst)
        self.assertEqual(sk, [f"sK{i}" for i in range(4)])
        self.assertEqual(sv, [f"sV{i}" for i in range(4)])
        self.assertEqual(dk, [f"tK{i}" for i in range(4)])
        # The previous code returned t dK0..dK3 here.
        self.assertEqual(dv, [f"tV{i}" for i in range(4)])
        self.assertEqual(layers, 4)

    def test_prefill_no_draft_decode_small_draft(self):
        src = _ptrs(6, 0, "s")
        dst = _ptrs(6, 2, "t")
        sk, sv, dk, dv, layers = _mgr(0, total_layers=6).get_mha_kv_ptrs_with_pp(src, dst)
        self.assertEqual(dk, [f"tK{i}" for i in range(6)])
        self.assertEqual(dv, [f"tV{i}" for i in range(6)])
        self.assertEqual(sv, [f"sV{i}" for i in range(6)])

    def test_both_no_draft_single_stage(self):
        src = _ptrs(8, 0, "s")
        dst = _ptrs(8, 0, "t")
        sk, sv, dk, dv, layers = _mgr(0).get_mha_kv_ptrs_with_pp(src, dst)
        self.assertEqual(dk, [f"tK{i}" for i in range(8)])
        self.assertEqual(dv, [f"tV{i}" for i in range(8)])
        self.assertEqual(layers, 8)

    def test_both_with_draft_pairs_main_blocks(self):
        # Symmetric speculative deployment: each side appends 2*D draft
        # entries (K+V per draft layer); only main blocks are paired here.
        src = _ptrs(4, 1, "s")  # 2 draft entries
        dst = _ptrs(4, 1, "t")
        sk, sv, dk, dv, layers = _mgr(2, total_layers=4).get_mha_kv_ptrs_with_pp(src, dst)
        self.assertEqual(sk, [f"sK{i}" for i in range(4)])
        self.assertEqual(sv, [f"sV{i}" for i in range(4)])
        self.assertEqual(dk, [f"tK{i}" for i in range(4)])
        self.assertEqual(dv, [f"tV{i}" for i in range(4)])

    def test_pp_stage_no_draft(self):
        # Prefill stage covers main layers [4, 8) of a 12-layer model (src ptr
        # list holds only this stage's 4 layers); decode registered the full
        # model (12 layers).
        src = _ptrs(4, 0, "s")
        dst = _ptrs(12, 0, "t")
        sk, sv, dk, dv, layers = _mgr(
            0, start_layer=4, end_layer=8, total_layers=12
        ).get_mha_kv_ptrs_with_pp(src, dst)
        self.assertEqual(dk, [f"tK{i}" for i in range(4, 8)])
        self.assertEqual(dv, [f"tV{i}" for i in range(4, 8)])
        self.assertEqual(layers, 4)

    def test_pp_stage_with_peer_draft_pairs_main_blocks(self):
        # Asymmetric draft with pp>1: the full-model layer count (set on
        # kv_args.hidden_kv_layers by prefill/decode setup) makes the peer's
        # main block derivable; pairing must skip the draft tail.
        src = _ptrs(4, 0, "s")
        dst = _ptrs(12, 2, "t")
        sk, sv, dk, dv, layers = _mgr(
            0, start_layer=4, end_layer=8, total_layers=12
        ).get_mha_kv_ptrs_with_pp(src, dst)
        self.assertEqual(dk, [f"tK{i}" for i in range(4, 8)])
        self.assertEqual(dv, [f"tV{i}" for i in range(4, 8)])

    def test_peer_with_hidden_draft_without_layer_count_falls_back_to_stage(self):
        # Setup paths that never populated hidden_kv_layers keep the old
        # stage-count assumption (correct for pp==1 no-draft).
        src = _ptrs(4, 0, "s")
        dst = _ptrs(4, 0, "t")
        sk, sv, dk, dv, layers = _mgr(0, total_layers=0).get_mha_kv_ptrs_with_pp(src, dst)
        self.assertEqual(dv, [f"tV{i}" for i in range(4)])

    def test_geometry_mismatch_never_truncates(self):
        src = _ptrs(4, 0, "s")
        dst = _ptrs(3, 0, "t")  # odd main layer count vs stage -> mismatch
        with self.assertRaises(RuntimeError):
            _mgr(0).get_mha_kv_ptrs_with_pp(src, dst)


if __name__ == "__main__":
    unittest.main()
