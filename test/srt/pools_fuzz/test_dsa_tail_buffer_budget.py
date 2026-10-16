# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""DSA kpool-compress tail-buffer budget bias (board #47).

The tail buffers are a fixed per-(request, layer) allocation inside
DSATokenToKVPool that the KV budget previously never charged: up to ~8.3 GB
at V4-class dims (kpool 4 + 512 extra slots, head 128, 61 layers, 512 reqs).
The estimator must stay in exact sync with the pool's torch.zeros sizing.
"""
import unittest

import torch

from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

try:
    from sglang.test.ci.ci_register import register_cpu_ci

    register_cpu_ci(est_time=2, suite="base-a-test-cpu")
except Exception:
    pass


def _est(**kw):
    return DSATokenToKVPool.estimate_compress_tail_bytes(**kw)


class TestDSATailBufferBudget(unittest.TestCase):
    def test_exact_match_with_pool_allocation(self):
        # Mirror the pool's own allocation arithmetic element-for-element.
        reqs, kpool, extra, head, layers = 512, 4, 512, 128, 61
        expected = 0
        for _ in range(layers):
            for _ in range(2):  # tail_k and tail_score
                expected += torch.zeros(
                    reqs + 1, kpool + extra, head, dtype=torch.bfloat16
                ).numel() * torch._utils._element_size(torch.bfloat16)
        actual = _est(
            index_kpool=kpool,
            index_kpool_compress=True,
            tail_extra_slots=extra,
            index_head_dim=head,
            num_alloc_index_layers=layers,
            max_running_requests=reqs,
        )
        assert actual == expected
        assert actual == 513 * 516 * 128 * 2 * 2 * 61
        assert actual > 8_000_000_000  # the deployment-relevant scale

    def test_disabled_or_empty_is_zero(self):
        base = dict(
            index_kpool=4,
            index_kpool_compress=True,
            tail_extra_slots=512,
            index_head_dim=128,
            num_alloc_index_layers=61,
            max_running_requests=512,
        )
        # kpool == 1 => no compress ring
        assert _est(**{**base, "index_kpool": 1}) == 0
        # compress flag off
        assert (
            _est(**{**base, "index_kpool_compress": False}) == 0
        )
        # no allocating layers (draft worker's skip_topk)
        assert _est(**{**base, "num_alloc_index_layers": 0}) == 0
        # sparse max_running_requests unset (pool asserts later; charge 0)
        assert _est(**{**base, "max_running_requests": None}) == 0

    def test_sentinel_slot_rule(self):
        # req_pool_size = max_running_requests + 1 mirrors req_to_token's
        # sentinel-slot convention.
        a = _est(
            index_kpool=2,
            index_kpool_compress=True,
            tail_extra_slots=0,
            index_head_dim=64,
            num_alloc_index_layers=1,
            max_running_requests=7,
        )
        assert a == 8 * 2 * 64 * 2 * 2 * 1


class TestTailBiasChargeDirections(unittest.TestCase):
    """#47 review hole: the bias is computed BEFORE resolve_max_num_reqs, so a
    user-unset max_running_requests must charge the resolver's AUTO estimate
    bound, not zero; user-set must charge the dp-divided exact bound."""

    def _make_cfg(self):
        from types import SimpleNamespace

        import sglang.srt.model_executor.pool_configurator as pc

        layer_info = SimpleNamespace(num_effective_layers=60, start_layer=0, end_layer=60)
        mc = SimpleNamespace(
            hf_config=SimpleNamespace(indexer_types=None),
            context_len=32768,
        )
        kvc = SimpleNamespace(
            model_config=mc,
            layer_info=layer_info,
            attn_dp_size=2,
            mambaish_config=None,
            is_draft_worker=False,
            kv_cache_dtype_str=None,
        )
        cfg = pc.DefaultPoolConfigurator.__new__(pc.DefaultPoolConfigurator)
        cfg.kvc = kvc
        cfg._cell_size = 60 * 128 * 2
        return pc, cfg, kvc

    def _patch(self, pc, user_max):
        from contextlib import ExitStack
        from unittest import mock

        cm = ExitStack()
        cm.enter_context(mock.patch.object(pc, "is_deepseek_dsa", return_value=True))
        cm.enter_context(mock.patch.object(pc, "get_dsa_index_kpool", return_value=4))
        cm.enter_context(mock.patch.object(pc, "get_dsa_index_kpool_compress", return_value=True))
        cm.enter_context(mock.patch.object(pc, "get_dsa_index_head_dim", return_value=128))
        cm.enter_context(mock.patch.object(pc, "dsa_layer_skips_topk", return_value=False))
        cm.enter_context(mock.patch.object(pc, "max_speculative_num_draft_tokens", return_value=512))
        from types import SimpleNamespace

        cm.enter_context(
            mock.patch.object(
                pc,
                "get_schedule",
                return_value=SimpleNamespace(max_running_requests=user_max),
            )
        )
        return cm

    def test_user_set_charges_dp_divided_exact(self):
        pc, cfg, kvc = self._make_cfg()
        with self._patch(pc, 512):
            b = cfg._dsa_compress_tail_overhead_bytes(kvc, available_bytes=50 << 30)
        # min(512//2=256, ...) bound -> (256+1) rows
        assert b == 257 * 516 * 128 * 2 * 2 * 60

    def test_user_unset_charges_auto_estimate_upper_bound(self):
        pc, cfg, kvc = self._make_cfg()
        avail = 50 << 30
        with self._patch(pc, None):
            b = cfg._dsa_compress_tail_overhead_bytes(kvc, available_bytes=avail)
        per = 516 * 128 * 2 * 2 * 60
        rows = b // per
        assert b == rows * per, "charge must be an integral number of rows"
        # auto route big card: est caps at 4096, token-cap UB is 3.4M -> 4097 rows
        assert rows == 4097

    def test_user_unset_small_card_uses_est_not_4096(self):
        pc, cfg, kvc = self._make_cfg()
        avail = 2 << 30
        with self._patch(pc, None):
            b = cfg._dsa_compress_tail_overhead_bytes(kvc, available_bytes=avail)
        tok_ub = avail // cfg._cell_size
        est = max(min(int(tok_ub / 32768 * 512), 4096), 2048)
        expected_rows = min(est, tok_ub // 2) + 1
        assert b == expected_rows * (516 * 128 * 2 * 2 * 60)

    def test_not_dsa_charges_zero(self):
        from unittest import mock

        pc, cfg, kvc = self._make_cfg()
        with self._patch(pc, 512), mock.patch.object(pc, "is_deepseek_dsa", return_value=False):
            assert cfg._dsa_compress_tail_overhead_bytes(kvc, available_bytes=50 << 30) == 0


if __name__ == "__main__":
    unittest.main()
