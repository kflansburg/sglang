# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Config-time int32 bound for DCP-widened virtual KV ids (allocplan seam fix).

req_to_token and the KV read-index buffers are int32; the widest virtual id
space is (max_total + page pad) * attn_dcp_size. A silent wrap bounds garbage
reads but lets the offload path copy one request's KV into another's slots.
The configurator must refuse such sizes loudly.
"""
import unittest

from sglang.srt.mem_cache.kv_cache_configurator import (
    _assert_dcp_widened_ids_fit_int32,
)

try:
    from sglang.test.ci.ci_register import register_cpu_ci

    register_cpu_ci(est_time=2, suite="base-a-test-cpu")
except Exception:
    pass


class TestDcpInt32Bound(unittest.TestCase):
    def test_normal_sizes_pass(self):
        # 100M tokens per rank, dcp 8, page 64: 0.8B < 2^31.
        _assert_dcp_widened_ids_fit_int32(
            max_total_num_tokens=100_000_000 + 64, attn_dcp_size=8, page_size=64
        )

    def test_dcp1_passes(self):
        _assert_dcp_widened_ids_fit_int32(
            max_total_num_tokens=2_000_000_000, attn_dcp_size=1, page_size=64
        )

    def test_exactly_at_limit_fails(self):
        with self.assertRaises(AssertionError):
            _assert_dcp_widened_ids_fit_int32(
                max_total_num_tokens=2**31 // 8 + 64 * 8
            , attn_dcp_size=8, page_size=64)
        # 2**31 itself must fail
        with self.assertRaises(AssertionError):
            _assert_dcp_widened_ids_fit_int32(
                max_total_num_tokens=2**31, attn_dcp_size=1, page_size=64
            )

    def test_page_pad_counts(self):
        # tokens*1 below the limit but tokens + pad must cross it
        with self.assertRaises(AssertionError):
            _assert_dcp_widened_ids_fit_int32(
                max_total_num_tokens=2**31 - 1, attn_dcp_size=2, page_size=64
            )

    def test_loc_space_scale_unwinds_draft_widening(self):
        # Draft sizes already carry the dcp factor via loc_space_scale: a
        # config that fits must NOT refail when loc_scale divides it back.
        # 512M per-request tokens * dcp 4 fails raw, passes with loc_scale=4.
        with self.assertRaises(AssertionError):
            _assert_dcp_widened_ids_fit_int32(
                max_total_num_tokens=2**31 - 1,
                attn_dcp_size=2,
                page_size=64,
                loc_space_scale=1,
            )
        _assert_dcp_widened_ids_fit_int32(
            max_total_num_tokens=2**31 - 1,
            attn_dcp_size=2,
            page_size=64,
            loc_space_scale=4,
        )
        # But the exact same division still fails when the result is ≥ 2^31.
        with self.assertRaises(AssertionError):
            _assert_dcp_widened_ids_fit_int32(
                max_total_num_tokens=2**32,
                attn_dcp_size=4,
                page_size=64,
                loc_space_scale=4,
            )


if __name__ == "__main__":
    unittest.main()
