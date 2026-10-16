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


if __name__ == "__main__":
    unittest.main()
