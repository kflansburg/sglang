"""Unit tests for the CLEF radix-cache sidecar store: budget accounting,
split alignment, eviction safety, identity-mismatch fallback (PR 3)."""

import unittest
from array import array

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

IDENTITY = "model@c1+head@h2+bf16"


def _cache(page_size=1):
    return RadixCache.create_simulated(page_size=page_size)


def _insert(cache, ids):
    res = cache.insert(InsertParams(key=RadixKey(token_ids=array("q", ids))))
    return res.last_device_node


class TestRadixSidecar(CustomTestCase):
    def test_budget_accounting_attach_and_drop(self):
        cache = _cache()
        node = _insert(cache, [1, 2, 3, 4])
        sidecar = torch.randn(4, 8)  # 4 rows x 8 cols fp32 = 128 bytes
        nbytes = sidecar.numel() * sidecar.element_size()
        self.assertEqual(cache.sidecar_bytes(), 0)
        cache.attach_sidecar(node, sidecar, IDENTITY)
        self.assertEqual(cache.sidecar_bytes(), nbytes)
        self.assertIs(node.sidecar, sidecar)
        self.assertEqual(node.sidecar_identity, IDENTITY)
        cache.drop_sidecar(node)
        self.assertEqual(cache.sidecar_bytes(), 0)
        self.assertIsNone(node.sidecar)
        self.assertIsNone(node.sidecar_identity)

    def test_get_sidecar_identity_mismatch_returns_none(self):
        cache = _cache()
        node = _insert(cache, [5, 6, 7])
        cache.attach_sidecar(node, torch.randn(3, 4), IDENTITY)
        self.assertIsNotNone(cache.get_sidecar(node, IDENTITY))
        # A head/weights/dtype change invalidates the sidecar even though the
        # backbone KV is still present -> None -> caller falls back to recompute.
        self.assertIsNone(cache.get_sidecar(node, "head@h9+fp8"))
        # A KV-only hit (sidecar dropped/absent) also yields None.
        cache.drop_sidecar(node)
        self.assertIsNone(cache.get_sidecar(node, IDENTITY))

    def test_split_aligns_sidecar_rows_with_tokens(self):
        cache = _cache()
        node = _insert(cache, [1, 2, 3, 4, 5, 6])
        rows = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
        cache.attach_sidecar(node, rows.clone(), IDENTITY)
        child = node
        # Split after 3 tokens: new_node = prefix [1,2,3], child = suffix [4,5,6].
        new_node = cache._split_node(child.key, child, split_len=3)
        self.assertEqual(new_node.sidecar.shape, (3, 4))
        self.assertEqual(child.sidecar.shape, (3, 4))
        torch.testing.assert_close(new_node.sidecar, rows[:3])
        torch.testing.assert_close(child.sidecar, rows[3:])
        self.assertEqual(new_node.sidecar_identity, IDENTITY)

    def test_eviction_unaccounts_sidecar_bytes(self):
        cache = _cache()
        # Two disjoint branches so one is an evictable leaf.
        nA = _insert(cache, [1, 2, 3, 4])
        _insert(cache, [9, 8, 7, 6])
        bytes_before = cache.sidecar_bytes()
        sidecar = torch.randn(4, 8)
        nbytes = sidecar.numel() * sidecar.element_size()
        cache.attach_sidecar(nA, sidecar, IDENTITY)
        self.assertEqual(cache.sidecar_bytes(), bytes_before + nbytes)
        # Evict more than the cache holds; the leaf carrying the sidecar is freed.
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        cache.evict(EvictParams(num_tokens=64))
        self.assertEqual(cache.sidecar_bytes(), 0)


if __name__ == "__main__":
    unittest.main()
