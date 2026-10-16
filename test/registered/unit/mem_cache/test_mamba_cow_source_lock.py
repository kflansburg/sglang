"""REGRESSION TEST for F-1: unified mamba deferred COW source use-after-evict.

A prefix match can stage a deferred mamba copy-on-write: the source state
lives in a tree node's mamba slot until the next forward copies it into the
request's own slot (schedule_batch._collect_deferred_mamba_cow_and_clear).
Before the fix, nothing pinned the source node: ``evict_for_alloc`` between
staging and the collect could evict it and free the slot, and the deferred
read would then copy whatever mamba state the slot was re-let to —
cross-request mamba state leakage (F-1).

The fix stages a MAMBA-component-only lock on the source node at match/finalize
time (unified_cache.mamba_cow.stage_mamba_cow_lock), and every exit
path (post-collect, finish-release, retract reset, not-admitted revert, pool
free, H2D reload, lmcache flow clears) releases it via ``clear_mamba_cow``.

These tests run the REAL UnifiedRadixCache + MambaComponent machinery on CPU
(reusing the fixture builder from test_unified_radix_cache_unittest.py):
R0's prefix is inserted so a tree node owns a mamba slot; R1 matches it with
cow_mamba=True (staging the deferred COW and the source lock);
``evict_for_alloc(mamba_num=1)`` must NOT evict the pinned source while it
evicts unlocked nodes; ``clear_mamba_cow`` must restore evictability (no
leaked pin freezing the node forever).
"""

import os
import sys
import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.mamba_cow import (
    clear_mamba_cow,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# Reuse the real fixture builder (same dir), which constructs a REAL
# UnifiedRadixCache: real unified tree core, real FULL/MAMBA components, real
# mamba slot allocator on CPU.
sys.path.insert(0, os.path.dirname(__file__))
from test_unified_radix_cache_unittest import CacheConfig, build_fixture  # noqa: E402

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_CFG = CacheConfig(components=(ComponentType.FULL, ComponentType.MAMBA))

SOURCE_TOKENS = [1, 2, 3, 4]
OTHER_TOKENS = [7, 8, 9, 10]


def _make_req():
    """Minimal request for the COW staging/clear paths.

    holds_mamba is False (fresh request row) so finalize allocates a dst slot —
    the production staging path.
    """
    return SimpleNamespace(kv=ReqKvInfo())


class TestMambaCowSourceLock(CustomTestCase):
    def setUp(self):
        self.cache, self.allocator, self.req_to_token_pool = build_fixture(_CFG)

    def _insert_node_with_mamba(self, tokens):
        """Insert `tokens`; the leaf tree node ends up owning a mamba slot
        (mamba_value taken from a throwaway request's allocator slot)."""
        value = self.allocator.alloc(len(tokens))
        holder = self.req_to_token_pool.mamba_allocator.alloc(1)
        self.assertIsNotNone(holder)
        self.cache.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=value[: len(tokens)],
                mamba_value=holder[0].view(1),
            )
        )
        node = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        ).best_match_node
        self.assertIsNotNone(
            self.cache.tree_core.get_component_device_value(
                node, ComponentType.MAMBA
            )
        )
        return node

    def _stage_cow(self, tokens):
        """R1 matches with cow_mamba=True: finalize stages the deferred COW
        and (post-fix) pins the source node's MAMBA component."""
        req = _make_req()
        match = self.cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", tokens)),
                cow_mamba=True,
                req=req,
            )
        )
        return req, match

    def _mamba_lock(self, node):
        return self.cache.tree_core.get_component_device_lock_ref(
            node, ComponentType.MAMBA
        )

    def _full_lock(self, node):
        return self.cache.tree_core.get_component_device_lock_ref(
            node, ComponentType.FULL
        )

    def test_staging_pins_source_mamba_component_only(self):
        node_src = self._insert_node_with_mamba(SOURCE_TOKENS)
        req, match = self._stage_cow(SOURCE_TOKENS)

        # Deferred COW staged with pre-fix metadata...
        self.assertIsNotNone(req.kv.mamba_cow_src_index)
        # ...and post-fix the receipt pins the source's MAMBA component only.
        self.assertIsNotNone(req.kv.mamba_cow_lock)
        self.assertGreaterEqual(self._mamba_lock(match.best_match_node), 1)
        self.assertEqual(
            self._full_lock(match.best_match_node),
            0,
            "FULL component must not be pinned by the COW source lock",
        )

    def test_evict_for_alloc_spares_locked_source(self):
        node_src = self._insert_node_with_mamba(SOURCE_TOKENS)
        node_other = self._insert_node_with_mamba(OTHER_TOKENS)
        src_value = self.cache.tree_core.get_component_device_value(
            node_src, ComponentType.MAMBA
        )
        req, _ = self._stage_cow(SOURCE_TOKENS)
        self.assertIsNotNone(req.kv.mamba_cow_lock)

        # Pressure: evict one mamba slot (another request admittance).
        self.cache.evict_for_alloc(EvictParams(num_tokens=0, mamba_num=1))

        # The locked source node MUST remain; the unlocked node was the victim.
        self.assertIsNotNone(
            self.cache.tree_core.get_component_device_value(
                node_src, ComponentType.MAMBA
            ),
            "evict_for_alloc stole the locked COW source node (F-1 still open)",
        )
        self.assertIsNone(
            self.cache.tree_core.get_component_device_value(
                node_other, ComponentType.MAMBA
            ),
            "sanity: the unlocked node should have been the eviction victim",
        )

    def test_clear_releases_and_restores_evictability(self):
        node_src = self._insert_node_with_mamba(SOURCE_TOKENS)
        req, match = self._stage_cow(SOURCE_TOKENS)
        self.assertGreaterEqual(self._mamba_lock(match.best_match_node), 1)

        clear_mamba_cow(req)

        self.assertIsNone(req.kv.mamba_cow_lock)
        self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertFalse(req.kv.mamba_needs_clear)
        self.assertEqual(
            self._mamba_lock(match.best_match_node),
            0,
            "clear_mamba_cow leaked the pin: source node stays unevictable",
        )

    def test_clear_without_staging_is_noop(self):
        req = _make_req()
        clear_mamba_cow(req)  # must not raise
        self.assertIsNone(req.kv.mamba_cow_src_index)

    def test_restaging_supersedes_prior_lock(self):
        node_src = self._insert_node_with_mamba(SOURCE_TOKENS)
        req, match = self._stage_cow(SOURCE_TOKENS)
        first_lock = req.kv.mamba_cow_lock
        self.assertIsNotNone(first_lock)
        # A rematch before collect restages: the old receipt must be released
        # so no double pin accumulates on the node.
        from sglang.srt.mem_cache.unified_cache.mamba_cow import (
            stage_mamba_cow_lock,
        )

        stage_mamba_cow_lock(self.cache, match.best_match_node, req)
        self.assertIsNot(req.kv.mamba_cow_lock, first_lock)
        # Exactly one outstanding pin after restaging + one clear:
        clear_mamba_cow(req)
        self.assertEqual(
            self._mamba_lock(match.best_match_node),
            0,
            "restaging leaked a pin (double-acquire without matching release)",
        )


if __name__ == "__main__":
    unittest.main()
