# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""F-1 PoC: deferred mamba COW staged on an evictable source node
(unified FULL+MAMBA radix cache, match -> stage -> evict window).

The production window (schedule_policy admission -> batch
_collect_deferred_mamba_cow_and_clear -> model_runner
_execute_mamba_deferred_ops) is wired through GPU forward passes, but every
cache-side piece of it lives in the unified radix cache and runs on CPU:

1. R0 finishes with prefix P; insert() leaves tree node D owning mamba slot S
   (D.component_data[MAMBA].value = S). The FULL prefix can extend past D's
   mamba checkpoint: best_match_node (deepest mamba checkpoint) is then a
   proper ancestor of last_device_node (deepest FULL hit), so the admission
   lock taken below MatchResult.last_device_node does NOT pin D's mamba
   component.
2. R1 matches P with cow_mamba=True. finalize_match_result_in_cache
   (components/mamba.py) stages req.kv.mamba_cow_src_index = S: the deferred
   COW reads S at batch-forward time.
3. Inside the window another request's admittance needs a mamba slot:
   evict_for_alloc(mamba_num=1). The mamba eviction walk
   (MambaComponent._evict_device_next_node) picks D -- it sits in the mamba
   LRU with lock_ref == 0 -- and frees S; the allocator re-lets S to a new
   request whose forward then REWRITES S.
4. R1's deferred COW would now read slot S and copy ANOTHER REQUEST'S mamba
   state into R1's fresh slot -- cross-request mamba-state leakage (F-1).

This test drives that exact sequence through the REAL UnifiedRadixCache,
REAL MambaComponent eviction, the REAL mamba slot allocator and the REAL
mamba state buffers (fixture builder shared with
test_unified_radix_cache_unittest.py), and pins the post-fix contract:

    While a deferred mamba COW is staged against a tree node, the node's
    mamba slot MUST NOT be evicted, MUST NOT be re-let to another owner,
    and its state content MUST NOT change until the staged copy runs.

Fix: a stage-time lock on the source node taken by
unified_cache.mamba_cow.stage_mamba_cow_lock (all components, to keep the
full_lock >= ct_lock tree hierarchy at unified_tree_core.py:3006),
released via clear_mamba_cow on every exit path.

Pre-fix the starred assertions FAIL (source evicted / slot re-let / content
rewritten while staged); post-fix the whole suite passes. The pre-fix
failure output of test_evict_window_rewrites_staged_source_content is the
F-1 repro recorded in FEATURES_FINDINGS_swa.md.

    python -m pytest test/registered/unit/mem_cache/test_mamba_cow_evict_window_poc.py -v
"""

import os
import sys
import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# Reuse the real fixture builder (same dir): REAL UnifiedRadixCache with
# FULL+MAMBA TreeComponents, real mamba slot allocator and pool on CPU.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_unified_radix_cache_unittest as fixture_module  # noqa: E402
from test_unified_radix_cache_unittest import (  # noqa: E402
    CacheConfig,
    build_fixture,
)

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

# mamba_cache_chunk_size=8 with page_size=1: a 16-token prefix puts a mamba
# checkpoint at depth 8 while FULL reaches 16 -> the COW source (mamba
# best_match_node) is a proper ancestor of the FULL lock anchor
# (last_device_node), the production divergent-hit shape the window needs.
MAMBA_SLOTS = 4
CHUNK = 8

_CFG = CacheConfig(
    page_size=1,
    components=(ComponentType.FULL, ComponentType.MAMBA),
    mamba_cache_size=MAMBA_SLOTS,
)

PREFIX = list(range(1, 17))  # 16 tokens: FULL depth 16, mamba checkpoint at 8
OTHER_TOKENS = list(range(101, 109))


def _make_req():
    """Fresh request row for the admission-time COW staging path:
    holds_mamba False so finalize allocates the request's own dst slot,
    exactly as for a newly admitted request in production."""
    return SimpleNamespace(kv=ReqKvInfo(), session=None)


def _build_cpu(cfg):
    """build_fixture on a pure-CPU box: the only non-portable line in the
    builder is get_device() (platform detection needs an accelerator); the
    pools/caches it passes the device string to all run on 'cpu'."""
    with mock.patch.object(fixture_module, "get_device", lambda *a, **k: "cpu"):
        return build_fixture(cfg, mamba_cache_chunk_size=CHUNK)


class TestMambaCowEvictWindowPoC(CustomTestCase):
    def setUp(self):
        self.cache, self.allocator, self.req_to_token_pool = _build_cpu(_CFG)
        self.mamba_alloc = self.req_to_token_pool.mamba_allocator
        self.core = self.cache.tree_core

    # ---- tree helpers -------------------------------------------------

    def _insert_with_mamba(self, tokens):
        """insert() a prefix whose leaf tree node owns a mamba slot (what
        cache_finished_req donations produce in production); return
        (leaf_node, mamba_slot_id)."""
        value = self.allocator.alloc(len(tokens))
        self.assertIsNotNone(value)
        holder = self.mamba_alloc.alloc(1)
        self.assertIsNotNone(holder, "fixture mamba pool exhausted during setup")
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
        value_t = self.core.get_component_device_value(node, ComponentType.MAMBA)
        self.assertIsNotNone(value_t, "insert must leave a mamba value on the leaf")
        return node, int(value_t.item())

    def _stage_cow(self, tokens):
        """The admission-time match: stages the deferred COW from
        best_match_node onto a fresh request via the real
        finalize_match_result_in_cache path."""
        req = _make_req()
        match = self.cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", tokens)),
                cow_mamba=True,
                req=req,
            )
        )
        return req, match

    def _mamba_value(self, node_id):
        return self.core.get_component_device_value(node_id, ComponentType.MAMBA)

    def _node(self, node_id):
        return self.core.node_by_id(node_id)

    def _mamba_lock(self, node_id):
        return self._node(node_id).component_data[ComponentType.MAMBA].lock_ref

    def _mamba_state_row(self, slot_id):
        """One row of the REAL mamba state buffer for a slot (layer 0
        temporal state), via the same virtual->physical translation the
        deferred COW's copy uses."""
        temporal = self.req_to_token_pool.mamba_pool.mamba_cache.temporal
        phys = self.req_to_token_pool.translate_mamba_indices(
            torch.tensor([slot_id], dtype=torch.long)
        )
        return temporal[0][phys[0]]

    # ---- tests --------------------------------------------------------

    def test_staging_pins_the_source_node(self):
        """The window at its purest: a staged deferred COW leaves the
        source node's mamba component unlocked and therefore evictable
        pre-fix. Post-fix, staging itself pins the source (stage-time lock)
        for the whole deferred window. Production never
        covers this source with the admission lock: the admission lock
        anchors at MatchResult.last_device_node, which HiCache matches can
        leave at a proper ancestor of the mamba hit (FULL evicted-to-host
        below it), and a match takes no lock by itself at all."""
        node_src, slot = self._insert_with_mamba(PREFIX)
        req, match = self._stage_cow(PREFIX)
        # Staging happened against the mamba checkpoint node.
        self.assertIsNotNone(req.kv.mamba_cow_src_index)
        self.assertEqual(int(req.kv.mamba_cow_src_index.item()), slot)
        self.assertEqual(match.best_match_node, node_src)
        # The req holds no tree row yet (admission lock is taken later);
        # the only thing standing between the source slot and eviction is
        # the component lock staging must take.
        self.assertIsNone(req.kv.req_pool_idx)
        self.assertGreaterEqual(
            self._mamba_lock(node_src),
            1,
            "F-1: a deferred mamba COW is staged on a node whose mamba "
            "component is unlocked (lock_ref == 0) -- evictable while the "
            "forward-stream copy is still pending",
        )

    def test_evict_window_rewrites_staged_source_content(self):
        """Core PoC, content-tracked. Stage a deferred COW against D; mark
        D's state content (sentinel) in the real pool; then drive the
        production-window eviction pressure and simulate the new owner's
        forward write. Post-fix contract: none of {source value evicted,
        slot re-let, content changed} may happen while the COW is pending.
        Pre-fix all three happen: the pending copy would read the foreign
        (rewritten) state."""
        node_src_id, slot = self._insert_with_mamba(PREFIX)
        node_other_id, slot_other = self._insert_with_mamba(OTHER_TOKENS)
        # Hold the NODE OBJECT: pre-fix eviction can delete the node from
        # the arena (id invalid), which is itself the demonstrated bug.
        node_src = self._node(node_src_id)

        req, _ = self._stage_cow(PREFIX)
        staged_src = int(req.kv.mamba_cow_src_index.item())
        self.assertEqual(staged_src, slot)

        # Sentinel into the staged source content (a real deferred COW reads
        # exactly this slot at forward-stream time).
        self._mamba_state_row(staged_src).fill_(7.0)
        staged_content = self._mamba_state_row(staged_src).clone()

        # The staging match refreshed node_src in the mamba LRU, so a
        # same-instant eviction would pick node_other, not the bug victim.
        # Production interleaving: a concurrent match on the OTHER prefix
        # (no COW) pushes node_src back to the LRU tail between staging and
        # the batch-forward collect -- node_src becomes the eviction victim.
        self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", OTHER_TOKENS)))
        )

        # Window: another admittance drives one mamba eviction; the freed
        # slot is re-let and its new owner "runs" (writes its own state).
        self.cache.evict_for_alloc(EvictParams(num_tokens=0, mamba_num=1))
        re_let = self.mamba_alloc.alloc(1)
        re_let_id = int(re_let.item()) if re_let is not None else None

        # *** Post-fix contract, part 1: the staged source node survives
        # the window with its mamba value intact. Pre-fix the mamba LRU
        # victim IS node_src (pushed to the tail by the OTHER match), its
        # slot is freed and the leaf node removed from the arena.
        self.assertIsNotNone(
            node_src.component_data[ComponentType.MAMBA].value,
            "F-1: the cache evicted the node a deferred COW is staged "
            "from; its slot was freed while the forward-stream copy is "
            "still pending",
        )
        # *** Part 2: the staged slot id must not be re-let pool-wide.
        if re_let_id is not None:
            self.assertNotEqual(
                staged_src,
                re_let_id,
                "F-1: the staged COW source slot was re-let to another "
                "request while the deferred copy is still pending",
            )
        if re_let_id == staged_src:
            # Simulate the new owner's forward writing its state into the
            # re-let slot -- the normal, expected thing a fresh request does.
            self._mamba_state_row(re_let_id).fill_(999.0)
        # *** Part 3: strongest signal -- the content the pending COW would
        # copy must be identical to what was staged.
        self.assertTrue(
            torch.equal(self._mamba_state_row(staged_src), staged_content),
            "F-1 demonstrated: staged source content changed mid-window "
            "(slot re-let + rewritten); the deferred COW copies another "
            "request's mamba state -- cross-request leak",
        )

    def test_evict_window_after_clear_releases_source(self):
        """Companion: once the request's staged COW is cleared (all exit
        paths funnel through clear_mamba_cow post-fix), the source node is
        evictable again -- the fix must not leave a permanent pin in the
        tree. Pre-fix there is no clear API on the req; in that world this
        test documents the no-lock reality (eviction frees the staged
        source) and returns early."""
        try:
            from sglang.srt.mem_cache.unified_cache.mamba_cow import (
                clear_mamba_cow,
            )
        except ImportError:
            clear_mamba_cow = None

        node_src, slot = self._insert_with_mamba(PREFIX)
        node_other, slot_other = self._insert_with_mamba(OTHER_TOKENS)
        req, _ = self._stage_cow(PREFIX)

        if clear_mamba_cow is None:
            # Pre-fix behavior this PoC exists to document: with no source
            # lock, eviction inside the window frees the staged source slot.
            self.cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", OTHER_TOKENS)))
            )
            self.cache.evict_for_alloc(EvictParams(num_tokens=0, mamba_num=1))
            return

        # Post-fix: clearing the staged COW releases the source pin; a later
        # eviction may reclaim the source (it is no longer read), and must
        # not trip lock_ref assertions.
        clear_mamba_cow(req)
        self.assertIsNone(req.kv.mamba_cow_src_index)
        self.cache.evict_for_alloc(EvictParams(num_tokens=0, mamba_num=1))
        # Re-stage + clear repeatedly: no pin accumulation.
        for _ in range(2):
            req2, match2 = self._stage_cow(PREFIX)
            clear_mamba_cow(req2)
            self.assertEqual(
                self._mamba_lock(match2.best_match_node),
                0,
                "clear_mamba_cow leaked a pin on the source node "
                "(tree node frozen unevictable forever)",
            )


if __name__ == "__main__":
    unittest.main()
