"""Regression for board #59: UnifiedRadixCache.checkpoint(up_to smaller than
cache_protected_len by >= 1 page) must not shrink cache_protected_len.

Trigger shape: decode-time checkpoints advanced protection into generated
output; a finished request trimmed below it (strip_thinking_cache reports
owned_kv_len == prompt, or a long stop tail) re-checkpoints at the trimmed
length. Pre-fix: multi-page shrink trips assert cache_protected_len <=
len(new_indices) + page_size - 1 (scheduler crash) or, under python -O,
silently releases tree-owned rows to the allocator (the #38 mechanism in the
unified cache). Post-fix: checkpoint skips the no-op/shrinking insert,
components see "no insert happened", protection is monotone.
"""
import unittest
import unittest.mock
from array import array

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

PAGE = 4


def make_cache():
    kv_pool = MHATokenToKVPool(
        size=64,
        page_size=PAGE,
        dtype=torch.float16,
        head_num=2,
        head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = PagedTokenToKVPoolAllocator(
        size=64,
        page_size=PAGE,
        dtype=torch.float16,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    req_pool = ReqToTokenPool(
        size=8,
        max_context_len=128,
        device="cpu",
        enable_memory_saver=False,
    )
    return UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=PAGE,
            eviction_policy="lru",
            tree_components=(ComponentType.FULL,),
            # Deterministic across CPU sandboxes: auto-selecting "rust" would
            # invoke a cargo build of sglang-radix-tree when the toolchain is
            # absent, failing the merge queue on an environment detail.
            tree_core_backend="python",
        )
    )


def make_req(cache, token_ids):
    req_idx = 1
    kv_indices = cache.token_to_kv_pool_allocator.alloc(len(token_ids))
    assert kv_indices is not None
    cache.req_to_token_pool.req_to_token[req_idx, : len(token_ids)] = kv_indices
    req = unittest.mock.Mock(
        kv=unittest.mock.Mock(
            req_pool_idx=req_idx,
            cache_protected_len=0,
            cache_inserted_len=0,
            swa_evict_floor=0,
            mamba_pool_idx=None,
            mamba_next_track_idx=None,
            session_id=None,
        ),
        extra_key=None,
        cache_salt=None,
        priority=0,
        session_id=None,
        kv_rotation_base=None,
        host_hit_length=0,
        lock=None,
        last_node=None,
    )
    req.full_untruncated_fill_ids = array("q", token_ids)
    req.origin_input_ids = array("q", token_ids[:PAGE])
    req.output_ids = array("q", token_ids[PAGE:])
    req.finished.return_value = False
    req.skip_radix_cache_insert = False
    return req


class TestUnifiedCheckpointMonotonic(unittest.TestCase):
    def test_checkpoint_never_shrinks_cache_protected_len(self):
        cache = make_cache()
        token_ids = list(range(1, 13))  # 3 pages
        req = make_req(cache, token_ids)

        # Decode-time checkpoint publishing two pages.
        cache.checkpoint(req, up_to=2 * PAGE)
        self.assertEqual(req.kv.cache_protected_len, 2 * PAGE)
        tree_pages = {
            v.item() // PAGE
            for v in cache.path_device_indices(req.last_node)
        }
        free_before = set(
            cache.token_to_kv_pool_allocator.get_all_free_pages().tolist()
        )
        self.assertFalse(tree_pages & free_before)

        # Trimmed finish re-checkpoints one page lower.
        cache.checkpoint(req, up_to=PAGE)
        self.assertEqual(
            req.kv.cache_protected_len,
            2 * PAGE,
            "unified checkpoint must not shrink cache_protected_len",
        )

        # No tree-referenced page may return to the allocator...
        free_after = set(
            cache.token_to_kv_pool_allocator.get_all_free_pages().tolist()
        )
        self.assertFalse(free_after & tree_pages, (free_after, tree_pages))

        # ...while the tree still serves the full protected prefix.
        m = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", token_ids[: 2 * PAGE])))
        )
        self.assertEqual(m.device_prefix_len, 2 * PAGE)

        # A later allocation must not collide with the tree-owned pages.
        nxt = cache.token_to_kv_pool_allocator.alloc(PAGE)
        if nxt is not None:
            self.assertFalse(nxt[0].item() // PAGE in tree_pages)


if __name__ == "__main__":
    unittest.main()
