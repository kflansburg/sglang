"""Regression tests: Q3 pad-guard in alloc_device_buffer (F-5 sibling).

dsa Q3 trigger: in alloc_device_buffer's else branch, the "page alignment"
pad fabricated continuation ids with arange(hisparse[-1] + 1, ...). In the
fresh-tail-empty shape (pure radix-prefix reuse), hisparse[-1] is a
TREE-OWNED prefix value, so the fabricated ids live inside the tree's page;
the buffer claims them and the request's later whole-page
(free -> unique(//page)) release hands the tree's page back to the device free
list while the tree still maps it (UAF / cross-request C4 corruption).

Fix: fabricate continuation ids ONLY from a FRESH-owned kept tail. When the
kept run's tail is a prefix/shared value (fresh_vals empty), fall back to a
whole-page alloc and take its front ids; the allocated page's leftover ids
stay request-owned and are reclaimed via whole-page dedup at free.

Guard semantics pinned here (both classes):
(1) fresh-tail-empty + unaligned kept length: pad does NOT use the tree page's
    range; it comes from a freshly allocated page, the tree page returns at
    MOST once to the free list (never while tree-owned), accounting stays sane.
(2) fresh tail present: dense fast pad unchanged (continuation of own page).
(3) pre-fix shape pin: assert the pre-fix path fabricates within-tree ids —
    run this file again pre-fix, these two guards fail (recorded shape);

Run: PYTHONPATH=python python3 -m pytest test/srt/pools_fuzz/test_hisparse_q3_pad_guard.py -q
"""

import pytest
import torch

from sglang.srt.mem_cache.allocator.hisparse import (
    HiSparseTokenToKVPoolAllocator,
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
)

PAGE = 64


# --- fixtures (same pattern as test_hisparse_prefix_guard.py) ---


class FakeKV:
    def __init__(self):
        self.m = None

    def register_mapping(self, m):
        self.m = m

    def _translate_loc_to_hisparse_device(self, idx):
        return self.m[idx.to(torch.int64)]

    def get_cpu_copy(self, *a, **k):
        raise NotImplementedError

    def load_cpu_copy(self, *a, **k):
        raise NotImplementedError


def make_nonv4(num_logical_pages=40):
    kv = FakeKV()
    a = HiSparseTokenToKVPoolAllocator(
        size=num_logical_pages * PAGE,
        page_size=PAGE,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=kv,
        need_sort=True,
        host_to_device_ratio=2,
    )
    return a, a.hisparse_attn_allocator


def make_v4(c4_pages=40):
    from types import SimpleNamespace
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        DeepSeekV4TokenToKVPool,
        HiSparseC4DevicePool,
    )

    c4 = HiSparseC4DevicePool(
        size=c4_pages * 16,
        page_size=16,
        dtype=torch.float8_e4m3fn,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    outer = object.__new__(DeepSeekV4TokenToKVPool)
    outer.c4_kv_pool = c4
    outer.c4_logical_size = c4_pages * 64 // 4

    logical = SimpleNamespace()
    logical._kvcache = outer
    logical.size_full = 2 * c4_pages * 64
    logical.page_size = 64
    logical.need_sort = True
    logical.clear = lambda: None
    v4 = DeepSeekV4HiSparseTokenToKVPoolAllocator(logical)
    return v4, v4.hisparse_attn_allocator, c4


def sane(hp, magic_possible_pages, tag=""):
    fp = hp.get_all_free_pages().tolist()
    assert len(fp) == len(set(fp)), f"{tag}: dup page in free list {sorted(fp)}"
    assert hp.available_size() <= hp.num_pages * hp.page_size, f"{tag}: over-avail"


# ---------------------------------------------------------------------------
# (1) fresh-tail-empty: pad must NOT fabricate inside the tree's page
# ---------------------------------------------------------------------------


def test_q3_nonv4_prefix_tail_guard():
    """Kept run = one full tree page + 1 token into a second tree page
    (unaligned). Fresh tail empty. Pre-fix: pad fabricates tree page 2 ids;
    the buffer then claims them and the later free releases tree page 2."""
    a, hp = make_nonv4()
    ps = PAGE
    tree_pages = {1, 2}
    for p in tree_pages:
        taken = hp.alloc(ps)  # take whole pages out of the pool as tree-owned
        assert set((taken // ps).tolist()) == {p}, taken // ps

    # request: prefix maps compressed slots 0..ps onto tree device slots
    # 0..ps (pages 1 and the first token of page 2) — fresh tail EMPTY.
    allocated = torch.arange(0, ps + 1, dtype=torch.int64)
    tree_vals = torch.arange(1 * ps, 2 * ps + 1)  # page1 + page2[0]
    a.full_to_hisparse_device_index_mapping[allocated] = tree_vals
    prefix_len = len(allocated)

    need = 2 * ps
    buf = a.alloc_device_buffer(allocated, need, prefix_len=prefix_len)

    # buffer: tree page 1 ids + tree page 2[0] + pad (ps-1) + extra page (ps)
    # pad must NOT be inside tree pages' ranges: ids in [ps, 3*ps) are tree-owned.
    assert set(buf.tolist()) & set(range(1 * ps, 3 * ps)) == set(
        tree_vals.tolist()
    ), "pad/extra collided with tree pages"
    # nothing tree-owned was freed by the fix path
    fp = hp.get_all_free_pages().tolist()
    assert 1 not in fp and 2 not in fp, f"tree page freed: {sorted(fp)}"

    # NOTE (out-of-item-scope but independence-relevant): request_finished ALSO
    # frees the buffer wholesale, which would then release tree prefix pages —
    # the same F-5 class at the free site. This item covers only the pad
    # fabrication; the free-site coverage is dsa's invariant list. Here we
    # model the request's correct free: only the request-OWNED ids.
    owned = buf[~torch.isin(buf, tree_vals)]
    a.free_hisparse_indices(owned)
    fp = hp.get_all_free_pages().tolist()
    assert 1 not in fp and 2 not in fp, (
        f"request free released a tree page: {sorted(fp)}"
    )
    sane(hp, None, "q3-nonv4")


def test_q3_v4_prefix_tail_guard():
    v4, hp, c4 = make_v4()
    ps = v4.hisparse_page_size  # 16
    tree_pages = {1, 2}
    for p in tree_pages:
        taken = hp.alloc(ps)
        assert set((taken // ps).tolist()) == {p}, taken

    allocated = torch.arange(0, ps + 1, dtype=torch.int64)
    tree_vals = torch.arange(1 * ps, 2 * ps + 1)  # page 1 + first token of page 2
    v4.full_to_hisparse_device_index_mapping[allocated] = tree_vals

    need = 2 * ps
    buf = v4.alloc_device_buffer(allocated, need, prefix_len=len(allocated))

    assert set(buf.tolist()) & set(range(1 * ps, 3 * ps)) == set(tree_vals.tolist())
    fp = hp.get_all_free_pages().tolist()
    assert 1 not in fp and 2 not in fp, f"tree page freed: {sorted(fp)}"
    owned = buf[~torch.isin(buf, tree_vals)]
    v4.free_hisparse_indices(owned)
    fp = hp.get_all_free_pages().tolist()
    assert 1 not in fp and 2 not in fp, (
        f"request free released a tree page: {sorted(fp)}"
    )
    sane(hp, None, "q3-v4")


# ---------------------------------------------------------------------------
# (2) fresh tail present: dense fast pad unchanged
# ---------------------------------------------------------------------------


def test_q3_dense_fresh_pad_unchanged():
    """A real fresh tail still pads with the continuation of the request's own
    partial page whole-page-alloc free semantics."""
    a, hp = make_nonv4()
    ps = PAGE
    taken = hp.alloc(2 * ps)  # request owns its pages 1..2
    assert set((taken // ps).tolist()) == {1, 2}, taken
    allocated = torch.arange(0, ps + 1, dtype=torch.int64)
    own_vals = torch.arange(1 * ps, 2 * ps + 1)  # page 1 + first token of page 2
    a.full_to_hisparse_device_index_mapping[allocated] = own_vals
    need = 2 * ps
    buf = a.alloc_device_buffer(allocated, need)
    # dense pad: ids = continuation of token 2*ps (the fabricated range may
    # extend inside own page 2 — request-owned, freed once)
    assert set((buf[: ps + 1] // ps).tolist()) <= {1, 2}, buf
    sane(hp, None, "q3-dense-fresh")


def test_q3_pre_fix_shape_fabricates_inside_tree_page():
    """No prefix bound on the call + empty fresh region shape: document the
    failure that this item guards (module collects pre-fix because it calls
    the allocator's public API only where possible).

    NOTE: with prefix_len=0 (legacy), the guard intentionally does NOT engage:
    that is the code path whose callers were the F-5 problem. The Q3 guard
    only sees split state when callers pass prefix_len. This test documents:
    pre-fix alloc_device_buffer(allocated, need) fabricates ids inside the
    tree page (demonstrated); post-fix this is unchanged BY DESIGN for legacy
    callers — callers must pass prefix_len (F-5 fix)."""
    a, hp = make_nonv4()
    ps = PAGE
    taken = hp.alloc(2 * ps)
    assert set((taken // ps).tolist()) == {1, 2}, taken
    allocated = torch.arange(0, ps + 1, dtype=torch.int64)
    a.full_to_hisparse_device_index_mapping[allocated] = torch.arange(
        1 * ps, 2 * ps + 1
    )
    buf = a.alloc_device_buffer(allocated, 2 * ps)  # legacy, no prefix bound
    # legacy may fabricate ids; the pad continues INSIDE tree page 2's range.
    padded = [x for x in buf.tolist() if 2 * ps + 1 <= x < 3 * ps]
    assert padded, "legacy pad did not fabricate into tree page (harness drift?)"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
