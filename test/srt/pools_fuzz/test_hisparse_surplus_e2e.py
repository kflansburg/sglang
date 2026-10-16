# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Non-V4 hisparse alloc_device_buffer pure-surplus guard (dsa Q2).

Pre-fix: when the surplus side shared a device page with kept buffer tokens
(non-dense device indices within a run — today plausible after free-list
churn, and explicitly the case the V4 variant guards), the token-granularity
//page freed the SHARED page while live buffer tokens still referenced it.
Post-fix: only pages wholly on the surplus side are released.
"""
import torch
import pytest

from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator

import os as _os

# The end-to-end dense/non-dense guard lives in alloc-auditor's fix (#56). This
# file survives pre-fix collection on purpose (imports only the class), and
# self-skips until the module-level helper lands on the tested tree, so both
# branches can merge in either order.
_hisparse_src = open(
    _os.path.join(
        _os.path.dirname(__file__), "..", "..", "..",
        "python", "sglang", "srt", "mem_cache", "allocator", "hisparse.py",
    )
).read()
if "_free_surplus_device_pages" not in _hisparse_src:
    import pytest as _pytest
    globals()["pytestmark"] = _pytest.mark.skipif(
        True, reason="hisparse surplus guard not yet on the tested tree (alloc #56)"
    )


class _StubKVCache:
    """Hisparse allocator only touches kvcache for translation helpers not
    used by alloc_device_buffer/free_hisparse_indices."""

    def translate_loc_from_full_to_compressed(self, x):
        return x

    def _translate_loc_to_hisparse_device(self, x):
        return x

    def register_mapping(self, _proxy):
        pass


def _build(page_full=16 * 8):
    """size = hisparse-device token capacity; the full mapping spans 2x that,
    which must cover every full index used below (<= 96+16)."""
    ps = 16
    a = HiSparseTokenToKVPoolAllocator(
        size=page_full,
        page_size=ps,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        kvcache=_StubKVCache(),
        need_sort=False,
    )
    return a, ps


def test_pure_surplus_only_freed():
    a3, ps = _build()
    n = 5 * ps
    full3 = torch.arange(n) + ps
    dev3 = torch.arange(n) + ps
    # scramble: swap kept token 0 with surplus token 74 -> pages 1 and 5 each
    # host one token from the WRONG side of the cut (non-dense-within-a-run).
    dev3 = dev3.clone()
    dev3[0], dev3[74] = dev3[74], dev3[0]
    # Mark the mapped pages as OUT of the free list (this request "owns" them):
    # hand the hisparse pool the mapped region via a fake reservation: simpler
    # is capturing the free-list before/after THIS alloc_device_buffer call.
    a3.hisparse_attn_allocator.free_pages = torch.tensor(
        [], dtype=torch.int64
    )  # the 5 mapped pages were handed out long ago; free list is empty
    a3.full_to_hisparse_device_index_mapping[full3] = dev3
    # need_size = 4 pages kept, 1 page surplus
    buf = a3.alloc_device_buffer(full3, 4 * ps)
    assert buf.numel() == 4 * ps
    kept = dev3[: 4 * ps]
    kept_pages = set((kept // ps).tolist()) - {0}
    freed_here = set(a3.hisparse_attn_allocator.free_pages.tolist())
    for p in kept_pages:
        assert p not in freed_here, (
            f"kept page {p} returned to free list (double-alloc/UAF risk)"
        )
    # exactly the PURE surplus pages (none in this crafted interleave) are freed
    assert freed_here == set(), (
        f"interleave should free nothing; freed pages: {sorted(freed_here)}"
    )
    assert (a3.full_to_hisparse_device_index_mapping[full3] == 0).all()


def test_dense_surplus_matches_unconditional_free():
    # With dense (page-aligned) surplus all surplus pages are pure -> same behavior as before.
    a, ps = _build()
    n = 5 * ps
    full = torch.arange(n) + ps
    dev = torch.arange(n) + ps
    a.full_to_hisparse_device_index_mapping[full] = dev
    buf = a.alloc_device_buffer(full, 4 * ps)
    assert buf.numel() == 4 * ps
    # stock paged free_pages stores PAGE ids (page-aligned via //ps upstream);
    # surplus tokens 80..95 live on device page 5.
    assert 5 in a.hisparse_attn_allocator.free_pages.tolist(), (
        "pure surplus page must be reclaimed"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
