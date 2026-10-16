"""Regression for F-4: SWA alloc_decode one-sided full-pool page leak.

SWATokenToKVPoolAllocator.alloc_decode used to allocate the FULL pool first
and return None when the SWA pool could not serve the same page crossings,
leaking the full-side pages forever (no mapping, no owner). Reachable with a
small --swa-full-tokens-ratio once decode-side SWA tail prealloc leaves the
SWA pool tighter than the full pool; the disagg decode admission assert
("KV cache is full! Bug in memory estimation") then fires spuriously hours
later. Affects v0.5.14..v0.5.20 and main.

Fix: pre-flight the exact boundary page count (get_num_new_pages(decode=True))
through new_pages_available() before touching either sub-pool, mirroring
alloc_extend and UnifiedSWAAllocator.alloc_decode. A rollback of the full
side's output would be WRONG: a non-boundary request's alloc_decode result is
last_loc+1 inside a page it still holds, and paged free() is page-granular -
freeing the result set releases live pages (verified on CPU: cross-request
page aliasing). The mapping needs no cleanup on failure paths because it is
written only after both allocations succeed.

These tests drive the REAL allocator classes with a faked KV pool (only
full_kv_pool/swa_kv_pool attributes and register_mapping are consumed on this
path), CPU tensors, page_size=4, need_sort=False (naive kernels on CPU).
"""

import importlib.util
import os
import unittest

import torch

try:
    from sglang.srt.mem_cache.allocator.swa import (  # noqa: F401
        BaseSWAKVPool,
        SWATokenToKVPoolAllocator,
    )
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase

    register_cpu_ci(est_time=10, suite="base-a-test-cpu")
except ImportError:  # no full sglang install here (e.g. bare merge-queue venv)
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _spec = importlib.util.spec_from_file_location(
        "_swa_allocator_loader", os.path.join(_HERE, "_swa_allocator_loader.py")
    )
    _loader = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_loader)
    _swa_mod = _loader.load()[3]
    SWATokenToKVPoolAllocator = _swa_mod.SWATokenToKVPoolAllocator
    BaseSWAKVPool = _swa_mod.BaseSWAKVPool
    CustomTestCase = unittest.TestCase


class _FakeSWAKVPool(BaseSWAKVPool):
    """SWAKVPool is not constructed here; the allocator only reads these."""

    swa_req_ring_size = None
    swa_uses_full_indices = False

    def __init__(self):
        self.full_kv_pool = None
        self.swa_kv_pool = None
        self.full_to_swa_index_mapping = None

    def register_mapping(self, mapping):
        self.full_to_swa_index_mapping = mapping

    def translate_loc_from_full_to_swa(self, kv_indices):
        return self.full_to_swa_index_mapping[kv_indices]


def _make_allocator(page_size, full_pages, swa_pages):
    return SWATokenToKVPoolAllocator(
        size=full_pages * page_size,
        size_swa=swa_pages * page_size,
        page_size=page_size,
        dtype=torch.float16,
        device="cpu",
        kvcache=_FakeSWAKVPool(),
        need_sort=False,
    )


def _extend(allocator, num_tokens):
    """Realistic alloc_extend call for a fresh request (prefix 0)."""
    seq = torch.tensor([num_tokens], dtype=torch.int64)
    prefix = torch.tensor([0], dtype=torch.int64)
    out = allocator.alloc_extend(
        prefix, prefix.clone(), seq, seq.clone(), torch.tensor([-1]), num_tokens
    )
    return out


class TestSWAAllocDecodeLeak(CustomTestCase):
    def test_alloc_decode_fail_returns_without_touching_pools(self):
        """F-4 regression: exhaust the SWA pool while FULL has pages; a boundary
        decode must return None and leave BOTH pools unchanged."""
        ps = 4
        allocator = _make_allocator(ps, full_pages=16, swa_pages=2)
        # Take one page off each pool.
        out = _extend(allocator, ps)
        self.assertIsNotNone(out)
        self.assertEqual(len(allocator.swa_attn_allocator.free_pages), 1)
        full_after_first = len(allocator.full_attn_allocator.free_pages)

        # Empty the SWA pool, FULL keeps 14 pages.
        out2 = _extend(allocator, ps)
        self.assertIsNotNone(out2)
        self.assertEqual(len(allocator.swa_attn_allocator.free_pages), 0)
        self.assertEqual(
            len(allocator.full_attn_allocator.free_pages), full_after_first - 1
        )

        # Boundary decode: seq_lens_next = ps + 1 -> (ps) % ps == 0 -> crossing.
        seq_next = torch.tensor([ps + 1], dtype=torch.int64)
        last_loc = torch.tensor([int(out[-1])], dtype=torch.int64)
        result = allocator.alloc_decode(seq_next, seq_next.clone(), last_loc)
        self.assertIsNone(result, "expected decode OOM: SWA pool exhausted")

        # The full pool must be untouched (pre-fix: one page leaked).
        self.assertEqual(
            len(allocator.full_attn_allocator.free_pages), full_after_first - 1
        )
        full_pages = allocator.full_attn_allocator.free_pages
        self.assertEqual(len(torch.unique(full_pages)), len(full_pages))
        # The failed call wrote no mapping: the previously-exhausted SWA free
        # list and the (unchanged) full side say nothing new was consumed.
        self.assertEqual(len(allocator.swa_attn_allocator.free_pages), 0)

    def test_alloc_decode_success_then_free_conserves(self):
        """Round trip: successful boundary decode plus free leaves both pools
        at initial capacity and the mapping all zero."""
        ps = 4
        allocator = _make_allocator(ps, full_pages=16, swa_pages=16)
        out = _extend(allocator, ps)
        self.assertIsNotNone(out)
        seq_next = torch.tensor([ps + 1], dtype=torch.int64)
        last_loc = torch.tensor([int(out[-1])], dtype=torch.int64)
        dec = allocator.alloc_decode(seq_next, seq_next.clone(), last_loc)
        self.assertIsNotNone(dec)
        allocator.free(torch.cat([out, dec]))
        self.assertEqual(
            len(allocator.full_attn_allocator.free_pages),
            allocator.full_attn_allocator.num_pages,
        )
        self.assertEqual(
            len(allocator.swa_attn_allocator.free_pages),
            allocator.swa_attn_allocator.num_pages,
        )
        mapping = allocator.full_to_swa_index_mapping[:-1]
        self.assertFalse(bool(mapping.any().item()))

    def test_alloc_decode_failure_is_recallable_after_room_frees(self):
        """After the failed decode, freeing SWA room lets the same call succeed -
        the failed call must not have consumed anything either side."""
        ps = 4
        allocator = _make_allocator(ps, full_pages=16, swa_pages=2)
        out = _extend(allocator, ps)
        out2 = _extend(allocator, ps)
        self.assertEqual(len(allocator.swa_attn_allocator.free_pages), 0)
        seq_next = torch.tensor([ps + 1], dtype=torch.int64)
        last_loc = torch.tensor([int(out[-1])], dtype=torch.int64)
        self.assertIsNone(allocator.alloc_decode(seq_next, seq_next.clone(), last_loc))
        # Free the second request entirely; room exists again.
        allocator.free(out2)
        self.assertGreaterEqual(len(allocator.swa_attn_allocator.free_pages), 1)
        dec = allocator.alloc_decode(seq_next, seq_next.clone(), last_loc)
        self.assertIsNotNone(dec)
        allocator.free(torch.cat([out, dec]))
        self.assertEqual(
            len(allocator.full_attn_allocator.free_pages),
            allocator.full_attn_allocator.num_pages,
        )
        self.assertEqual(
            len(allocator.swa_attn_allocator.free_pages),
            allocator.swa_attn_allocator.num_pages,
        )


if __name__ == "__main__":
    unittest.main()
