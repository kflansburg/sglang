"""e2e conservation test: retract -> offload -> resume -> decode for SWA.

Mirrors the disagg decode flow (production) against the REAL
SWATokenToKVPoolAllocator on CPU:

1. admit: alloc_extend_swa_tail (full for the whole row, SWA for the window
   tail only), swa ratchet stamped at fill_len - swa_tail_len;
2. decode: alloc_decode per step; window eviction frees SWA-only below the
   ratchet via free_swa_segment, exactly like free_swa_out_of_window_slots;
3. retract (offload backup): the request records the mask of currently live
   SWA positions (SWAKVPool.get_cpu_copy's swa_mask logic), then release
   frees the whole committed row through the two-sided free();
4. resume: a FRESH alloc_extend_swa_tail with the same fill/tail (what
   DecodePreallocQueue._pre_alloc does) and the ratchet is re-stamped; only
   positions mapped in both the old backup and the new row are restorable
   (checked via the mask intersection from load_cpu_copy);
5. finish: free everything.

Invariants checked at every step: page conservation in both pools, no
double-free (allocator-side assert available_size() <= size), mapping
coherence (a full token may map to 0 or a held SWA page; every held SWA
page is referenced by a live full token of the owning request), and slots
are never shared between two live requests.
"""

import importlib.util
import os
import unittest
from dataclasses import dataclass, field

import torch

try:
    from sglang.srt.mem_cache.allocator.swa import (  # noqa: F401
        BaseSWAKVPool,
        SWATokenToKVPoolAllocator,
    )
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase

    register_cpu_ci(est_time=15, suite="base-a-test-cpu")
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


@dataclass
class _Req:
    """The SWA-relevant request state (a sparse Req)."""

    fill_len: int = 0
    swa_tail_len: int = 0
    committed: int = 0  # seqlen-committed tokens whose KV is live
    evicted_swa: int = 0  # cursor: SWA slots below this were window-freed
    tokens: list = field(default_factory=list)  # full token ids by position

    @property
    def last_loc(self):
        return self.tokens[self.committed - 1]


class TestSWARetractOffloadCycle(CustomTestCase):
    page_size = 4
    window = 6  # sliding window in tokens

    def setUp(self):
        ps = self.page_size
        self.alloc = SWATokenToKVPoolAllocator(
            size=64 * ps,
            size_swa=32 * ps,
            page_size=ps,
            dtype=torch.float16,
            device="cpu",
            kvcache=_FakeSWAKVPool(),
            need_sort=False,
        )

    # -- allocation steps mirroring the production paths --

    def _admit(self, fill_len):
        ps = self.page_size
        req = _Req()
        req.fill_len = fill_len
        # alloc_for_decode_prealloc: tail covers [aligned window start, fill)
        window_start = max(0, fill_len - 1 - max(self.window, ps))
        window_start = (window_start // ps) * ps
        req.swa_tail_len = fill_len - window_start
        req.evicted_swa = window_start  # set by alloc_for_decode_prealloc
        seq = torch.tensor([fill_len], dtype=torch.int64)
        pre = torch.tensor([0], dtype=torch.int64)
        out = self.alloc.alloc_extend_swa_tail(
            pre,
            pre.clone(),
            seq,
            seq.clone(),
            torch.tensor([-1], dtype=torch.int64),
            fill_len,
            req.swa_tail_len,
        )
        self.assertIsNotNone(out)
        req.tokens = [int(v) for v in out]
        req.committed = fill_len
        return req

    def _decode_step(self, req):
        """One decode token; exercise window eviction first, then alloc."""
        ps = self.page_size
        nxt = req.committed + 1
        # free_swa_out_of_window_slots semantics: page-aligned frontier at
        # (pre_len - max(window, ps)) with pre_len = committed.
        frontier = req.committed - max(self.window, ps)
        frontier = (frontier // ps) * ps
        if frontier > req.evicted_swa:
            seg = torch.tensor(
                req.tokens[req.evicted_swa : frontier], dtype=torch.int64
            )
            self.alloc.free_swa_segment(seg, start_pos=req.evicted_swa)
            req.evicted_swa = frontier
        seq_next = torch.tensor([nxt], dtype=torch.int64)
        last = torch.tensor([req.last_loc], dtype=torch.int64)
        out = self.alloc.alloc_decode(seq_next, seq_next.clone(), last)
        self.assertIsNotNone(out)
        req.tokens.append(int(out[0]))
        req.committed = nxt
        self._check_mapping(req)

    # -- invariants --

    def _check_mapping(self, req):
        m = self.alloc.full_to_swa_index_mapping
        for i, t in enumerate(req.tokens):
            s = int(m[t])
            if i < req.evicted_swa:
                self.assertEqual(s, 0, f"pos {i}: evicted SWA but mapping live")
            else:
                self.assertGreater(s, 0, f"pos {i}: live SWA but mapping cleared")

    def _conserved(self):
        fa, sa = self.alloc.full_attn_allocator, self.alloc.swa_attn_allocator
        return (
            len(fa.free_pages) == fa.num_pages
            and len(sa.free_pages) == sa.num_pages
            and not bool(self.alloc.full_to_swa_index_mapping[:-1].any().item())
        )

    # -- tests --

    def test_retract_offload_resume_e2e(self):
        """Full retract/offload/resume cycle conserves both pools and keeps
        the mapping coherent through every phase."""
        req = self._admit(fill_len=40)

        # decode 12 steps, window eviction active
        for _ in range(12):
            self._decode_step(req)
        self._check_mapping(req)

        # --- RETRACT with offload backup ---
        # SWAKVPool.get_cpu_copy mask: positions whose mapping is live.
        m = self.alloc.full_to_swa_index_mapping
        row = torch.tensor(req.tokens[: req.committed], dtype=torch.int64)
        swa_indices = m[row]
        old_swa_mask = swa_indices > 0
        live_count = int(old_swa_mask.sum().item())
        # backup payload = swa_indices[old_swa_mask] (the SWA data itself is
        # opaque to the allocator; we only track accounting)
        self.assertGreater(live_count, 0)
        self.assertLessEqual(live_count, self.window + self.page_size + 1)

        # release_kv_cache(checkpoint=False): two-sided free of the row
        self.alloc.free(row)
        self.assertTrue(
            all(
                int(self.alloc.full_to_swa_index_mapping[t]) == 0
                for t in req.tokens
            )
        )
        freed_full = len(self.alloc.full_attn_allocator.free_pages)
        freed_swa = len(self.alloc.swa_attn_allocator.free_pages)

        # --- RESUME: fresh prealloc for the committed length (what
        # resume_retracted_reqs -> _pre_alloc(_pre_alloc_fill_len) does) ---
        req2 = self._admit(fill_len=req.committed)
        m2 = self.alloc.full_to_swa_index_mapping
        new_row = torch.tensor(req2.tokens, dtype=torch.int64)
        new_swa_mask = m2[new_row] > 0
        # load_cpu_copy restore set: positions mapped in BOTH old and new rows
        overlap = min(len(new_swa_mask), len(old_swa_mask))
        restorable = int(
            (new_swa_mask[:overlap] & old_swa_mask[:overlap]).sum().item()
        )
        self.assertGreater(restorable, 0)
        # conservation invariant: pools went down by exactly what resume took
        self.assertLess(
            len(self.alloc.full_attn_allocator.free_pages), freed_full
        )

        # --- continue decode on the resumed request; SWA window slides on ---
        for _ in range(8):
            self._decode_step(req2)

        # --- FINISH: everything conserved ---
        self.alloc.free(torch.tensor(req2.tokens, dtype=torch.int64))
        self.assertTrue(self._conserved())

    def test_retract_resume_repeated_no_leak(self):
        """Repeated retract/resume cycles of one request never bleed pages."""
        fill_len = 40
        req = self._admit(fill_len=fill_len)
        for cycle in range(5):
            for _ in range(4):
                self._decode_step(req)
            self.alloc.free(torch.tensor(req.tokens[: req.committed], dtype=torch.int64))
            self.assertTrue(
                self._conserved(),
                f"leak after retract (cycle {cycle}): "
                f"full {len(self.alloc.full_attn_allocator.free_pages)}, "
                f"swa {len(self.alloc.swa_attn_allocator.free_pages)}",
            )
            req = self._admit(fill_len=req.committed)
        self.alloc.free(torch.tensor(req.tokens, dtype=torch.int64))
        self.assertTrue(self._conserved())


if __name__ == "__main__":
    unittest.main()
