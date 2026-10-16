# SWA KV-cache audit — COVERAGE ROLL-UP (final)

## Verdict table
| Area | Verdict | Evidence | Prod versions affected |
|---|---|---|---|
| F-4 SWA alloc_decode one-sided full-pool page leak | CONFIRMED, FIXED (agent/swa-auditor, approved; MR #14) | CPU repro leaks exactly 1 page pre-fix (features-lead + me, independently); registered tests test_swa_alloc_decode_leak.py (3) pin the fix | v0.5.14, v0.5.15.post1, v0.5.17, v0.5.20, HEAD (byte-similar); backport diffs below |
| Reserved page/token 0 handed to live requests | CLEAN | clear() starts free lists at 1 (paged + token); _release_swa filters swa>0; PureSWA free filters >0; free-path mapping reads guard the 0/padding sentinel | n/a |
| alloc_extend / alloc_extend_swa_tail one-sided leaks | CLEAN | both pre-flight new_pages_available(n_full, n_swa) with exact page counts; unified_hybrid_swa uses ensure_capacity (asserts post-check) | n/a |
| SWA+offload retract guard (decode offload/retract + resume) | CLEAN at HEAD | static trace + registered e2e test_swa_retract_offload_cycle.py: conservation exact after every phase, mapping coherent, 5 cycles zero bleed | n/a |
| Unified tombstone-once (FreeDeviceKVFullOnly sites x3) | CLEAN | all three generation sites consistent with ratchet semantics; _set_swa clears mapping before free_full_segment; tripwires (GUARD expect, v2p>0 filters) back it up | n/a |
| maybe_evict_swa trigger / overlap gating | CLEAN | page-64 math keeps window + page margin; decode_batch_idx==0 skip mirrors running extend; resumes reset counters; SGLANG_OPT_RELEASE_PREFILL_SWA default-OFF (not our configs) | n/a |
| SWA-tail prealloc vs decode radix prefix assert | GUARDED (no bug) | _match_prefix_and_lock caps max_prefix_len = fill_len - swa_tail_len; release_swa drops prefix SWA tier before tail alloc | n/a |
| SWAPrefillBudget admission math | CLEAN | fences (< remaining, +page headroom); over-commit surfaces as alloc failure, not corruption | n/a |
| _filter_swa_cpu_copy chunk boundaries incl. fp8 per-token scales | CLEAN | exact-source exec stress: 8 shape combos x 4 masks; chunk contract + row alignment hold; per-token scale filtering aligned (no per-block scale pools exist) | n/a |
| Pattern sweep for alloc-pair non-rollback siblings | CLEAN | only alloc_decode had the shape; all siblings pre-check (swa.py alloc_extend/x2, unified x4) | n/a |

## Files & functions read (all at origin/main HEAD)
- mem_cache/allocator/swa.py: SWATokenToKVPoolAllocator — alloc, alloc_extend, alloc_extend_swa_tail, alloc_decode, free*, groups, clear/resize, reclaim_for_prealloc, evict_to_free_tokens, _expand_to_full_pages, mapping writers — full trace. PureSWATokenToKVPoolAllocator — full trace.
- mem_cache/allocator/paged.py + token.py: sub-allocator alloc/free semantics (merge_and_sort_free, release_page_ids, debug checks) for all used paths.
- mem_cache/swa_memory_pool.py: full read (set_kv_buffer, move_kv_cache, get_cpu_copy, load_cpu_copy, _filter_swa_cpu_copy).
- mem_cache/pure_swa_radix_cache.py: full read.
- mem_cache/base_swa_memory_pool.py: full read (contract surface).
- mem_cache/common.py: free_swa_out_of_window_slots, free_kv_row_segments, backup_kv_cache, restore_kv_cache, discard_kv_cache_backup, release_kv_cache, _release_overallocated_kv_indices.
- mem_cache/base_prefix_cache.py: free_kv_row, evict flow, claim_kv_row, on_release; radix_cache.py: _insert_cache, checkpoint, evict, lock flows (SWA-relevant lines).
- managers/schedule_batch.py: maybe_evict_swa, _swa_eviction_trigger, _evict_swa, swa_dead_lo/floor, evicted-seqlen cursors, reset_for_retract, release_req, offload_kv_cache, load_kv_cache, retract_all.
- disaggregation/decode.py: _swa_tail_len, _swa_retractable_len, _prealloc_required_tokens/reclaim paths, _match_prefix_and_lock, _pre_alloc, alloc_for_decode_prealloc, resume_retracted_reqs, _swa_payload (KV transfer).
- mem_cache/prefill_budget.py: SWAPrefillBudget + estimate_swa_kv_tokens.
- mem_cache/allocator/unified_hybrid_swa.py: all alloc + free entries (tombstone semantics).
- unified_cache/unified_tree_core.py + components/swa.py: FreeDeviceKVFullOnly generation sites.
- mem_cache/allocation.py: alloc_for_decode_default, alloc_paged_token_slots_decode, alloc_for_spec_decode — caller-side page-guarantee analysis.
- mem_cache/memory_pool.py: quantized get/load_cpu_copy (fp8 per-token scales).

## Known-class <-> shipped registered tests (which pin it, what they do NOT cover)
| Known class | Pinned by shipped? | Notes |
|---|---|---|
| Page 0 handed to live requests | partial (free-path debug assertions in test_swa_unittest) | no alloc-path test of the sentinel |
| alloc_decode one-sided leak | NOW: test_swa_alloc_decode_leak.py (this work) | un-pinned before F-4 |
| SWA evict interval cadence | test_swa_eviction_interval.py | covers _swa_eviction_trigger gating |
| SWA ring page return over-credit | test_swa_ring_page_return.py | ring mode only in v0.5.20/HEAD |
| SWA offload cpu-copy mask algebra | test_swa_cpu_copy_filter.py | mask-only; NO end-to-end retract cycle |
| Retract->offload->resume conservation | NOW: test_swa_retract_offload_cycle.py (this work) | un-pinned before |
| Unified SWA shared virtual ids | test_unified_swa_shared_virtual_ids.py | unified-side trace (pools cross-own) |
| SWA locked-full recover (unified) | test_swa_locked_full_recover_unified.py | unified component path |
| Paged pure-SWA release semantics | test_pure_swa_radix_cache.py | all-SWA model path |
| Tail-only buffer admissions | test_buffer_swa_tail_only.py | buffer-mode surface |

Uncovered branches: spec-decode SWA eviction under verify batches (deeper spec logic with spec-auditor); GPU-kernel-level write bounds (out of CPU scope).

---

Scope: `mem_cache/allocator/swa.py`, `mem_cache/swa_memory_pool.py`,
`mem_cache/pure_swa_radix_cache.py`, `mem_cache/base_swa_memory_pool.py`,
SWA eviction in `managers/schedule_batch.py` (`maybe_evict_swa`,
`_swa_eviction_trigger`, `_evict_swa`), disagg decode SWA tail prealloc +
retract + offload in `disaggregation/decode.py` and `mem_cache/common.py`
(`free_swa_out_of_window_slots`, `backup_kv_cache`, `release_kv_cache`).

Production configs assumed: page 64, fp8_e4m3, disaggregated P/D,
--swa-full-tokens-ratio 0.075-0.1, decode-side radix cache + SWA tail
prealloc, decode KV offload retraction backup.

## CONFIRMED

### F-4 (was C1). SWA `alloc_decode` one-sided full-pool page leak — FIXED on agent/swa-auditor
Verified independently by features-lead (msg #202). CPU PoC driving the
REAL allocator classes reproduces the leak at HEAD (full free pages
drop by 1 after alloc_decode returns None when SWA is exhausted):
`test/swa_audit/test_swa_allocator_cpu.py` (loader
`conftest_real_alloc.py` injects stubs for triton/platform deps and
imports base/token/paged/swa from their repo paths).

Fix: joint pre-flight of the exact decode page count before touching
either sub-pool (mirrors alloc_extend / UnifiedSWAAllocator.alloc_decode
which already do this; unified_hybrid_swa.py uses ensure_capacity).
Rollback-via-free was considered and REJECTED with a runnable counter-
example (page=4, real allocators): a decode batch [boundary req A,
non-boundary req B] makes full alloc_decode return [12, 6]; 6 = B's
last_loc+1 inside page 1 which request R STILL HOLDS. paged free() is
page-granular, so rollback free([12,6]) returns page 1 to the free list
next_alloc then hands page 1 to a third request -> two live requests share
one KV page (cross-request aliasing). alloc_decode outputs alias live
pages and cannot be identified from the output, so the failure path must
be declared before consumption. Pre-flight has identical external OOM
semantics; mapping untouched (written only after both allocations
succeed, per features-lead msg #256).

Pattern sweep (board #22): the 'alloc A; alloc B; if None or None: return'
non-rollback shape exists only in swa.py::alloc_decode (HEAD; also at
v0.5.14:294-297). alloc_extend and alloc_extend_swa_tail pre-check via
new_pages_available; unified_hybrid_swa.py joint-checks ensure_capacity in
alloc / alloc_extend / alloc_extend_swa_tail / alloc_decode, and asserts
post-check sub-alloc success. hisparse.py / unified_mamba.py: no pair
allocations. allocation.py is_dsv4 branch is NPU-only
(hasattr(allocator, 'c128_attn_capacity')); unreachable in our fleet.

Affected: v0.5.14, v0.5.15.post1, v0.5.17, v0.5.20 (Kimi K3 branch-era),
HEAD. All heap alloc/free bookkeeping is logical; the leak is a slow
capacity bleed (full-pool pages vanish without mapping or owner) until
disagg decode admission trips its "KV cache is full! Bug in memory
estimation" assert in `_pre_alloc`.
Backport: the same pre-flight applies textually to all four tags — in
each, alloc_decode sits next to an identical new_pages_available helper
and get_num_new_pages import (v0.5.14:157/278, v0.5.15.post1:166/287,
v0.5.17:166/287, v0.5.20:234/376). v0.5.20 keeps the HEAD _swa_req_ring
guard; insert the pre-flight after it (same as HEAD).

Regression tests (CPU, no GPU): test/swa_audit/test_swa_allocator_cpu.py
- test_alloc_decode_one_sided_leak_FIXED (fails without the fix)
- test_alloc_decode_success_then_free (round trip conservation)
- test_conservation_after_lifecycle (randomized extend/decode/free sweep)

## CANDIDATES (pending independent verification)

(none open right now)

### C1. SWA `alloc_decode` one-sided leak of full-attn pages — NOT FIXED upstream, all prod versions
File: `python/sglang/srt/mem_cache/allocator/swa.py`,
`SWATokenToKVPoolAllocator.alloc_decode` (HEAD: lines ~436-458; identical
text in v0.5.14, v0.5.15.post1, v0.5.17, and by inheritance HEAD/v0.5.20).

```
alloc_full_indices = self.full_attn_allocator.alloc_decode(...)   # mutates full pool
alloc_swa_indices  = self.swa_attn_allocator.alloc_decode(...)    # may return None
if alloc_full_indices is None or alloc_swa_indices is None:
    return None          # <- full-side pages already popped -> LEAKED
```

`PagedTokenToKVPoolAllocator.alloc_decode` (allocator/paged.py) returns None
only after it can prove it lacks pages, but by then the OTHER pool has
already consumed pages and there is no rollback. The pre-check that
`alloc_extend` / `alloc_extend_swa_tail` have (`new_pages_available(...)`
before touching either pool) is missing here.

Reachability: the invariant "full fails before swa" does NOT hold.
Full and SWA pools do not allocate in lockstep: decode-side prealloc uses
`alloc_extend_swa_tail` (full for whole prompt, SWA only for the window
tail), so under `--swa-full-tokens-ratio 0.075-0.1` the SWA pool can be
exhausted (many live windows) while the full pool still has pages. Then a
page-boundary decode step allocs 1 full page, fails the SWA side, returns
None, and the full page(s) are gone from the free list without any mapping.
The caller (`alloc_paged_token_slots_decode`, mem_cache/allocation.py:594)
treats None as fatal OOM and raises — so each hit leaks a page right at the
moment the pool is under maximum pressure; the capacity accounting then
drifts, so later prealloc budgets over-promise (disagg decode asserts
"KV cache is full! Bug in memory estimation." in `_pre_alloc`,
disaggregation/decode.py ~2303). Effect: permanent full-pool page leak ->
spurious OOM / crash, no corruption.

Note this is the known class "SWA alloc_decode one-sided leak"; verified the
same code shape exists in every production tag fetched (v0.5.14,
v0.5.15.post1, v0.5.17; v0.5.20 same file lineage) and at origin/main HEAD.

## REFUTED / GUARDED (checked, no bug)

- UnifiedSWAAllocator (unified_hybrid_swa.py) one-sided-leak pattern: all
  alloc entries (alloc, alloc_extend, alloc_extend_swa_tail, alloc_decode)
  run joint `ensure_capacity(...)` pre-checks before any sub-pool mutation;
  sub-pool failure after the check is asserted impossible. Clean.
- Asymmetric segment frees (free_swa / free_swa_segment tombstone-aware via
  v2p>0; free_full "caller tombstoned" contract; free_full_segment):
  every external caller satisfies the contract —
  unified_radix_cache FreeDeviceKVFullOnly (SWA tier freed by tree actions
  first), unified_cache/components/swa.py RecoverSWAWithLockedFull static
  path calls clear_full_to_swa_mapping(incoming_full) inside _set_swa
  before alloc.free_full_segment (checks out with the _SWA_PEER_RELEASED
  expect), unified path uses unified.free() which tombstone-filters.
- _swa_eviction_trigger / maybe_evict_swa math at page 64: eviction frontier
  N-1-max(window,page) page-floored keeps >= the live window plus one page;
  overlap decode_batch_idx==0 skip mirrors the still-running extend; resumes
  reset decode_batch_idx=0 so post-retract decode cannot evict early.
  SGLANG_OPT_RELEASE_PREFILL_SWA (release_swa_prefix_lock) is opt-in,
  default OFF — not production-relevant.

- `alloc_extend` / `alloc_extend_swa_tail` one-sided leak: both pre-flight
  `new_pages_available(num_full_pages, num_swa_pages)` with the exact page
  counts each sub-allocator will consume, so neither sub-pool alloc can fail
  after the other succeeded. Guarded.
- SWA tail prealloc vs decode radix prefix (`total_prefix_len >
  window_start` making `alloc_extend_swa_tail`'s
  `swa_tail_len <= extend_num_tokens` assert fail): guarded by
  `_match_prefix_and_lock` capping the match with
  `max_prefix_len = fill_len - swa_tail_len(fill_len)` and by
  `release_swa(req.lock)` dropping the prefix's SWA tier
  (disaggregation/decode.py:837-, 1488-).
- Reserved page 0 handed to live requests: both `Paged..clear()` and
  `TokenToKVPoolAllocator.clear()` start the free list at page 1;
  `_release_swa` filters `swa_indices > 0` so a 0 read from the mapping can
  not enter the SWA free list; `PureSWA..free/free_swa` filters
  `free_index > 0`. No allocator path frees page/token 0 seen so far
  (need harness sweep for req_to_token rows containing 0 in a freed
  segment — open).
- Retract-with-offload SWA restore (`Req.offload_kv_cache` /
  `load_kv_cache` via SWAKVPool.get_cpu_copy/load_cpu_copy): old/new
  `swa_mask` intersection logic is consistent — copies exactly the
  positions mapped both before and after, drops stale tail, leaves the one
  newest token for recompute. `reset_for_retract` clears cursors via
  `mark_kv_released()` and `resume_retracted_reqs` re-runs
  `alloc_extend_swa_tail` which re-stamps `set_evicted_seqlen(SWA, ...)`.
  No double-free/leak found in this path.

## Shipped-test coverage map (test/registered/unit/mem_cache/)
- test_swa_unittest.py (24 tests): free paths — paged free clears full page
  mapping, free-group ownership of deferred indices / mapping-read timing,
  free_full full-only semantics + deferral, floor-decided row frees, segment
  free flags double-free of SWA peer pages (peer-mapped invariant),
  free_swa_segment debug checks, tail page release. Pins the free() ordering
  and tombstone conventions; NO alloc-failure pairing tests (gap F-4 fell
  through), no retract/offload tests.
- test_pure_swa_radix_cache.py: all-SWA RadixCache release semantics.
- test_swa_eviction_interval.py: _swa_eviction_trigger interval gating.
- test_swa_locked_full_recover_unified.py: unified RecoverSWAWithLockedFull.
- test_swa_ring_page_return.py: per-request SWA ring page return regression.
- test_swa_cpu_copy_filter.py: SWA offload cpu-copy masking (retract backup
  payload filtering; unit only — no end-to-end retract+restore test).
- test_buffer_swa_tail_only.py: tail-only SWA backup admission keys.
Gaps: alloc_decode failure pairing (now covered by my fix test); full
retract->offload->resume->decode conservation; maybe_evict_swa under
overlap scheduler; spec-decode (EAGLE/draft KV) SWA eviction interplay.

## Coverage summary (scope = board #9)
- allocator/swa.py SWATokenToKVPoolAllocator: all alloc entries (alloc,
  alloc_extend, alloc_extend_swa_tail, alloc_decode) and all free entries
  (free, free_swa, free_swa_segment, free_full, free_full_segment,
  free_segment, free groups, clear/resize) traced; CPU harness sweeps
  alloc/decode/free cycles with conservation+mapping invariants. 1 bug
  (F-4), fix + tests on agent/swa-auditor.
- allocator/swa.py PureSWATokenToKVPoolAllocator: single-pool semantics;
  free_segment overridden to avoid double release; page-0 filter present.
  No issues found.
- swa_memory_pool.py SWAKVPool: set_kv_buffer / get/load_cpu_copy mask
  logic traced (retract offload) — consistent. Mapping tensor bounds
  (size+ps pad + trailing -1 sentinel) cover last_loc=-1 and padded reads.
- pure_swa_radix_cache.py: thin RadixCache subclass; evict param routing
  (max of num/swa tokens); no state of its own. Clean.
- spec-decode (EAGLE) SWA interplay: alloc_for_spec_decode over-reserve is
  tracked via kv_allocated_len and released through the standard
  free_kv_row/overallocated path; maybe_evict_swa frontier pre_len=seqlen-1
  cannot touch draft slots (they live beyond seqlen). Row-width fail-fast
  assert for page>1 topk>1 present. No leak found in the SWA-relevant part
  (deeper spec-internal logic belongs to spec-auditor).
- scheduler maybe_evict_swa / _swa_eviction_trigger / _evict_swa: trigger
  math and overlap gating reviewed (see REFUTED section).
- SWAPrefillBudget: admission estimates have fences (< remaining, +page
  headroom); transient vs permanent shortfall handling reviewed; budget
  over-commit would surface as alloc failure (supported by fences) not as
  corruption. Clean.
- retract+offload guard (known class): release_req -> backup_kv_cache
  (cpu_tensor: get_cpu_copy incl. swa mask) -> release_kv_cache ->
  mark_kv_released clears cursors -> resume: _pre_alloc re-stamps
  evicted_seqlen via alloc_extend_swa_tail, restore writes only
  old-mapped AND new-mapped positions. Consistent; see REFUTED.

## Unified tombstone-once trace (board #9 follow-up, msg #301)
FreeDeviceKVFullOnly (full-only free; SWA side assumed already released)
is generated in exactly three places, and each is consistent:
1. unified_tree_core.py _insert_dup handling (~:1337-1349): the dup slice
   straddling the request's SWA eviction ratchet is SPLIT at
   get_evicted_seqlen(SWA) - abs_start; below -> FreeDeviceKVFullOnly (SWA
   was window-freed for this row), above -> FreeDeviceKV (both). The
   ratchet is advanced only when free_swa_out_of_window_slots actually
   frees, so by construction SWA is gone exactly below it.
2. components/swa.py _recover_from_incoming (~:469): frees the OLD node's
   full value full-only; the old SWA pages stay bound to the node whose
   component SWA value is deliberately kept (_publish_window rebuilds the
   window's SWA from the adopted incoming full value). Freeing both sides
   would double-free pages the node still references.
3. components/swa.py RecoverSWAWithLockedFull (~:1560): _set_swa moves the
   incoming mapping to kept_full (static: clear+set; unified: page
   transfer), then free_full_segment(incoming_full) — SWA side already
   claimed, mapping cleared first.
Post-retract consistency: restore re-stamps the ratchet to
fill_len - swa_tail_len (alloc_extend_swa_tail) and the row below it has
mapping==0, so a later insert's FullOnly/FreedeviceKV split again matches
reality.
Tripwires that would catch a violation anyway: _SWA_PEER_RELEASED expect
(GUARD) on the static free_full*, v2p>0 tombstone filters on the unified
free paths. Verdict: invariant holds by construction at all three sites;
no leak/double-free found.

## _filter_swa_cpu_copy / quantized-pool CPU copies (fp8_e4m3 production)
SWAKVPool.get_cpu_copy/load_cpu_copy re-chunking verified shape-consistent:
_filter_swa_cpu_copy flattens per-layer chunks, applies the per-token
row_mask to every tensor the chunk carries (k, v, plus per-token fp8 k/v
scales — memory_pool.py quantized get_cpu_copy emits PER-TOKEN scales via
_read_scales(layer_id, chunk_indices), so the mask aligns; a per-BLOCK
scale layout would misalign but no such pool exists), then re-chunks with
the same cpu_offloading_chunk_size the loader uses, so load_cpu_copy's
chunk indexing aligns (it also asserts per-chunk row counts). No fp8-scale
garbling found. SGLANG_USE_HND_KVCACHE + CPU offload is assert-rejected
(we don't use HND).

## In progress / open
- MR #7 (F-4) in merge queue.
- Backport patches for v0.5.14/v0.5.15.post1/v0.5.17/v0.5.20 (awaiting
  features-lead's preference: findings-file reference diffs vs branches).

## retract->offload->resume scheduler-e2e conservation (board #39)
Registered CPU test test_swa_retract_offload_cycle.py pins the whole
production decode flow against the real allocator: tail-prealloc admit,
window eviction via free_swa_segment at the page-aligned frontier
(committed - max(window, ps)), retract with the get_cpu_copy mask, fresh
tail prealloc on resume, restorable intersection, repeated cycles.
Result: conservation exact at every phase; mapping always coherent; no
bleed over 5 cycles. The SWA+offload retract guard class is CLEAN at
HEAD as seen from the allocator; the earlier static verdict now has a
runtime pin.

## F-4 backport reference diffs (apply with `patch` or 3-way; verified anchors present in each tag)

### v0.5.14 / v0.5.15.post1 / v0.5.17 — IDENTICAL alloc_decode body; one patch serves all three
```diff
--- a/python/sglang/srt/mem_cache/allocator/swa.py
+++ b/python/sglang/srt/mem_cache/allocator/swa.py
@@ def alloc_decode
         assert self.page_size > 1
         swa_last_loc = self.translate_loc_from_full_to_swa(last_loc)
 
+        # F-4: both pools must serve the same page crossings; pre-flight the
+        # exact count so a SWA shortfall cannot strand freshly-popped full
+        # pages (rollback-by-free would release live pages of non-boundary
+        # requests: their alloc_decode output aliases a still-held page).
+        num_new_pages = get_num_new_pages(
+            seq_lens=seq_lens_cpu, page_size=self.page_size, decode=True
+        )
+        if not self.new_pages_available(num_new_pages, num_new_pages):
+            return None
+
         alloc_full_indices = self.full_attn_allocator.alloc_decode(
             seq_lens, seq_lens_cpu, last_loc
         )
```
Anchors verified: `new_pages_available` at v0.5.14:157, v0.5.15.post1:166,
v0.5.17:166; `get_num_new_pages` imported from sglang.srt.utils.common in
all three; alloc_decode body byte-identical across the three tags.
Note: these tags lack the `_swa_req_ring` branch, so the patch hunk context
is simply `assert self.page_size > 1; swa_last_loc = ...`.

### v0.5.20 — same patch inserted AFTER the `_swa_req_ring` early return
```diff
--- a/python/sglang/srt/mem_cache/allocator/swa.py
+++ b/python/sglang/srt/mem_cache/allocator/swa.py
@@ def alloc_decode
             return self.full_attn_allocator.alloc_decode(
                 seq_lens, seq_lens_cpu, last_loc
             )
 
         swa_last_loc = self.translate_loc_from_full_to_swa(last_loc)
 
+        # F-4 pre-flight: see comment in module history / PR notes — a SWA
+        # shortfall after full pops pages strands them (never rolled back).
+        num_new_pages = get_num_new_pages(
+            seq_lens=seq_lens_cpu, page_size=self.page_size, decode=True
+        )
+        if not self.new_pages_available(num_new_pages, num_new_pages):
+            return None
+
         alloc_full_indices = self.full_attn_allocator.alloc_decode(
             seq_lens, seq_lens_cpu, last_loc
         )
```
Anchors verified: `new_pages_available` at v0.5.20:234 (ring-aware);
`get_num_new_pages` imported in v0.5.20; `_swa_req_ring` early-return block
directly precedes `swa_last_loc`. The Kimi K3 branch (base ~v0.5.16) sits
between .15p1 and .17 in this file and matches the first diff unless it
carries the ring back-port; check for `_swa_req_ring` to pick the variant.

### Kimi K3 (mamba-radix cache fork): verify presence of
`SWATokenToKVPoolAllocator._swa_req_ring`; if present use the v0.5.20
variant else the v0.5.14/.15p1/.17 variant. Same bug shape confirmed via
git tags spanning the era.

## _filter_swa_cpu_copy chunk-boundary stress (fp8 approved deep-dive) — CLEAN
Method: exec() of the exact source of _filter_swa_cpu_copy from
swa_memory_pool.py (zero transcription drift) against synthetic per-layer
chunk lists mimicking a quantized pool (4 tensors per chunk: k, v, k_scale,
v_scale), across (nrows, chunk) in {(7,4),(8,4),(9,4),(1,4),(3,1),
(16,8192),(8193,8192),(3,8192)} x masks {all, none, sparse, tail}.
Invariants held: per-chunk tensor row-count alignment; re-chunk emits
chunk-sized pieces except the last; total filtered rows == mask.sum();
empty mask -> all chunks empty/[] (load path's `swa_indices.numel()==0`
early-returns before touching buffers); all-true mask -> identical object
returned (fast path). Combined with the earlier load_cpu_copy chunk-index
contract analysis: no row/scale desync at chunk boundaries, including
fp8_e4m3 per-token scales.
