# SWA/mamba/spec features audit — COVERAGE ROLL-UP
MERGES on dev: F-4 SWA leak @28acaf2 (MR#14) | replayssm lifecycle @9b15563 (MR#36) | F-1 fix + #69 R-1 fix train @4d0e4fe (agent/dsa-auditor MR#?) + F-1 durable PoC @4e9842c (MR#46) | R-1 durable witness @66d6ede (MR#53).
HOTFIX PATCHES (v0.5.14/.15p1/.17[.20 F-4]) live in repo-root hotfix/ — F-4 x4, FF-1 x3, all git-apply+compile verified against pristine tags; README with per-tag apply commands, TP=1 no-op note, rollback guidance, HEAD refutation route (unified pool_hits MIN-reduce = hicache-lead C4).



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


## CARRY-OVER from mamba-auditor (branch agent/swa-auditor; item #8 now mine)

### CANDIDATE 1 (deferred mamba COW source eviction window) — ALIVE (HiCache mode);
  my earlier REFUTED verdict was WRONG (valid only for non-HiCache)

CORRECTION (swa-auditor, after features-lead's counter-code review):
The earlier proof concluded 'when COW is staged, best_match_node ==
best_match_device_node == req.last_node'. That equality holds only for
non-HiCache runs (validators == device validators). In HiCache mode the FULL
component's relaxed validator (components/full.py:113): `value is not None or
node.backuped` passes at an EVICTED-but-host-backed node. The tree walk
(unified_tree_core match) can therefore yield:

  best_match_device_node = A (deepest node with ALL device values, FULL evicted
      at E below A), strictly ANCESTOR of
  best_match_node = D (relaxed pass through the backuped gap; mamba DEVICE
      value at D is non-None, e.g. D's own FULL value still on device).

Stage check in finalize_match_result_in_cache (components/mamba.py:193) relies
ONLY on D's mamba device value being non-None -> the COW IS staged from D.
Admission lock binds to req.last_node = match_result.last_device_node = A
(schedule_policy.py:1607 via common.set_prefix_state last_device_node), and
acquire_component_lock (components/mamba.py:444) locks ONLY that single node;
D's mamba slot stays lock_ref == 0 in the mamba LRU.

Window: get_new_batch_prefill loops over waiting reqs; each later request's
finalize/alloc can call evict_for_alloc(mamba_num=1) (dst allocs or KV
pressure); D's value is freed via evict_component -> mamba_allocator.free and
reused/donated; req A's deferred COW (_execute_mamba_deferred_ops at the batch
forward) then copies a rewritten slot into a fresh active slot -> silent
cross-request mamba-state leak / garbled output.

Production reachability: HiCache on most deployments (page_first_direct,
host chunks extending past device); mamba on Kimi K3. Whether K3 runs the
unified radix + hierarchical combo in production determines blast radius
(needs prod-config confirmation). Non-HiCache unified cache: not reachable
(validators equal device validators, equality proof stands).

Fix direction (mamba-auditor's sketch stands): in
finalize_match_result_in_cache, inc_lock_ref(best_match_node, MAMBA-only) when
staging a COW, dec after _collect_deferred_mamba_cow_and_clear /
_execute_mamba_deferred_ops; or copy eagerly at match time (slot pair
memcpy, no deferral).

Status: CANDIDATE -> OPEN (mechanism confirmed at code level; PoC harness
pending: hicache page_first_direct eviction-gap path with mamba checkpoints
only below the gap). Independent re-verification requested of F-1's
opens.

NOTE (04:29, features-lead adjudication in FEATURES_FINDINGS_lead.md on
agent/features-lead): F-1 confirmed via an even STRONGER mechanism than the
evicted-gap path above: per-component independent eviction lets node N lose
its FULL device value to host (host_value set, value None) while MAMBA on
the SAME node survives device-side. Relaxed validators pass at N
(best_match_node=N) while the FULL device validator fails
(best_match_device_node=proper ancestor of N) — no second gap node needed.
Our HiCache configs (page_first_direct, write_through) evict FULL-to-host
routinely, making this the production-realizable shape; the gap-path variant
is a strict special case of it. Plus the undisputed non-prefill windows
(retract-before-extend; dec between boundary checkpoints; donation-reuse).



## Mamba #8 deep-dive ledger (continuing)

### Ping-pong / donate / extra_buffer accounting — CLEAN at HEAD
- donate_mamba_ping_pong_slot: atomic swap (old slot out, new_slot in) + device
  mapping sync in set_mamba_ping_pong_slot. Ownership: one holder per slot at
  every step (int8_ckpt path: quantize + free(src_active) once; tree owns the
  int8 slot; freed-once via _free_mamba_value at evict).
- Finished donate: keep_idx chosen once in prepare_for_caching_req, recorded
  into insert_params.mamba_keep_idx; cleanup_after_caching_req reuses the
  recorded index (cannot pick another) -> free_mamba_cache frees exactly the
  other slots. spec (buffer size 1) keeps only slot -> frees empty slice.
- Abort-after-retract: holds_mamba guard (scheduler:5560) + reset_for_retract
  nulling + mark_kv_released clearing component_evicted_seqlens make the
  disagg-decode checkpoint=True abort branch is_kv_released-consistent.
- Lazy-mode alloc failures (mamba_lazy_prealloc_at_boundary,
  mamba_lazy_spec_prepare): on failure the forward writes in-place into the
  keep slot and req.kv.mamba_last_track_seqlen is poisoned to None, which
  gates all later donates (cache_len None -> None checkpoint). No -1 reaches
  free paths: every free guards '!= -1'.
- free_mamba_cache resets ALL ping-pong cursors after freeing - the stale-ref
  reuse bug class is guarded ('double free? mamba_index is None' assert).
### ReplaySSM write_pos vs donation — CONSISTENT for extra_buffer production path;
  no_buffer mid-prefill checkpoint relies on 'ring empty at checkpoint' (debug
  assert in copy_from), production K3 runs extra_buffer whose track slots are
  fully-scattered at flush boundaries by construction; load_back supersedes
  replay cursors (finalize_allocation zeroes write_pos for the req slot).
### Candidates remaining from mamba-auditor worth breaking further:
- -1 sink writes: caller-side guard evidence assembled; all free/copy sites
  guard. open: any clear_slots caller passing raw indices (not seen).

### F-1 FIX review residuals (dsa-auditor patch agent/dsa-auditor @ 91e3f20)
CANDIDATE [leak-of-pin / availability]: dsa's clear_mamba_cow catch-all sits
INSIDE HybridReqToTokenPool.free_mamba_cache's extra_buffer branch
(memory_pool.py:1886). release_kv_cache's early-alloc-KV-failure branch
(common.py:398-408: `if not req.kv.holds_kv: ... mamba_allocator.free(mamba_pool_idx); return`)
bypasses free_mamba_cache entirely — a request that staged the COW (lock set
on the src node at match), allocated its mamba dst, then fails KV admission
and later dies via this branch leaks the SOURCE NODE PIN permanently
(lock_ref stays elevated → node never evictable in the mamba LRU —
availability rot; no corruption). Reported to dsa (message stream) with the
suggested seam (clear in release_kv_cache on every mamba-capable exit — call
is None-guarded). Spec harness attack deployed to pin it.

### R-1 fix (board #69, merged-shape note)
dsa-auditor @279eb8a: pad-verify last_row_cap=speculative_num_draft_tokens wired
into deepseek_v4:1279, trtllm_mha:1149, flashattention:3122. VERSION NOTE
(features-lead directive): trtllm_mla:1559 ALREADY passed cap=draft_token_num —
only 3 of 4 consumers were live-risk. Layout semantics: fold=min(leftover,
cap-len), remainder ghosted via total_verify_tokens=None (pre-existing cap=N
route); triton sentinel -1 matches policy. 6/6 CPU suite green; pre-fix failure
is mechanical (kwarg TypeError on 5/6 — semantic anchor = the legacy-compat
test), independent witness PoC assigned to mamba-poc-author as post-merge task.
Residual watched: sum(padded)<graph leaves static q-buffer tail unassigned for
dense consumers; ghost-neutrality is asserted by the total_verify_tokens=None
flow, pending witness-PoC verification.

### F-1 PoC DELIVERED (board #60 complete)
Author: mamba-poc-author @4e24639 (branch agent/mamba-poc-author);
test/registered/unit/mem_cache/test_mamba_cow_evict_window_poc.py (330 lines).
Architecture: REAL UnifiedRadixCache end-to-end via shared build_fixture —
FULL+MAMBA components, real MambaSlotAllocator, real state buffers. Sole seam:
mock.patch of get_device -> "cpu" (the builder's single non-portable call);
no sgl_kernel dependency, runnable under the mandated queue command (verified
in the fresh 05:17 shared env by swa-auditor).

Mechanism: mamba_cache_chunk_size=8 + page_size=1 puts a mamba checkpoint at
depth 8 with FULL at depth 16 -> COW source = proper ancestor of the FULL
admission-lock anchor = the HiCache divergent-hit shape reproduced w/o HiCache
(production's relaxed FULL validator yields the same anchor divergence; the
stage->collect window hole is identical). Triple content-tracking assertions:
source survival, slot re-let, content snapshot vs new-owner write (fill 999).
Test 3 pins clear-balance (no permanent freeze after clear_mamba_cow).

Pre-fix verbatim (origin/main, fresh env, exact queue cmd, 2 failed/1 passed):
  staging_pins_the_source_node: AssertionError: 0 not >= 1 : F-1: a deferred
    mamba COW is staged on a node whose mamba component is unlocked
    (lock_ref == 0) -- evictable while the forward-stream copy is pending
  evict_window_rewrites_staged_source_content: AssertionError: unexpectedly
    None : the cache evicted the staged node; slot freed while the
    forward-stream copy is still pending (arena deletion + slot-free witness)

Post-fix: 3 passed in 18.93s against agent/dsa-auditor @3832b59 (c9b4d15
full-hierarchy lock + c33f4e3 catch-all), re-verified by swa-auditor in a
fresh worktree.

VERSION SWEEP (author): F-1 staging in ALL production versions — v0.5.14
(unified_cache_components/mamba_component.py:112), v0.5.15.post1, v0.5.17
(same pattern), v0.5.20 (unified_cache/components/mamba.py:214, no lock).
Kimi K3 base era (87dd4596/bd4828e2) pre-unified-cache; staging existed in
the mamba_radix_cache.py era shape — LIKELY AFFECTED; vendor-branch ref not
fetchable from origin, flagged for the coverage report / fix-port owner.

Merge gating: PoC ships AFTER dsa's #17 lands on dev (asserts post-fix
property).

### R-1 WITNESS PoC (board #72, approved)
Branch agent/mamba-poc-author-r1 @6f3c3e0 (based on dev ac0f192 post-F-1).
File test/registered/unit/speculative/test_ragged_verify_last_row_inflation.py.
REAL torso: RaggedVerifyLayout.from_verify_lens -> padded_to_bucket (cap=None
full-coverage shape) + BuildQoIndptr + build_ragged_target_verify_geometry;
no mocks/fixtures; adaptive via inspect.signature('last_row_cap').

G1 VERIFIED (content-observation, grade as #66); independently re-run by
swa-auditor: pre-fix lens [2,2,8] (fold of 4 over draft budget 4), overrun 4,
sentinel stale slot ids [591000-591003] enter the last real row's verify
window. Post-fix (dsa @3832b59 carrying #69, exact queue cmd): 3 passed.

G2 REFUTED (numerics-corrupting grade), executable evidence + independent
verification: accept_index tree-bounded per request (reject_sampling
speculative_sampling_classic_kernel walks RetriveIndex with NUM_SLOTS =
num_draft_tokens; eagle_utils accept entry argmax reshapes
[bs, draft_token_num]); gather_spec_extras row-indexed (req_pool_indices);
KV writes bounded by live total_verify_tokens (extras have no out_cache_loc).
Extras' outputs structurally discarded -> grade stays G1; caveat (future
flat-width consumer reopens G2) pinned in the witness docstring.

MERGE PAUSE: branch parks until dsa's #69 merges to dev (durable-witness
ordering); poc-author messages swa-auditor to drive the merge then.

### K3 VENDOR-BRANCH DIAGNOSIS (lead directive #1554, 2026-10-16)
The internal Kimi K3 branch point (base 87dd4596 / bd4828e2, tagged
'about v0.5.16' in fleet metadata) is NOT reachable as a ref from the public
sgl-project/sglang remote — git fetch confirms neither SHA exists there.
Per lead: document as KNOWN-UNVERIFIABLE; record the evidence shape for the
production hotfix channel instead.

F-1 (mamba COW source lock-gap) on the K3-era shape: F-1 exists in v0.5.14
v0.5.15.post1 v0.5.17 v0.5.20 (staging in
unified_cache/components/mamba.py; pre-unified era
unified_cache_components/mamba_component.py:112 in v0.5.14). The K3-era
mamba checkpointing machinery predates the modern unified_cache tree: the
staging site lives in red-black/hi_mamba-era code (mamba_radix_cache.py era).
K3 runs prod = hybrid mamba + HiCache + PD-disagg (extra_buffer,
linear-replayssm, hicache mooncake, page 64, fp8e4m3) — the divergent-hit
precondition (FULL evicted-to-host below the mamba checkpoint node) maps
directly. Adjudication: LIKELY AFFECTED. Fixing path for production: hotfix
channel via vendor maintainers, NOT this repo's dev/main (file proprietary
branch SHA 87dd4596/bd4828e2, affected-file: file_path-at-K3: report
findings in KB/hotfix tracker with the F-1 PoC + verdict + dsa dev fix
@mamba_cow.py as the reference patch). Same shape applies to the F-4 SWA
alloc_decode leak (SWA fix @28acaf2 — leak in swa.py alloc_decode paths:
K3-era status same-code likely affected — swa_memory_pool internals were
stable across those versions; hotfix-channel ping recommended too) and the
spec/iverted claim queue lines where flagged (spec VERSION NOTE in
FEATURES_FINDINGS_spec.md route to .15/.16 hotfix checks).

HOTFIX-CHANNEL DATA PACKAGE per finding: {branch-point SHA, era file map
with the line numbers as of the affected production version, PoC evidence
link = F-1 PoC SHA 4e24639/4e9842c + verbatim pre-fix assertions, verdict
fixed-on-dev SHA, suggested minimal patch (dsa mamba_cow.py full-hierarchy
lock + clear funnel)}.
