# SWA KV-cache memory-safety audit — swa-auditor

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

## In progress / open
- Awaiting features-lead review of F-4 fix (board #26).
- Open delegated item for pools-lead/radix-lead: unified tree-action level
  proof that FreeDeviceKVFullOnly is only emitted after the SWA tier is
  tombstoned (my trace covered the allocator's external callers; the tree
  action generation is unified-cache internals).
