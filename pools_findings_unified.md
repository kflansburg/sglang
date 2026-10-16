# unified-auditor findings

Scope: unified_memory_pool.py, allocator/unified_sub_pool.py, unified_mamba.py, unified_hybrid_swa.py,
unified_draft_pool.py, kv_cache_configurator.py, allocation.py, allocation_sizing.py, kv_index_translator.py,
kv_loc_plan.py, base_swa_memory_pool.py, swa_memory_pool.py, deepseek_v4_memory_pool.py (pool mechanics),
allocator/mamba.py, mamba_slot_fused.py, mamba_checkpoint_pool.py.

Production config lens: --page-size 64, fp8_e4m3/bf16, chunked prefill 12288-32768, DP attention 1-8,
SWA + hybrid mamba (Kimi K3: --mamba-radix-cache-strategy extra_buffer), disaggregated prefill/decode.

## Findings (incremental)

## Coverage status (2026-10-16)
- unified_sub_pool.py: READ END-TO-END (2680 lines). Verdict: engineered sound by construction;
  defense gaps F0 (hardened) + F1 (contract, unenforced).
- unified_memory_pool.py: READ END-TO-END (2376): init factories sizing additive-exact (full+swa+mamba
  byte sums, bs1 floors exact, _reserved_floor_bytes single source, anchors all 0 asserted);
  views/stride math validated; UnifiedMambaSlotAllocator v2p gather propagates -1 as sentinel
  (translate(v)=-1 -> consumers must skip; mamba-auditor F-1 caller-side).
- allocator/unified_hybrid_swa.py: READ END-TO-END (1654): 2-END joint K1/K2/K3 capacity math sound
  (under-count = denial, never over-alloc); tail-prealloc page accounting consistent at page 64;
  ensure_capacity flush ladder + float policy verified; tri-pool _fits_page_demand ordering correct.
- allocator/unified_mamba.py: READ FULLY. DCP widening consistent; free paths defer to F0 hardening.
- unified_draft_pool.py: READ FULLY — sound, fail-loud on unsupported ops.
- swa_memory_pool.py (composite SWAKVPool): READ FULLY — mapping mechanism sound; latent
  move_kv_cache -1 scenario requires caller violation; maybe_detect_oob is env-gated in prod.
- mamba_checkpoint_pool.py / allocator/mamba.py / mamba_slot_fused.py: READ FULLY (pool mechanics);
  free-list no-dedup + fused-kernel trusted bounds are caller-contract (mamba-auditor); int8 store
  index math exact; estimate_mem_usage_bytes == real layout incl slot-0 reserve.
- kv_cache_configurator.py: configurator-auditor full verdict (no runtime bug; 2 LOW budget items —
  board #47); I added the allocplan-seam int32 DCP assert (_assert_dcp_widened_ids_fit_int32) with 4
  tests, committed.
- allocation.py / kv_index_translator.py / kv_loc_plan.py / allocation_sizing.py: allocplan-auditor
  full verdict — clean with evidence; 1 latent int32 class -> my config-time assert above.
- deepseek_v4 pool mechanics: dsv4-pool-auditor full verdict (F1 fixed+tested, F2 PD-trust logged,
  F3 hisparse data_ptrs -> #hicache); board #24 approved by me.
- pools_fuzz added by this team: model-checker (pair+tri, ps=4/64), stale-free hardening suite,
  DCP int32 bound suite, dsv41 window layout suite, mamba slot suite, kernel hardening suite
  (pools-lead), alloc_extend boundary sweep (pools-lead), allocplan 20k-trial pure fuzz.



### F0. Known-class verification status (upstream HEAD, 2026-10-16)

- "unified sub-pool double-free and -1 sink OOB write": PARTIALLY MITIGATED by design.
  - Base EAGER free (`unified_sub_pool.py:1290-1310`) has stale-slot assert on `(freed_p_pages < 0).any()`.
  - `_free_lazy` + `free_unbind_inplace_kernel` (`python/sglang/kernels/ops/memory/virtual_slot.py:101-127`):
    on a double-freed virtual page, v2p reads -1, then `tl.store(p2v_ptr + p, -1)` = **raw write at
    p2v_ptr - 8 bytes (BEFORE the table)** — kernel has NO bounds guard. Safety rests entirely on the
    caller contract "must not double-free". A GPU OOB write of 8 bytes here corrupts whatever tensor
    the allocator placed below p2v (undefined).
  - SWA composite (`unified_hybrid_swa.py:566-712`) filters swa frees with tombstone mask
    `swa_v2p_pages > 0` (strictly: -1 tombstoned and page-0 padding-sink both skipped) — a double-free
    on the swa side is filtered at the composite. Full side relies on "full owns the ids".
  - `FloatMultiEndedAllocator.free` (`unified_sub_pool.py:2352+`): NO stale-slot check even in a debug
    mode; on double-free: `freed_p_pages` contains -1 -> `physical_to_virtual.index_fill_(0, p2v_with_-1)`
    (torch index_fill_ with negative index => WRAPS: writes p2v[-1] == p2v[num_pages-1]: a LIVE page
    belonging to the high END pool gets tombstoned) AND -1 joins `_free_phys_pages`, from which
    `take_physical_pages` later hands out physical page **-1**, and `bind` then writes
    `tl.store(p2v_ptr - 1)` (OOB write) + v2p[v] = -1 (translate -> negative kv loc -> OOB in
    set_kv_buffer). Severity gated on reachability of a double-free into a float pool: need to find
    a caller. IN PROGRESS.

### Verified-clean / mechanism checks (coverage log)

- UnifiedKVPool construction (unified_memory_pool.py:391-600): all sub-pool anchors are 0 (asserted);
  max_slots = total_bytes // entry_bytes per pool; page-0 sink reservation via `_reserved_floor_bytes`
  (page-aware pools charged page_size*entry, mamba charged once); min_slot_index=ceil(reserved/entry).
  build_dense_views asserts computed end within raw buffer; layout validates overlap + alignment. SOUND.
- copy_pages / zero_pages Triton kernels (kernels/ops/kvcache/): NO bounds guards on page ids; the
  `raw[:num_pages*page_bytes]` slice is only a base pointer for the kernel. Bad page id from an
  allocator = raw OOB or cross-sub-pool corruption (anchors are all 0). RECOMMEND: add
  SGLANG_ENABLE_ASYNC_ASSERT guards (max id check) to copy_pages/zero_pages as cheap tripwires.
- UnifiedMHA/MLA move_kv_cache derives pages from token ids via view(-1,ps)[:,0]//ps — a
  non-page-aligned caller would silently drop tokens; current callers (both compaction paths) always
  expand whole pages. Fragile-by-design, not a reachable bug now.
- lazy take_physical: drains free-list then extends watermark; _extend_watermark checks BOTH index
  space and chain frontier, both grow directions. Eager _take_physical_eager checks ONLY index space;
  frontier safety delegated to callers' available_size() gate — documented, but any future gate-less
  caller = silent cross-pool corruption. Suggest debug assert in _take_physical_eager vs
  _current_gap_bytes(). VERDICT: sound today.
- Memoized capacity views: _CapacityField descriptor (rebind-only) + chain epoch sums +
  _capacity_memo_violations debug checks. SOUND.
- Compaction race safety: disagg/HiCache move gates require lazy_compaction and cover BOTH composite
  ends via _move_gate_targets; _flush urgent settles in-flight forward event; write-set materialized
  from out_cache_loc_virtual (DCP-widened ids collapsed); pending-reuse read-race gating. SOUND.
- moves_blocked() also True while _pending_hicache_load_pages>0; cancel_physical_reservation asserts
  count consistency. SOUND.
- FloatMultiEndedAllocator.make_room / _relocate_to_positions: capacity pre-check before leapfrog;
  src/dst disjointness enforced (batched disjoint, ordered singletons on overlap shift); span
  reconstruction preserves interior holes as free list. Math checked. SOUND.
- unified_mamba.py composite DCP widening consistent (composite page widened, sub-pool physical page
  shard-local; num widened virtual pages == local page count; _release_page_reps uses widened
  page_size for virtual derivation). SOUND.
- unified_draft_pool.py: fused draft binds host sub-pool views, lanes distinct/in-range; move/
  transfer/offload raise NotImplementedError (fail-loud). SOUND.
- mamba_slot_fused.py fused kernels: trusted slot indices; a negative index is a raw OOB write
  (triton) while the non-fused path silently wraps (corrupts an innocent slot). Callers: radix COW
  and spec paths — caller validation owned by mamba-auditor.
- MambaSlotAllocator: slot 0 reserved via arange(1,size+1); free() no dedup — caller double-free =>
  slot handed to two requests (state aliasing, silent). Caller lifecycle owned by mamba-auditor.
- free_unbind_inplace design: caller uniqueness contract; see F0 for the OOB write consequence.
- SWAKVPool (non-unified, swa_memory_pool.py): full_to_swa_index_mapping = zeros(size+ps) + [-1]
  sentinel (allocator/swa.py:103); unmapped full loc moves route to slot 0 sink (safe); -1 loc maps
  to sentinel -1... then MHA move_kv_cache receives -1: maybe_detect_oob probes exist (env-gated).
  If SGLANG_ENABLE_ASYNC_ASSERT off (prod default), a -1 tgt/src would wrap to the LAST swa slot
  in torch indexing = silent corruption. Reachability requires a -1 full loc, which all current
  callers prevent by construction; flagging as latent only.
- alloc_physical/free_physical (HiCache load reservations): bound_mask filters v2p clear; physical
  pages always cat'ed into holes — a duplicate cancel would duplicate the hole entry (same F0 pattern).

### F0-update (2026-10-16): EMPIRICAL PROOF + hardening patch landed on agent/unified-auditor

Empirical (CPU, lazy END pool via build_swa_pool(page_size=4)):
```
tok = alloc.alloc(16); alloc.free(tok.clone()); alloc.free(tok)
=> holes = [..., -1, -1, -1, -1]; live_page_count = -4; _byte_accounting_violations() == []
```
=> **pre-patch, a caller double-free is FULLY SILENT and instantly corrupts** (holes with -1 get
handed out later -> OOB writes in bind/copy/zero kernels; live counter underflows the leak
invariant and capacity views). Patch skeleton (message from pools-lead) implemented + tested:

1. `free_unbind_inplace_kernel` + `bind_inplace_kernel`: store guards `p >= 0` (+ CPU reference
   branch guards) -> stale frees/binds can no longer write OOB or wrap to the last table slot.
2. `_filter_stale_frees` on `_free_lazy` + `Float.free`: SGLANG_DEBUG_MEMORY_POOL => sync, filter,
   count, raise like EAGER; prod default => no host sync (no-sync contract asserted by
   test_unified_free_no_host_sync — empirically hit when I synced unconditionally), plus
   SGLANG_ENABLE_ASYNC_ASSERT tripwire. Remaining prod-drift from a caller bug is caught by (3).
3. `_byte_accounting_violations` (END lazy + FLOAT): NEW value checks — duplicate/out-of-range
   holes and live_page_count < 0 (count identity provably misses double-frees).
4. `_stats_n_stale_frees` counter + error log.
Regression tests: test/srt/pools_fuzz/test_unified_stale_free_hardening.py (6 tests) +
test_unified_alloc_fuzz.py (model checker over 2-end and tri-pool at ps=4/64, ~1500 random ops,
invariants: no double-alloc, v2p/p2v bijection, physical census, byte accounting). ALL PASS;
86 existing unified CPU tests PASS (incl. the no-host-sync suite).
OPEN: GPU-side equivalent test (guards are in the Triton kernels; CPU reference covers semantics).
OPEN: caller-reachability of a double-free — radix-sched investigating (cached as gate to
upstreaming).

### F1. `free_group_end` duplicate pages (documented contract, unenforced)
`MultiEndedAllocator.free_group_end` (unified_sub_pool.py:1980): `reps = torch.cat(pending)` then
`free(reps, _pages=reps // page_size)` — if two segments in one group touch the same page (caller bug),
the SAME virtual page enters `free_virtual_ids` twice -> same virtual page allocated to two requests
-> cross-request KV aliasing (silent corruption, no OOB). Contract comment at free_segment says
"a page must be freed by only one call per group" — not asserted. Callers: radix/scheduler paths; flag
to radix-lead to confirm the invariant is maintained at cache_finished/unfinished with page 64.
