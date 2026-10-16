# Pools/Allocators Audit — Master Findings (pools-lead)

Scope: `python/sglang/srt/mem_cache/` — allocator/*, memory_pool.py, swa/unified pools, configurator, sizing, kv_loc_plan, kv_index_translator, kv_vmm_backing, layout/, events.py, device_pool_info.py, common.py. Production config: page-size 64, fp8_e4m3 (mostly) / bf16 KV, chunked prefill 12k–32k, TP 1–8, DP attention, disaggregated prefill/decode.

Status legend: CONFIRMED (repro or exact counterexample), CANDIDATE (code argument, needs verification), TRACE-OK (checked the risky pattern, guarded), CLEAN (read fully, nothing found).

---

## Lead's own pass

### allocator/paged.py — read fully (421 lines)
- Capacity checks on alloc/alloc_extend/alloc_decode: **TRACE-OK**.
  - `alloc_extend` uses strict upper bound `extend_num_tokens // ps + bs + 1`, and when that cannot prove depth, computes exact `get_num_new_pages` before launch. The Triton kernel reads `free_page_ptr` up to `sum(num_new_pages)-1` only. The kernel clamps (`torch.clamp(page_idx, max=numel-1)`) in the naive path can only trigger on dead lanes (verified the owner/phase masks).
  - `alloc_decode` capacity check: `num_new_pages = get_num_new_pages(...decode=True)` compared against merged free list. Guards the Triton read `free_page_ptr[new_page_start_loc]` since `new_page_start_loc <= num_new_pages-1`. OK.
- `free(free_index)` dedups token indices to pages via `torch.unique(free_index // page_size)`. **No dedup in `free_page_ids` / `free_segment`** — contract: no page twice. Only `debug_mode` asserts duplicates; in production a caller bug → silent double-free → same page handed to two requests (cross-request corruption). Searched callers of `free_page_ids`/`free_segment`: paged.py internal, swa.py (`_free_swa_pages` after mapping read, guarded by invariant), allocator/unified_sub_pool.py (unified-auditor's). No concrete double-free caller found yet. **CANDIDATE-class risk, no concrete bug.**
- `free_pages` tensor slicing aliasing: `out_pages = free_pages[:n]` view, `free_pages = free_pages[n:]` view; `free()`/`_release_page_ids` always `torch.cat` → new tensor. No in-place mutation of the views anywhere (checked `torch.sort` assigns a new tensor; `merge_and_sort_free` reassigns). **TRACE-OK.**
- `clear()`: free_pages = arange(1, num_pages+1) — reserves page 0 as the sink for padded writes. `available_size` = (free+staged)*ps. **OK.**
- Note: `free_group_end` in paged: `_release_page_ids(cat(free_page_ids_group))` then base class frees `free_group` token indices. Both dedup paths (unique in free()) — a page freed by id AND by token index in the same group is double-freed silently (debug-only check after). Callers mixing both APIs in one group is the hazard; searching for such a caller → none found in scheduler paths so far (free_kv_row_segments uses segments API consistently).

### allocator/token.py — read fully (96 lines)
- `release_pages` staging + merge on alloc shortfall; `available_size` counts both lists. `alloc` merges only when `need_sort` and shortfall. Conservation OK. **CLEAN** (only reachable at page_size==1, not our prod).

### allocator/swa.py — read fully (872 lines)
- `full_to_swa_index_mapping = zeros(size + page_size) + [-1]`: index at `size+page_size` is the -1 sink for `last_loc=-1` (no prefix). Zeros (not -1!) for unmapped full slots — invariant `_SWA_PEER_MAPPED` guards `free_swa` on unmapped slots. Mapping cleared to 0 on free; allocated slots written before use. **TRACE-OK** for the -1 path.
- `alloc_extend` for SWA pair: `new_pages_available(num_new_pages, num_new_pages)` checks both pools first, then both sub-allocs asserted non-None. On assert-failure (python -O removes asserts → None propagates to caller `alloc_paged_token_slots_extend` → RuntimeError, but full pages ALREADY popped from full free list → **leak** of full pages on that failure path. Window: swa pool check passed pre-check but its `alloc_extend` still returned None (can't happen — same bound pre-checked by new_pages_available using identical num_new_pages for both pools; symmetric sub-allocator call). → window is empty. **TRACE-OK** (pre-check uses the exact same num_new_pages).
- `alloc_extend_swa_tail` (bs==1): num_swa_pages = ceil(swa_tail_len/ps) pre-checked; full checked too. swa sub-call `alloc_extend(prefix=0, seq=swa_tail_len, last=-1)` needs ceil(swa_tail_len/ps) pages — exactly num_swa_pages. OK. Mapping written for tail; `clear_full_to_swa_mapping` for non-tail prefix. **TRACE-OK.**
- `free_swa` (paged, arbitrary set): expands to full pages, reads mapping, clears mapping, defers release of the gathered swa_indices (filtered >0) to group end. Expansion picks up slots never allocated → mapping reads 0 → filtered by `> 0`. **TRACE-OK.**
- `free(full)` = free_swa + free_full; `free_full` invariant `_SWA_PEER_RELEASED` (Bucket.GUARD — runtime-enforced?) — checked `expect()` semantics: Invariant with Bucket.GUARD raises/warns depending on env config; FATAL_UNCONTAINABLE always fatal. So in default prod, `free_full` on a mapped slot logs/raises a guard instead of double-freeing. OKish — the protection exists.
- `free_group_end` ordering: swa_page_ids released BEFORE full group end. A cache action in the same scheduler step that re-allocs SWA pages could then install a new mapping for a full index whose old mapping this group already cleared + released — that's the design (ownership resolved at enqueue). The stale-mapping hazard would be a scheduler-side alias; SWA tree policy is features-lead's scope. Flagged to features-lead.
- `PureSWATokenToKVPoolAllocator` (page 1 only, not prod): free filters `> 0`. OK.

### allocator/page_interleave.py — read fully (406 lines)
- Class-sharded logical pages (DCP-style widened page ids: logical page l → class l%N, physical l//N). free() splits by owner class with unique dedup. `available_size` = min-class floor × N × ps, admission reserved per chain — closed-form pops per class pre-simulated. **No memory bug found; CLEAN-ish** (needs the fuzz to confirm arithmetic, handed to alloc-auditor as P2).
- `free_segment` degrades to plain free() — note it ignores the fixed-shape fast path; safe.
- `_release_page_ids` raises NotImplementedError — base-class free paths that would use stock lists are fenced. But: base `free_group_end` calls `self._release_page_ids(...)` ONLY for paged's free_page_ids_group; page_interleave overrides free() fully — but does its `free()` inside a group append to `self.free_group` (ok) — base free_group_end then calls `self.free(cat(pending))` → PageInterleave free with free_group=None → works. OK.

### allocation.py — read alloc paths (841 lines)
- `alloc_for_extend` / `alloc_for_decode`: `alloc_aux_to_lengths` failure → free(out_cache_loc) + raise. out_cache_loc here is exactly what was just allocated; freeing it whole is correct (paged free dedups pages). **TRACE-OK.** But note: for `alloc_for_extend` paged path, out_cache_loc len == extend_num_tokens with token indices; freeing all may release pages that contain the *prefix tail page* (phase-1 tokens `last_loc+1+local` live on the page ALREADY owned by the request's prefix!). num1 phase-1 tokens: prefix partial page. If seq extends the last prefix partial page, out_cache_loc contains indices on a page the request already owned. Freeing the whole out_cache_loc on the error path then frees the prefix's last partial page while the prefix is still in req_to_token → **CANDIDATE P1: partial-page double-accounting on the alloc_aux_to_lengths error path**; same in decode? Decode: last pages: `out_indices` for decode is either last_loc+1 (same page as existing seq = owned already!) or new page start. Freeing out_cache_loc on decode error path frees pages containing last_loc+1 — the request's CURRENT last page → **double-accounting when a request continues later**; but this error path raises → batch aborts → likely request-level cleanup. IS this reachable? alloc_aux_to_lengths raises when? need to check ReqToTokenPool.alloc_aux_to_lengths — aux buffers sized per req; failure when seq exceeds preallocated aux capacity. If it raises after some rows written: free() returns pages including possibly-shared with other reqs? No — alloc gave fresh pages; only phase-1 (extend) or last-page (decode) tokens alias the request's own pre-existing pages. Free on this path: `token_to_kv_pool_allocator.free(out_cache_loc)` → unique(free_index//ps) → frees the prefix's tail page → page goes back to free list while request still references it in req_to_token → next request allocs it → **cross-request corruption — IF the error path raises with the request surviving.** Need to trace who calls alloc_for_extend / what happens post-raise (scheduler catches? aborts batch?). Candidates for radix-lead (abort path) — pass on.
  → VERDICT: `_aux_cache` is None unless a MiniCPM sparse aux cache is attached — MiniCPM is NOT on our production surface, and with None the call can't raise. The double-accounting window (phase-1/decode tail-page indices inside out_cache_loc being freed onto pages still owned by the request) is **latent and real for MiniCPM-Q/SparseV** but out of our scope. Marked TRACE-OK (prod), latent-bug noted for upstream.

### kv_index_translator.py — read fully (609 lines)
- Glue over allocator-owned v2p tables (unified pool translator tables). The index-math risk lives in `build_kv_read_table` kernels + allocator tables (unified-auditor). `pad_with_sink` sink = id 0 = slot 0 reserved everywhere. OK.
- `build_iteration_table`: `width = min(max(-(-max_seq // ps), 1), row_pages)` — max_seq includes read_extent; capped by row_pages. `out[bs:] ` zeroed when rows > bs. rows < table in previous-copy branch: `out = zeros((rows, previous.ids.shape[1]))` then `out[: previous.ids.shape[0]].copy_(previous.ids)` — if previous has MORE rows than rows... `out[:p] = p ids` with p > rows → shape mismatch error (not memory). Edge: `previous.ids.shape[0] > rows` can't happen since rebuild only happens when `table.ids.shape[0] < rows`. OK.
- `copy_page_table`: stale columns past width kept — comment claims kernels never read past lengths; width = min(out.shape[1], table.ids.shape[1]). Only copies table's rows. **TRACE-OK.**

### kv_loc_plan.py — read fully (348 lines)
- Plan translation + bind; `_window_in` calls `space.write(write_virtual)` = allocator translate (unified-auditor scope). `_cols` with tensor index: `torch.where(cols >= 0, ids[cols.clamp(0)], 0)` — clamp + where, no OOB. sink 0. OK. **CLEAN.**

### kv_vmm_backing.py — read fully (278 lines)
- VMM arena: bump arena within a VA reservation; `commit_range` monotonic per offset, aligned, reserved-bound checked. `_back_spans` validates every span ≤ reserved_span before any commit (two-phase) → no partial-commit state. `finalize` bounds `final ≤ reserved`. On finalize **underestimate** → later KV writes beyond backed span → GPU fault, not silent corruption — loud fail. **CLEAN** (VMM only used when configured).
- `close()` clears tensors/specs before arena close; commit_range raises after close. OK.

### events.py — read fully: event recording only, hashing, no buffer ownership. **CLEAN.**
### device_pool_info.py — read: msgspec structs + validate. **CLEAN.**
### utils.py — read: hash helpers, eviction strategy factory. `split_node_hash_value` uses `split_len // page_size` — pure list slicing, safe. **CLEAN.**
### allocation_sizing.py — read: pure integer sizing helpers; `page_aligned_decode_alloc_lens` aligns committed+reserve up to page. Comment "(unaligned tails leak at ps>1)" documents rounding. No defect found in formulas themselves. **CLEAN** (formulas consumed by configurator → unified-auditor crosschecks values).
### base_swa_memory_pool.py — read: ABC. **CLEAN.**

## Files delegated (coverage assignments)
- alloc-auditor: allocator/{hisparse,page_interleave detail,swa kernels} + fuzz harness (paged alloc/extend/decode/free/merge vs model) — RUNNING
- kvpool-auditor: memory_pool.py (6188 lines: MHATokenToKVPool, MLATokenToKVPool, fp8 scale buffers, TP head math, ReqToTokenPool free/alloc slots, mamba ping-pong tracking), layout/, cpp_utils/native_hash, kv_cache_dtype — RUNNING
- unified-auditor: unified_memory_pool.py, allocator/unified_{sub_pool,mamba,hybrid_swa}.py, unified_draft_pool, kv_cache_configurator.py, mamba pools, deepseek_v4_memory_pool.py, swa_memory_pool.py (mechanics) — RUNNING

### sparsity/ — read sparse_coordinator.py, deepseek_dsa.py, backend_adaptor.py (lead)
- Thin glue. SparseCoordinator trackers sized by req pool capacity, keyed by req_pool_idx; register/clear on request begin/end — stale-state risk if abort misses on_request_end → **radix-lead territory** (scheduler abort paths), flagged.
- FlashAttentionAdaptor.adapt_for_attn_metadata: for a sparse-masked request with valid_lengths==0, `sparse_seq_lens = -(diff) <= 0` → negative/zero cache_seqlens_int32. Quest/PQCache path only — NOT production (our DSA uses DeepSeekDSAAlgorithm + dedicated trtllm backends; DSABackendAdaptor is an unimplemented `pass`). Out of scope; noted.

## Version check queue (for version-triage)
- (pending confirmed bugs)

## Harness results (pools-lead, CPU, torch 2.14.1)
- test/srt/pools_fuzz/test_paged_allocator_fuzz.py: conservation + no-double-alloc + drain, 20 configs (need_sort x seeds), 400 mixed ops each: ALL PASS — stock allocator sound under contract-abiding callers.
- ADVERSARIAL-1 (CONFIRMED consequence statement, production debug=off): caller double-free of one page via free() → free list holds the page twice (21 entries/20 unique) → subsequent two allocs hand page 1 to two requests → cross-request KV aliasing. No assert fires with SGLANG_DEBUG_MEMORY_POOL unset. This is the concrete production consequence motivating caller-side invariant fixes (radix-lead's abort/retract hunt) + an allocator hardening option.

## VERIFIED MECHANISM (pools-lead, code-level): F0 chain fully real
unified_sub_pool.py Float free (L2352-2395) + _free_lazy (L1339-1375) + free_unbind_inplace_kernel (virtual_slot.py:101-127):
1. double-free virtual page v: `freed_p_pages = v2p[v] = -1` (already tombstoned).
2. Float free: `physical_to_virtual.index_fill_(0, [-1], -1)` — VERIFIED torch index_fill_ wraps → tombstones the LAST physical page's p2v (a LIVE page owned by another request).
3. `-1` appended to `_free_phys_pages` → `take_physical_pages` hands physical -1 out → `bind` writes v2p[v]=-1, p2v[-1]=v (wrapped) → KV store follows physical -1 → torch eager wraps to last KV row = cross-request KV corruption; C/Triton kernels with no mask (copy_all_layer_kv_cache_tiled, store_cache) = raw OOB byte write; free_unbind_inplace_kernel tl.store(p2v_ptr-8) = raw OOB write below the table.
4. EAGER MultiEnded free HAS always-on stale-slot assert; LAZY path + Float have NONE (not even debug-gated). free_unbind_inplace also races if a single call carries duplicate v ids (callers must unique at ps==1; contract-only).
Consequence classes: cross-request KV aliasing/corruption or 8-byte GPU OOB write — both match the audit's target classes. Open gate: caller reachability (radix-lead hunt).
Fix skeleton (pending reachability): mask `freed_p_pages >= 0` before index_fill_/cat in Float.free + the unique-dedup exists; add debug-mode stale assert to _free_lazy + Float (mirroring EAGER's), production filter (mask-and-count) + counter surfaced in debug_print/verify_byte_accounting. Consistent with the SWA composite's existing `swa_v2p > 0` filter pattern.

## Triage of alloc-auditor candidates (pools-lead)
- BUG-ALLOC-1 (hisparse guard spurious-None): **REFUTED** (CPU repro): need_sort staged pages are whole pages counted exactly in available_size()//ps; guard==inner capacity. No divergence. Residual note: hisparse internal asserts leave nothing uncommitted (sub-allocs pre-guarded) — assert-crash leak only theoretical.
- BUG-ALLOC-2: same refutation as 1 for the spurious-None direction.
- BUG-ALLOC-4 (Triton alloc_extend_kernel part-3 unmasked load OOB): **REFUTED** — reaching Part 3 implies num_part3>0 which implies num_page_start_loc_self>=1 ⇒ load address in-bounds; dead lanes never evaluate the load thanks to Triton per-program early returns. Modeled + swept (test_pools_fuzz/test_kernel_hardning_regressions.py::test_alloc_extend_kernel_part3_unreachable_on_dead_lanes). Kernel change REVERTED (keeping the tree clean of non-bugs).
- BUG-ALLOC-5 (ps==1 host/kernel formula desync): latent only (no prod reachability: ps==1 uses TokenToKVPoolAllocator; SWA asserts ps>1). Landed defense: `assert page_size > 1` in PagedTokenToKVPoolAllocator.alloc_decode + regression test. KEPT.
- BUG-ALLOC-3 (hisparse alloc_device_buffer page-split free): aliasing question routed to features (DSA window policy). Mechanics stand by.

## FIX LANDED (agent/pools-lead): MambaSlotAllocator double-free hardening (features C1, routed)
- Mechanics CONFIRMED: caller double-free of a slot → free list holds it twice → handed to TWO requests → shared mamba state tensor (cross-request state corruption). Repro: pools_findings + test_mamba_slot_allocator.py::test_double_free_consequence_documented.
- Fix: (a) always-on integer accounting assert `available_size() <= size` on free() (no sync: python ints; catches accounting overflow AND the later overfill symptom of a corrupt list); (b) SGLANG_DEBUG_MEMORY_POOL-gated unique sweep mirroring paged `_debug_check_no_duplicate_pages`; (c) documented caller contract in free().
- Tests (4, CPU, all pass): group roundtrip; overflow assert on dup; debug unique sweep; documented dup-handout consequence.
## FIX LANDED: paged alloc_decode ps==1 assert (BUG-ALLOC-5) — see board #32 in review.

## FIX LANDED (agent/pools-lead): int32 guard on DCP-widened virtual id spaces (allocplan F1, LATENT->guarded)
- kv_cache_configurator.logical_token_capacity: assert widened virtual capacity < 2^31 (req_to_token is int32; a DCP-widened wrap silently aliases requests' KV + torch negative-index offload reads wrong rows = cross-request leakage). Mirrors unified_memory_pool._assert_physical_id_bound.
- memory_pool.DSATokenToKVPool: same bound on derived DSA index_buf_size.
- Version surface (allocplan): v0.5.17/v0.5.20+/latest affected class; v0.5.14/.15 lack attn-DCP (class absent) unless a plain pool hits 2^31 (unrealistic).
- Trigger bound: max_total_num_tokens * attn_dcp_size >= 2^31 — latent today (no prod deployment near), would be silent corruption if ever configured.
