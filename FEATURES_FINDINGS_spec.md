# Spec-auditor findings: EAGLE/DSPARK draft KV + multimodal embedding
Branch: agent/spec-auditor (off origin/main 3420bae)

## Scope files covered
- [ ] python/sglang/srt/mem_cache/unified_draft_pool.py (read: binds draft lanes in unified pool; move_kv_cache raises; no alloc/free of its own)
- [ ] python/sglang/srt/mem_cache/ple_state_pool.py (read: ShortConvPool/NGramPool slot-indexed states; reset_slots/copy_slots null-safe)
- [x] python/sglang/srt/speculative/draft_checkpoint.py (27-line track-index buffer helper; null-safe; CLEAN)
- [x] python/sglang/srt/speculative/draft_utils.py (backend factory only; no KV lifecycle; CLEAN)
- [x] python/sglang/srt/speculative/draft_worker_common.py (build_draft_tp_worker/make_draft_input_v2; no KV lifecycle; CLEAN)
- [x] python/sglang/srt/speculative/eagle_worker_v2.py (draft/verify/_draft_extend_for_decode/_draft_extend_for_prefill/adaptive state; CLEAN)
- [ ] python/sglang/srt/speculative/eagle_worker_common.py (read: prepare_for_draft, run_eagle_verify, move_accept)
- [x] python/sglang/srt/speculative/eagle_info.py (EagleDraftInput/VerifyInput/DraftExtendInput filter/merge; no KV ownership in spec_info; CLEAN)
- [ ] python/sglang/srt/speculative/eagle_utils.py (eagle_prepare_for_decode)
- [ ] python/sglang/srt/speculative/eagle_disaggregation.py (read)
- [ ] python/sglang/srt/speculative/dspark_disaggregation.py (read)
- [x] python/sglang/srt/speculative/dspark_components/* (worker/verify/planner/draft/kv_inject read; commit-gated KV inject; CLEAN)
- [x] python/sglang/srt/speculative/dflash_disaggregation.py (32 lines, draft-input assembly only; CLEAN)
- [ ] python/sglang/srt/mem_cache/allocation.py (alloc_for_spec_decode)
- [ ] python/sglang/srt/mem_cache/common.py (release_kv_cache, _release_overallocated_kv_indices)
- [x] python/sglang/srt/mem_cache/multimodal_cache.py (refcount-free LRU, idempotent free; CLEAN)
- [x] python/sglang/srt/mem_cache/embedding_cache_controller.py (C1 CONFIRMED -> radix-emb #16-pending fix w/ my PoC; I verify)
- [x] python/sglang/srt/mem_cache/embedding_store.py (abstract base; mooncake subclass short-read F-3 CONFIRMED -> radix-emb #14 shipped)
- [x] retract/abort path during speculation (retract_decode->release_req->release_kv_cache trace + reset_for_retract; CPU harness test_spec_retract_free_roundtrip.py PINS invariant; CLEAN)
- [x] pd_first_draft_proposal / first-verify-after-handoff lifecycle (verified decode prealloc seeds committed=allocated=fill_len; CLEAN)

## Draft KV lifecycle model (traced)
- eagle_prepare_for_decode (eagle_utils.py:1112): page_aligned_decode_alloc_lens(reserve=2*alloc_len_per_decode,
  page_size) computes nxt = ceil(kv_committed_len + reserve, page); alloc_for_spec_decode assigns req_to_token
  [cur_kv_allocated_len, nxt) and sets kv_allocated_len = max(allocated, nxt).
- verify: accepted-token KV moved via move_accept_tokens_to_target_kvcache to slots at seq_lens (+accept); 
  batch_result_processor increments kv_committed_len += num_accept_tokens. Unaccepted draft slots are NOT
  freed per-iteration - they stay inside [kv_committed_len, kv_allocated_len) over-allocation and are
  absorbed/freed at release_kv_cache: free_kv_row([cache_protected_len, owned_kv_len)) then
  _release_overallocated_kv_indices frees [ceil_align(owned_kv_len, page_size), kv_allocated_len).
- So the invariant to check everywhere: kv_committed_len <= kv_allocated_len, and any page >=
  ceil(kv_committed_len, page_size) is allocator-owned (not written to tree, not double-referenced).

## Candidates
(none confirmed yet)

## Notes / open questions
- _release_overallocated_kv_indices asserts start==end when spec off; with spec on,
  [owned_kv_len, ceil_align) page-shared with freed tail relies on free_kv_row page-disjointness. CHECK.
- first-verify-after-PD-handoff: decode allocates spec window over received KV; does
  kv_committed_len/allocated_len get seeded correctly at handoff? (eagle_disaggregation build_eagle_disagg_draft_input)
- retract/abort during spec: are draft slots in the over-alloc freed once (backup_kv_cache vs release)?

---

## CANDIDATE C1 (2026-10-16, spec-auditor): Embedding prefetch wedge -> permanent FILLING page leak + permanent cache-miss

**Class:** leak (pool pages) + availability/perf degradation. Production path: multimodal on Kimi/Gemma with
`mm_global_cache` (Mooncake embedding store), which our HiCache Mooncake deployments can enable.

**Where:**
- `python/sglang/srt/disaggregation/encoder/server.py::_wait_global_cache_prefetch` (~L1415-1450): waits up to 60s
  for `check_prefetch_progress(req_id)`; on ANY exception (incl. `asyncio.TimeoutError`) marks hit items for ViT
  fallback and moves on. Does NOT cancel/evict the prefetch's FILLING entries.
- `python/sglang/srt/mem_cache/embedding_cache_controller.py`:
  - `prefetch()` (~L562) allocates pages and sets entries FILLING, enqueues op keyed by `req_id`.
  - `check_prefetch_progress()` (~L748) only pops `ongoing_prefetch[req_id]` when the op is finished.
  - `_finish_get()` is the ONLY place entries are marked READY or freed; it runs on `_io_loop` only after
    `embedding_store.batch_get_into_multi_buffers()` returns.
  - `MooncakeEmbeddingStore.batch_get_into_multi_buffers` (mooncake_embedding_store.py:87) has NO timeout - it
    calls into the Mooncake store client; a wedged store/replica blocks the single `_io_loop` thread forever.
  - `EmbeddingCacheEntry.is_evictable()` requires `state == READY and ref_count == 0`; `_evict_for_pool` only pops
    from the READY LRU. A FILLING entry is unreachable by eviction.
- Net effect of one wedge:
  1. The op's pages leak for process lifetime (pool page capacity permanently reduced).
  2. `_io_loop` is single-threaded and serial: one wedged `batch_get` blocks ALL later prefetches AND inserts for
     every request -> the FILLING-entry set grows on every new-image request until the pool is exhausted;
     `store_to_pool_async` then fails allocation and `_allocate_with_eviction` cannot evict FILLING entries ->
     global cache dead (all requests fall back to ViT; embeddings never cached).
  3. Subsequent requests with the SAME image hash: `prefetch()`/`store_to_pool_async` skip the existing FILLING
     entry (dedup), so they can never re-drive the load; `has_local_embedding` is False forever -> permanent
     cache-miss for that content hash.
- 60s timeout path alone (store slow but not wedged): op finishes later, `_finish_get` marks READY - no leak.
  The leak needs `batch_get` to never return True/False (hang or raise-inside-worker swallowed?). A raise inside
  `_io_loop` IS caught (`except Exception: results=[False]*len`) so only a true hang leaks. Mooncake client call
  hang is the realistic trigger (backend tag shared across replicas, MT master stall, RDMA wedge - listed in our
  known classes 'Mooncake transfer worker keep-alive').

**PoC sketch (regression test, CPU-only, no Mooncake):**
- Instantiate `EmbeddingCacheController` with a fake `EmbeddingStore` whose `batch_get_into_multi_buffers` blocks
  on an event (or sleeps beyond a threshold).
- Call `controller.prefetch("req1", ["h1"], [N], Modality.IMAGE)`; assert pool `free_pages` drops.
- Do NOT unblock; call `controller.prefetch("req2", ["h2"], [N], IMAGE)`; assert it also allocates (queue grows).
- Simulate the encoder-server timeout: there is no public cancel; assert after T that `entries["h1"].state ==
  FILLING`, `free_pages` unchanged, and `controller._evict_for_pool(pool, N)` frees nothing (pool full -> later
  prefetch returns "failed to allocate").
- Then unblock the fake store; `_finish_get` should (after fix) also be raced by a cancel path.
- FAIL on current code: free_pages never recovers without the fake returning. PASS criteria after fix:
  a `cancel_prefetch(req_id)` / timeout inside `_io_loop` frees FILLING pages and marks op done.

**Fix direction (minimal, ownership: radix-lead shared - coordinate before editing):**
1. Give `EmbeddingPrefetchOperation` a deadline; in `_io_loop`, re-drive or fail the op after deadline and call
   `_finish_get(op, [False]*len)` (frees pages via existing path).
2. Add controller `cancel_prefetch(req_id)` (remove op, `_finish_get` with False); call it from the encoder
   server's timeout `except` branch.
3. Consider: `batch_get_into_multi_buffers` backend call wrapped in a thread with `future.result(timeout)` so a
   wedged Mooncake client can't block the single io thread (still leaves the thread parked; bounded by thread
   pool).

**Versions:** check needed - controller introduced with mm global cache; likely present in v0.5.14+ (Kimi/Gemma
deployments). Confirm via git tag diff before filing.


## Turn-2 additions
- Read existing tests: test_embedding_cache_controller.py (rich pin/unpin LRU coverage), test_staging_draft_kv_slots
  (Mooncake draft slot ordering), kernels/ops/speculative tests. GAPS confirmed: no test for disagg+spec both-sides
  KV release; no test for retract-during-spec alloc/free round trip; no test on embedding prefetch timeout.
- eagle_utils.eagle_prepare_for_verify: verify out_cache_loc = row view [seq_lens, seq_lens+draft_token_num)
  (assign_extend_cache_locs_uniform_func), bound by page-aligned reserve; no independent alloc -> no free needed.
- dp_spec_prefill_coordination.py (53 lines, phase orchestration only), spec_tp_sync.py: no KV ownership.
- adaptive spec (eagle_worker_v2 apply_runtime_state/_override_worker_state): swaps whole backend/graph-runner
  objects; reserve reserve-per-decode reads max over configured candidates via bags; pool slot math unchanged.
- DSpark RaggedVerifyLayout validates cap/verify_lens bounds at construction; worker caps budget per-req.
- mm abort paths: Req.release_mm_inputs_on_abort -> MultimodalInputs.release_features (per-item
  release_transport_proxies + feature=None; finally-guarded) for session/non-session; scheduler handle_generate_request
  wraps proxy release in ExitStack mm_cleanup. Double-release safe (proxy release is idempotent pop-and-release).
- disagg prefill + spec: draft KV rides target's virtual ids in a SEPARATE pool object (fused lanes) shipped as
  extra entries (num_draft_entries) in the same transfer; decode prealloc allocates fill_len; deferred decode KV
  release class is disagg-lead's territory - leaving both-sides-release verification to coordinate with them.


## F-spec-1 (CONFIRMED by features-lead): embedding prefetch wedge leak — VERSION SWEEP
- embedding_cache_controller.py: ABSENT in v0.5.14, v0.5.15.post1, v0.5.17; present v0.5.20 + HEAD
  (features-lead's matrix agrees). K3 base 87dd4596/bd4828e2: unfetchable from this sandbox (upstream
  remote rejects those SHAs — they are branch-internal); inferred same-generation as v0.5.16+ => likely PRESENT.
- Sibling S1 (short-read-as-success): mooncake_embedding_store batch_get + batch_get_into_multi_buffers treat
  res>0 as success in EVERY tag v0.5.14..v0.5.20 and HEAD (byte-count semantics documented in
  mooncake_store.py _batch_postprocess: positive = bytes read, never length-checked). AFFECTED: all production
  versions; embeddings consumed partially-garbage on short read. Same class exists once in generic
  mooncake_store.py (_batch_postprocess res>0 group check) — hicache team territory (earlier-audit class
  'Mooncake store short reads treated as success'); embedding variant is mine to fix per features-lead.
- PoC test_embedding_prefetch_timeout_leak.py RUNS on CPU (demonstrates leak on HEAD).
- test_spec_retract_free_roundtrip.py added (retract-during-spec alloc/free roundtrip; pins invariant
  kv_allocated_len >= ceil(owned_committed,page) at release; uses REAL page_aligned_decode_alloc_lens).

## Test-class mapping (lead request: which shipped tests pin which known class in my area)
- test_embedding_cache_controller.py: pin/unpin LRU semantics (embedding pin ownership class) - pins the
  'embedding cache unpin of entries not pinned by the caller' class at LRU/refcount level; NOT abort/timeout.
- test_staging_draft_kv_slots.py: Mooncake draft KV slot ordering (sibling of 'Mooncake transfer batch indices').
- test_dense_fused_draft_geometry.py / test_fused_draft_placement.py: unified draft lane geometry.
- test_kv_cache_configurator_draft_gate.py / test_kv_index_translator_draft_disposition.py: draft KV virtual-id
  disposition on unified pools.
- test_pd_eagle_draft_proposal.py (registered/spec/): PD first-verify proposal distribution correctness.
- NO shipped test pins: draft KV verify/FAIL free paths, disagg+spec both-sides release, retract during spec,
  embedding prefetch timeout (the four gaps features-lead named). My two new tests cover retract + timeout.

## Extra files covered turn-3
- multi_layer_eagle_worker_v2.py + kernels/ops/speculative/multi_layer_eagle.py widened-window kernel:
  pos bounds clamped (max(pos,0) low; pos<=seq_len+draft_token-1 < row width high); padded rows -> req_pool slot 0 dummy; CLEAN.
- eagle_draft_extend/frozen_kv_mtp cuda graph runners: padded rows write reserved req_pool slot 0 / out_cache_loc zeroed (dummy slot), seq_lens fill; consistent with pool's +1 dummy slot; CLEAN.
- decoupled_spec_io.py + pp_spec_relay.py: NOT in production surface (decoupled_spec_role=null, PP not deployed); skipped with reason.
- draft_utils.py has NO alloc/free/kv mentions (grep-verified).

## Turn-4 additions (all CLEAN)
- spec_tp_sync.py / spec_sampling_mask.py / adaptive_spec_params.py: zero alloc/free/kv refs (grep). CLEAN.
- _compact_accept_to_front adversarial CPU check: valid positions preserved for mixed accept lengths
  (bs=4, nd=10, s1=6); padded clamp-to-0 lands only past accept_lens where downstream never reads
  (draft-extend select_index = i*nd + accept_lens-1; committed-KV reads [0..accept_len)). Documented claim verified.
- dspark planner verify budget: local_verify_tier_num_tokens caps total at bs*verify_num_draft_tokens;
  RaggedVerifyLayout.__post_init__ validates per-row cap + sum bounds; cap-accept consistency enforced;
  SGLANG_SIMULATE_ACC misuse guarded by explicit raise. CLEAN.

## Turn-5 (lead-priority order)
1. (3) _compact_accept_to_front + accept_index clamping (topk>1) — VERDICT SAFE:
   CPU-validated valid-position preservation; padded clamp-0 touches only slots >= accept_len
   which downstream never reads (select_index = i*nd + accept_lens-1; committed range [0..accept_len)).
   Mid-gap -1 in accept_index is impossible: the sgl-kernel tree_speculative_sampling_target_only
   writes accepted nodes ordered at columns [0..accept_len), and prepare_draft_extend's
   (accept_index/num_draft, %num_draft) retrive coords rely on that same ordered contract
   (a mid-gap -1 would index -1 and break everything, tested surfaces would fail loudly).
2. (2) dspark verify budget cutoff — VERDICT SAFE (earlier trace + planner read):
   per-req cap = verify_num_draft_tokens enforced by RaggedVerifyLayout.__post_init__ chain;
   tier totals capped at bs*verify_num_draft_tokens; cap-accept consistency constraint
   (is_cap_accept == predicts.size(1)-1==verify_w) enforced in planner; SIMULATE_ACC misuse
   guarded by explicit raise.
3. (1) tail files spec_tp_sync.py / spec_sampling_mask.py / adaptive_spec_params.py — no KV
   lifecycle; CLEAN. (spec_tp_sync prevents TP divergence in spec decisions.)

## G2 close-out (kvpool-auditor handoff: set_kv_buffer_prefix_valid input validation)
- memory_pool.set_kv_buffer_prefix_valid (memory_pool.py:3226) validates: loc_2d rank-2,
  commit_lens shape == batch, cache_k/cache_v rows == loc_2d.numel (raises otherwise),
  normalizes dtype/device, contiguous.
- Kernel set_kv_buffer_prefix_valid_tiled (kernels/ops/kvcache/cache_move.py:14): per-row
  clamp `row >= commit_len -> return`; byte-mask bounded by ROW_BYTES. commit_len > shape[1]
  is a no-op (bounded by grid dim = shape[1]); commit_len smaller = partial commit (silent
  truncation, NOT corruption; spec caller's kv_committed_len uses the same commit inputs).
- Spec caller side: kv_index_translator.commit computes commit_i = min(plan target commit_lens,
  verify staging verify_lens) (kv_index_translator.py:1213-1218) and pcl plan commit_lens <= seq
  bounds validated (pcl/target_planner.py:206-229) — invariant handed to spec callers holds.
- VERDICT: no memory-unsafe window; the API contracts on the spec callers hold at HEAD.
