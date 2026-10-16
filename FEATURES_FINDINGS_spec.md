# Spec-auditor findings: EAGLE/DSPARK draft KV + multimodal embedding
Branch: agent/spec-auditor (off origin/main 3420bae)

## Scope files covered
- [ ] python/sglang/srt/mem_cache/unified_draft_pool.py (read: binds draft lanes in unified pool; move_kv_cache raises; no alloc/free of its own)
- [ ] python/sglang/srt/mem_cache/ple_state_pool.py (read: ShortConvPool/NGramPool slot-indexed states; reset_slots/copy_slots null-safe)
- [ ] python/sglang/srt/speculative/draft_checkpoint.py
- [ ] python/sglang/srt/speculative/draft_utils.py
- [ ] python/sglang/srt/speculative/draft_worker_common.py
- [ ] python/sglang/srt/speculative/eagle_worker_v2.py (partial: draft(), verify(), draft_forward)
- [ ] python/sglang/srt/speculative/eagle_worker_common.py (read: prepare_for_draft, run_eagle_verify, move_accept)
- [ ] python/sglang/srt/speculative/eagle_info.py (partial: EagleDraftInput filter/merge)
- [ ] python/sglang/srt/speculative/eagle_utils.py (eagle_prepare_for_decode)
- [ ] python/sglang/srt/speculative/eagle_disaggregation.py (read)
- [ ] python/sglang/srt/speculative/dspark_disaggregation.py (read)
- [ ] python/sglang/srt/speculative/dspark_components/* 
- [ ] python/sglang/srt/speculative/dflash_disaggregation.py
- [ ] python/sglang/srt/mem_cache/allocation.py (alloc_for_spec_decode)
- [ ] python/sglang/srt/mem_cache/common.py (release_kv_cache, _release_overallocated_kv_indices)
- [ ] python/sglang/srt/mem_cache/multimodal_cache.py
- [ ] python/sglang/srt/mem_cache/embedding_cache_controller.py (co-owned with radix-lead)
- [ ] python/sglang/srt/mem_cache/embedding_store.py (co-owned with radix-lead)
- [ ] retract/abort path during speculation
- [ ] pd_first_draft_proposal / first-verify-after-handoff lifecycle

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

