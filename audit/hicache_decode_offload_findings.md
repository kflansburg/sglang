# HiCache decode-offload audit (decode-offload-audit)

Scope:
- `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py` (host/storage side)
- `python/sglang/srt/disaggregation/decode_hicache_mixin.py` (host/restore side)
- `python/sglang/srt/mem_cache/mla_host_dedup.py` (fully mine)

Production config: `--disaggregation-mode decode` + `--disaggregation-decode-enable-offload-kvcache`, page 64, fp8_e4m3/bf16, host-pool offload target, SWA + MLA models.

Method: trace one offloaded page through enqueue -> host alloc -> D2H -> ack/abort/retract -> H2S backup -> ack_backup -> host free. Status per item: CLEAN / SUSPECT / BUG.

---

## Trace skeleton (verified in code)

1. `batch_result_processor._handle_finish_state_updated_req`: every non-finished decode step calls `offload_kv_cache(req)`; on finish it calls `offload_kv_cache(req)` once more and falls back to `finalize_release_on_finish(req)` if that returns False.
2. `offload_kv_cache`: allocs host slots via `cache_controller.write` (device->host, enqueues `CacheOperation` in `write_queue`, returns `host_indices`), records `(req, host_indices, tokens, t, pool_transfers)` in `ongoing_offload[ack_id]`, bumps `offload_inflight[req]` (WeakKeyDict keyed on Req identity) and `offloaded_state[req].inc_len`.
3. `process_decode_queue` (every scheduler loop) -> `check_offload_progress` -> `_check_offload_progress`: pops `ack_write_queue`, `finish_event.synchronize()`, removes from `ongoing_offload`, decrements inflight, triggers `_trigger_backup` (host->storage `write_storage`, tracked in `ongoing_backup[ack_id]` keyed by StorageOperation.id). If `req.finished() and not inflight`, it calls `_release_finished_req` which does NOT free device KV used by prefill prefix (cache_protected_len) and DOES free `[protected, allocated)` + unlock + req_to_token_pool.free. 
4. `_check_backup_progress`: on `ack_backup_queue` items, frees `host_indices` and each pool_transfer's host_indices on the appropriate host sub-pool.

## Findings

### F1. Device-row lifetime vs in-flight D2H — **CLEAN** for all reachable paths.
Verified: (a) finishing abort gated by `_has_inflight_offload` in `finalize_release_on_finish` — rows stay allocated until D2H ack; (b) retraction gated by `drain_before_retraction` which raises (see B1) *before* any rows are freed if anything is pending, and `check_offload_progress` inside it drains acks first; (c) transfer-queue `_release_request` only fires for requests that never entered a running batch (offloads are only submitted from `batch_result_processor` for running decode reqs); (d) checksum-mismatch `release_kv_cache` is the only ungated free with possible in-flight D2H, and it aborts the entire CI run / production request path — in production it aborts only one request after a *detected corruption*, where engine state is already suspect; low severity. Additionally unified pools gate `free`/`move` on transfer-done events (`unified_sub_pool._wait_hicache_transfers`) as a second layer.
Residual assumption: non-unified MHA/MLA decode offload (the MHA/MLA branch of the manager) relies solely on gates (a)-(c); no deferred-free. Gates hold today; flag for future release-path additions.

### B1 (BUG, confirmed by trace): `drain_before_retraction` RuntimeError escapes to the event loop and kills the decode engine.
`update_running_batch` (scheduler.py:4397-4399) and `pause_generation` (scheduler.py:5724-5727) call `decode_offload_manager.drain_before_retraction()` with no try/except. When any offload copy is still pending after one `check_offload_progress()` pass (D2H DMA or unacked write — plausible for Mooncake/page-64 chunks under load, exactly when KV pressure triggers retraction), the manager raises `RuntimeError("refusing to retract ...")`. The exception propagates: `event_loop_normal/overlap` have no catch → `run_scheduler_process` catches it, logs, sends SIGQUIT to the engine (scheduler.py:6203-6209). Result: **engine-wide crash** under memory pressure instead of a skipped retraction. `pause_generation` (operational action) also crashes the engine if a copy is pending at pause time. No test references `drain_before_retraction`. Intended semantics per docstring is "refuse retraction on every rank", i.e. skip-and-retry, but the raise implements refuse-by-crash.
Impact: availability (whole decode replica restart), triggered by internal load, no external fault needed; cross-rank: MAX-allreduce makes every rank raise together even if only one rank lagged.
Suggested fix direction (not mine to submit): catch in the two call sites and treat like `batch_is_full`/skip retraction this iteration (loop retries after acks drain), or have drain loop-wait with a bounded budget. Invariant check: retraction may only proceed when `not (ack_write_queue or ongoing_offload or offload_inflight)`; today the invariant is "enforced" by dying. Regression-test shape: stub decode_offload_manager with ongoing_offload non-empty, call update_running_batch under kv_full_retract_flag — expect retraction deferred (batch survives, no exception), and acks processed next iteration.
B1 regression test status: repro at audit/repros/repro_drain_crash.py exec's the exact upstream drain source with a pending offload and proves the raise; scheduler-side uncaught confirmed by source read (event_loop_normal/overlap no try; run_scheduler_process catch-all → `parent_process.send_signal(signal.SIGQUIT)` at scheduler.py:6203-6209). Anchor tests: no existing test references `drain_before_retraction` (grep test/ empty) — gap confirmed.

Evidence:
- `release_kv_cache` (mem_cache/common.py:394) -> `tree_cache.free_kv_row` -> `free_kv_row_segments` -> `allocator.free/free_segment` immediately returns pages to `free_pages` (`paged.py` has no hicache gate; `base.py` `set_hicache_transfer_done_event` is a no-op there).
- Gating exists only in `unified_sub_pool.py:2005` (`_wait_hicache_transfers`) and `unified_{hybrid_swa,mamba}.py` forwarding; `PagedTokenToKVPoolAllocator` (paged.py) doesn't gate.
- Decode offload supports `MHATokenToKVPool`/`MLATokenToKVPool` (non-unified) via `build_kv_host_pool` when `self.kv_cache` is not `UnifiedSWAKVPool` — `decode_kvcache_offload_manager.py:105-115`. Those pools are exactly the ones whose allocator does not defer frees.
- Production SWA models use unified pools (gated) — so for SWA this is defended. Non-SWA MLA decode (e.g. DeepSeek-style single-pool MLA on decode without unified cache, if any exists in our fleet) would be exposed.

Action: confirm with fleet which decode pools are non-unified; if any, this is a confirmed UAF-window (cross-request KV pollution) on abort/drain-less release. Note abort path itself: `set_finish_with_abort` -> finished next step -> normal offload finish flow; the in-flight earlier offload ack arrives in a later loop iteration — between finish and ack, `offload_inflight` keeps the req unreleased (guarded by `_has_inflight_offload` in `finalize_release_on_finish`), so device rows stay allocated until ack. So plain finishing abort is OK even for non-unified pools; the hole is retract (`drain_before_retraction` raises before retracting, which callers do not catch? verified: scheduler.py:4397-4399 and 5724-5727 call it unguarded around `retract_decode`/`retract_all`) — so device-free-after-drain happens only after `ongoing_offload`/`offload_inflight` are empty. That closes the device-row race for all pools **provided drain raises reliably** (see F2).

### F2. `drain_before_retraction` drains but cannot prevent a *new* offload from being submitted between drain and retract? Scheduler is single-threaded; no offload submissions happen between the drain and `retract_decode()` in the same iteration. Offloads are submitted only from `batch_result_processor` (next step) and `check_offload_progress` (already inside drain). **CLEAN** — verified the call order in `scheduler.py:4394-4414` and `5715-5735`.

### F3. `offloaded_state` WeakKeyDict entry persists across retract (`reset_for_retract` does not reset offload state; req object is the same). On resume, decode restarts from a fresh rebootstrap: prefill re-sends KV for prompt + prior output. `_prefill_offloaded_len(req)` = len(origin_input_ids) aligned — origin_input_ids unchanged by retract. `state.inc_len` already counts tokens offloaded pre-retract. The resumed request re-decodes the same output tokens; when `offload_kv_cache` next runs, `incremental_total = len(origin)+len(output-1) - prefill_offloaded_len` and `incremental_new = total - inc_len` — if the retract popped outputs such that total < inc_len temporarily, `incremental_new` is negative and `incremental_aligned_len = negative // stride * stride` is negative or 0. Negative non-zero is NOT equal to 0, so it passes the `== 0` guard: `incremental_tokens = all_tokens[start:end]` with end < start gives an empty slice, `token_indices[start:end]` empty, `cache_controller.write(empty, node_id)` allocs 0 host slots (or errors), and a backup of empty tokens is queued. Harmless? `host_indices.numel()==0` — host alloc of 0 may return empty tensor; backup with 0 pages is a no-op. **SUSPECT (benign-looking; verify empty-slice write path doesn't corrupt bookkeeping)**. More important: dock `state.inc_len` is left permanently ahead of the true offloaded prefix, so the tokens between (true_total, inc_len) are never re-offloaded after resume — a host-cache *coverage gap*, not a memory-safety bug.

Wait — re-examine: after rebootstrap, does decode re-supply tokens already output? `_rebootstrap_prefill_len` includes len(origin)+len(output_ids) (decode.py:936), so the resumed request rebuilds KV for everything and continues decoding with the same `output_ids` list. So `incremental_total` only grows. `incremental_new >= 0`. The pre-resume offload entries have inc_len already counted; the same token rows are re-offloaded under different host slots (device indices different post-rebootstrap). No double-free; possible duplicate backup entries with same hashes (idempotent set). **Downgrade to CLEAN with assumption** that rebootstrap keeps `output_ids` monotonic (verified: nothing in retract/resume truncates `output_ids` except line 806 pop of last id on ... `pd_rebootstrap_forced_output_id = req.output_ids.pop()` — pops exactly one; under offload this path only runs when `req.output_ids` non-empty; inc_len is <= len(origin)+len(output-1) - prefill_len, and pop reduces len by 1 making total possibly = inc_len - 1 + ... borderline equal, never less than inc_len - stride. If it goes negative by < one stride, aligned_len becomes 0 (since negative//stride*stride == -stride when new in (-stride,0)?? Python floor division: -1 // 64 == -1, so (-1)//64*64 == -64, **negative, passes the ==0 guard**. So there IS a one-token window where aligned_len = -64 → end = start - 64 → `all_tokens[start:end]` = empty, `token_indices[start:end]` empty (negative end beyond length still empty). CacheOperation with 0 tokens: `HiCacheController.write` allocs 0 slots returning empty tensor (not None), `ongoing_offload[ack_id]` holds empty, `_trigger_backup` with `incremental_tokens=[]` → `page_hashes = []` → `write_storage(host_indices=[], token_ids=[], hash_value=[])` → `_page_backup` iterates 0 pages → ack → `_check_backup_progress` frees empty host indices. Also `transfer.keys = page_hashes[-0:]` = whole list [] ... `page_hashes[-0:]` == full list (==[]). Benign but wastes ack cycles and, critically, **`state.inc_len += incremental_aligned_len` ADDS -64**, moving inc_len backwards by a page — those 64 tokens get RE-offloaded later under new host slots while the earlier backup of them (from before retract) already exists in storage under the same hashes. Storage set idempotent. Host slots: new alloc each time; no aliasing. **Verdict: no memory-safety bug; at most redundant writes.** CLEAN (noise-level).

### F4. `_check_offload_progress` releases device KV via `_release_finished_req` but `tree_cache.free_kv_row(req.kv, [(cache_protected_len, kv_allocated_len)])` frees up to `kv_allocated_len`, which for a finished request that stopped early (stop token before last committed token) may exceed the rows the offloads covered. That's fine for device (deferred free handles) but the overallocated tail `[owned, allocated)` rows: `_release_finished_req` does NOT call `_release_overallocated_kv_indices` (unlike `release_kv_cache`). With page 64, `kv_allocated_len` may exceed `kv_committed_len` by up to 63 tokens worth of last partial page. Freeing `[protected, allocated)` frees the overallocated partial-page rows too — same as `release_kv_cache` does via two calls (free_kv_row up to owned + release_overallocated for the aligned tail). Net effect equivalent; allocator page-dedups. **CLEAN** but flag: the assert diagnostic in `_release_overallocated_kv_indices` is skipped; behavior parity maintained.

### F5. `_check_backup_progress` frees `host_indices` for the KV pool and each `pool_transfers[i].host_indices` for SWA side pool. Where are transfer.host_indices set? In `HybridCacheController.write` via `allocate_host_transfers` — side-pool indices allocated there and mutated onto the transfer objects stored in `ongoing_offload[ack_id][4]`. The same transfer objects are then mutated again in `_trigger_backup` (`transfer.keys = ...`) and passed to `write_storage(extra_pools=pool_transfers)`. The backup thread frees nothing; free happens only here on ack. If `write_storage` raises or backup thread dies (`backup_thread_func` swallows exceptions in `_page_backup_or_log` but still puts operation into ack_backup_queue), ack always arrives → host freed once. However `StorageOperation` id collisions? ids are unique monotonic. **CLEAN.**

But: `ack_backup_queue` None when storage disabled. **RESOLVED CLEAN**: `kv_cache_hook.py:404-408` — `--disaggregation-decode-enable-offload-kvcache` requires `--hicache-storage-backend` non-None, so storage is always enabled when the manager exists. Also note: unified-memory+offload is further restricted to storage 'file' and asserts NOT MLA/SWA/mamba (kv_cache_hook.py:561-577) — the `UnifiedSWAKVPool` branch of the offload manager is therefore currently unreachable from server args (dead-ish code; SWA decode deployments can only use the plain-pool path, and plain `SWAKVPool` raises ValueError "Unsupported KV cache type for decode offload" in the manager __init__). Coverage note: SWA+offload can not even start on main; production SWA versions must differ (K3 branch).

### F6. mla_host_dedup.py review. 
- `MLAHostDedupBroadcaster._bcast_layer`: src does `index_select(out=chunk_rows)`, then NCCL broadcast of `chunk` (a view of the staging buffer); every rank broadcasts the full staging (non-src ignoring its buffer content). Non-src then `index_copy_` into its own layer_buf at the *same* `target` indices — `target` came from `prepare_broadcast(device_indices)` where `device_indices` are **rank-local** KV pool rows. Dedup means non-src ranks have identical KV content for the same request only if their rank-local `device_indices` correspond to identical logical tokens. The broadcast group is attn_tp ranks with MLA (kv same across ranks) — each rank allocates identical token indices for the request ( MLA dedup assumption: all attn tp ranks mirror the same rows) — that's the design premise of dedup (buffer-less host pools on non-source ranks: only src has host buffer; the broadcast fills *device* layer_buf on non-src so device rows match). This is a device-consistency mechanism; host memory exists only on src. Memory-safety: staging buffer is reused every layer; `broadcast` is synchronous NCCL on the load_stream — but `torch.distributed.broadcast` runs on the current stream, and `load_stream` is passed only for `record_stream`. If `prepare_broadcast`/`broadcast_loaded_layer` are invoked while current stream != load_stream, index_select on current stream may race the load_stream H2D that fills layer_buf on src — but index_select *reads* layer_buf, so ordering vs the load matters. Callers in cache_controller drive layerwise load: load on load_stream, then broadcast. record_stream only frees safely; correctness of read-after-write within src requires the load D2H and index_select be stream-ordered. Need the caller. **SUSPECT (needs caller code in cache_controller load path; possibly owner hicache load-back agent already covers).**
- `page_idx = indices[::page_size] // page_size` for DSA: assumes rank-local allocations are page-contiguous and aligned; guarded by numel % page_size. If a rank had non-aligned allocation (partial last page) the // page_size mapping mismatches across ranks → cross-rank row divergence, not memory unsafety per se. Raises on misaligned. OK.
- `destroy()` destroys NCCL group; called from pool teardown. No free of staging (GC). OK.
- `_DEDUP_COMPATIBLE_STORAGE = {None, "", "file"}` — mooncake storage NOT dedup-compatible; dedup disabled there (`storage_supports_host_dedup` false) → `maybe_create_mla_host_dedup_context` returns None. So in production (mooncake store) this whole file is inert. **CLEAN for production surface (gated off); the buffer-less host pool on non-src ranks only engages with file/no storage.**
Anchor test: test_mla_host_dedup_primitives.py — 10 passed / 4 failed in my sandbox; the 4 failures are sandbox-RAM-only (`host_memory_budget_bytes` returns negative free in a 6GB container; ValueError at pool_host/base.py:242 capacity precheck, not logic). No dedup refcount/aliasing test failures.

### F7. HiCacheRestoreGatedKVReceiver / restore state machine / `_clean_hicache_prefetch_resources` (decode_hicache_mixin.py).
- Deferred-free safety of abort-with-inflight-H2D verified CLEAN for unified pools: abort unlocks the restore tree lock, restored slots become tree-owned & evictable, but device `free` is stream-ordered after the load finish_event registered at `start_loading` (unified_sub_pool._wait_hicache_transfers; event set at cache_controller.py:994). Load DMA writes only into tree-restore slots (req-row touched only at commit), and host source indices are reaped by `check_hicache_events`/`loading_check` every step independent of request fate — no leak, no write-into-freed.
- `unlock(None)` is safe (base_prefix_cache.py:532-534 guards None). T2 CLOSED.
- Prefetch host-buffer lifecycle on the FAILED-after-pop path (`pop_prefetch_loaded_tokens` then `finish(ABORT)`) — DEFERRED to prefetch-ctl-audit (their anchor test_storage_prefetch_lifecycle.py covers this exact locus; not duplicating).
- `_commit_hicache_local_restore_to_req`: writes `hicache_restored_kv_indices` into `req_to_token[l1:decode_prefix_len]`. `hicache_restored_kv_indices` computed from `path_device_indices(restored_node)[l1:decode_prefix_len]`. If `restored_node` path is shorter than decode_prefix_len (e.g. loaded_len==0 shortcut when rematch covered everything but pm had l3>0... loaded_len==0 sets READY and returns False BEFORE setting restored_kv_indices? No — line order: `dr.hicache_restored_kv_indices = ...` set BEFORE the `if loaded_len == 0` check. But `_commit_hicache_local_restore_to_req` is only called when restore lock is not None (assert). For loaded_len==0 case, `dr.hicache_restore_lock = self.tree_cache.lock(restored_node)` was set, status READY, commit proceeds, slice assignment with indices of length (decode_prefix_len - l1). `path_device_indices(restored_node)` must cover decode_prefix_len — guaranteed by "restored path holds the whole prefix" comment and the coverage check `rematch.device_prefix_len + loaded_len >= pm.decode_prefix_len`. Then later normal abort cleanup of the req: `release_kv_cache` unlocks req.lock (restore_lock now) and frees row. OK.
- `_clean_hicache_prefetch_resources` unlock(None) safe? `tree_cache.unlock(None)`? Called unconditionally with `decode_req.hicache_restore_lock = None` after; if lock None and unlock doesn't handle None → crash. Check `tree_cache.unlock` None-handling. (T2)

### B2 (note-level, operator path): flush_cache with in-flight decode-offload backups.
`scheduler.flush_cache` is gated by `is_fully_idle`, which checks `decode_offload_manager.ongoing_offload` but NOT `decode_offload_manager.ongoing_backup` (host->storage writes still uploading). `tree_cache.reset()` → `cache_controller.reset()` first runs `_stop_storage_threads()` (stop event + 10s joins, raises if threads still alive — so a hung Mooncake RDMA write >10s during flush_cache crashes the scheduler via uncaught RuntimeError, B1-sibling on a different call path). If threads stop cleanly, in-flight `_page_backup` ops run to completion before queues clear — no torn-write race. Offload manager's `ongoing_backup` entries for ops completed during reset never get acked (queue cleared): stale dict entries accumulate and their host slots are never freed by the manager; pool `clear()` reclaims everything anyway, so impact is bounded dict growth, not host leak. Severity: low (dict growth + crash-on-hung-storage during operator flush). Versions: same structure in all of main/v0.5.x (reset/_stop_storage_threads predate offload manager); the B1-class uncaught RuntimeError from `_stop_storage_threads` applies to main and all four tags wherever reset is reachable with in-flight storage IO — owners of flush_cache gating should note it.

### F8. `check_offload_progress` MIN-allreduce of queue sizes across tp_group (dp_tp_cpu_group). Each rank pops `finish_count` acks — MIN ensures no rank pops an ack another rank hasn't seen, but ack_write_queue is a plain list fed by the scheduler thread only; a rank could append new acks between qsize() snapshot and pop — fine, pops only finish_count. `pop(0)` with finish_count from MIN across ranks: if local queue shorter than finish_count (impossible given MIN includes local qsize)... unless local queue lies about ordering — list, FIFO. OK. **CLEAN.**

### F9. `release_host_resources` calls `decode_host_mem_pool.destroy()` at scheduler teardown (scheduler.py:1900) — but does teardown first stop the D2H transfer engine threads / drain write_queue? `cache_controller.reset` handles storage threads; l2_transfer_engine teardown ordering unknown. If engine thread still references host buffers after destroy → UAF at exit (low severity, process going down). **SUSPECT (teardown-only).** (T3)

### F10. `offload_kv_cache` allocs host BEFORE enqueue; on `write()` returning None it logs and returns False — then caller (finish path) calls `finalize_release_on_finish` which (no inflight) releases device KV. Host not leaked (write returned None means alloc failed and nothing enqueued — but partial alloc? `mem_pool_host.alloc(len)` all-or-none? assume yes). **CLEAN** assuming host alloc atomic (standard).

### F11. `_resolve_offload_transfers` SWA path: `translate_loc_from_full_to_swa(virtual_indices)`; `live_swa_indices = swa_indices[swa_indices > 0]` — SWA rows evicted mid-decode read as <=0 and are skipped, so the trailing SWA window offloaded may be shorter than full. `PoolTransfer(name=SWA, device_indices=live, hit_policy=TRAILING_PAGES)`. Then `_trigger_backup`: `page_count = transfer.device_indices.numel() // self.page_size`; `transfer.keys = page_hashes[-page_count:]` — if live_swa has holes (evicted middle pages dropped by >0 filter but not page-contiguous), keys[-page_count:] pairs the LAST page_count hashes with pages that may not be the trailing pages → hash/content mismatch in storage (integrity bug, salami-slice class: wrong data under right keys → cross-request garbage on later load!). `swa_indices > 0` filter preserves order but drops zeros anywhere, including mid-sequence evictions. With SWA, eviction is from the front (sliding window sinks at row 0?), so holes are at the front: 0 → zero denotes the sink page (comment) — sink pages are interspersed? For page-aligned chunks "whole-page bindings; zero denotes the sink page". If sink pages are the only zeros and sinks appear at deterministic positions (window start), then filtered live indices ARE the trailing pages. Mid-window evictions between offload strides: SWA eviction granularity is page; filtered-out zero pages inside the chunk break the trailing assumption **if SWA can evict a partial middle page** — normally SWA evicts oldest (front) pages, chunk covers [start,end) of full attention positions; the corresponding SWA rows are full-window rows; a middle full-row maps into SWA window only if within window; rows outside window were never bound → 0. Those map to the FRONT of the chunk (oldest). So live == trailing. **CLEAN under SWA-front-eviction assumption; document assumption.** Known class "retract with SWA offload guard" is handled by drain gate (F1/F2).

---

## Anchor test status (sandbox, CPU torch 2.14.1)
- test_decode_hicache_tree_core.py: **PASS** (3/3) — pins my F7/mixin verdicts (storage probe/prefetch node handles, stale anchor degrades to L2, restored indices off restored path).
- test_mla_host_dedup_primitives.py: 10 pass, 4 fail — all 4 are sandbox-RAM `host_memory_budget_bytes` precheck failures (-4.9GB free artifact of this container), not logic.
- test_decode_retraction_backup.py / test_unified_hicache_transfer_lifecycle.py: 17 fail — 13 "Torch not compiled with CUDA enabled" + related retries, 4 same RAM-budget precheck. Zero logic failures. These anchor suites need a GPU box (or larger cgroup) to validate; CI covers them.
- No existing test references `drain_before_retraction` (B1 unpinned) — targeted regression test recommended with fix.

## Follow-up tasks
- T1: DEFERRED to prefetch-ctl-audit (their anchor test_storage_prefetch_lifecycle covers pop/finish ABORT lifecycle).
- T2: CLOSED — unlock(None) guarded.
- T3: CLOSED with SUSPECT-low: `handle_shutdown` sets gracefully_exit with NO drain of `ongoing_offload`/`offload_inflight`; finally-block calls `release_host_resources` → `decode_host_mem_pool.destroy()` while an L2 D2H DMA engine may still reference host buffers (teardown-time UAF/abort in the DMA thread; severity low since process is exiting, but pinned-memory teardown under in-flight DMA can SIGABRT the whole engine and break graceful shutdown). Recommend: drain/check_offload_progress loop (bounded) before destroy, or skip destroy when `ongoing_offload` non-empty (exception path already skips). Not repro'd on CPU (needs CUDA engine).
- T5: CLOSED — F5 clean (args hook mandates storage backend).

## Versions (fetched tags: v0.5.14=4289f36e, v0.5.15.post1=658e0a94, v0.5.17=b6a09f38, v0.5.20=d158602f)

### Retract-vs-offload class (known class "deferred decode KV release")
- main: `drain_before_retraction` guard exists but raises uncaught → **B1 crash**.
- v0.5.14, v0.5.15.post1, v0.5.17, v0.5.20: **NO drain guard at all** (`git grep drain_before_retraction <hash>` empty; the scheduler retract path calls `batch.retract_decode()` / `retract_all(offload_kv=False)` with no offload drain in all four tags — verified scheduler source in d158602f: lines 4124 & 5422, with zero decode_offload_manager references anywhere near retraction). `retract_all` → `release_req` → `release_kv_cache(checkpoint=False)` frees device rows immediately while a queued/in-flight D2H offload still reads them; re-admission (enabled the very next loop since freed rows raise available_size) hands the pages to a new request whose incoming prefill-KV RDMA writes race the stale D2H reads → wrong-pair KV persisted to L2/L3 under correct hashes, served to future requests (cross-request context leakage). **All four production tags affected; main fixed the UAF with the drain but introduced crash B1.** K3 branch not public; assume affected unless the local patch is carried.
- Finish-path guard `offload_inflight` exists already in v0.5.20 (verified) — finish/abort path safe on those tags; the retract path is the hole.

### B1 (main only)
N/A on v0.5.14–v0.5.20 (method absent — they have the UAF instead).

---

## TASK from hicache-lead: unified checkpoint shrink vs tree ownership (radix-lead sibling trace)

Question: unified_radix_cache.checkpoint_into_tree (~L1085-1248) shrinks
`req.kv.cache_protected_len` to `len(new_indices)`. Do the cleanup_after_caching_req
impls (components/base.py:689 FULL, swa.py:1032, mamba.py:644) release TREE ownership
of [len(new_indices), old_protected)? If not, and that range is tree-referenced →
UAF/double-free (plain-radix class).

### Verdict: NOT VULNERABLE on unified (the assert at L1210 converts any real
    ownership violation into a crash, and the only reachable shrink covers
    request-owned rows). No repro constructible without inventing a producer
    that breaks an invariant no current producer breaks. Plain radix lacks the
    assert (radix_cache.py:527-541) — radix-lead's fix should keep unified's
    protocol as the reference.

### Trace (all line refs on main 3420bae)

Ownership invariant I: tree device rows are page-granular; tree owns exactly
[0, page_floor(tree_committed_len)) of a request row, and
cache_protected_len ∈ {page-aligned(tree-owned boundary), page-aligned + r,
0 <= r < page_size} where the +r tail is REQUEST-owned (never tree-referenced):
- Producers of unaligned protected: common.py:265 (match_result.cache_protected_len),
  decode.py:1604 (total_prefix_len = l1+l2+l3), pipeline.py:1514 (matched_len),
  schedule_policy.py:1506 (prefix_len when cache-owned load-back; restored slots
  are tree-owned pages). All bounded by tree + at most one partial page — for
  unaligned sources the tree owns page_floor(protected) (tree keys page-aligned:
  RadixKey.page_aligned at L1152; l2/l3 hit lengths are page-multiples since both
  host nodes and storage hashes are per-page).

Insert (L1126-1161): inserts [prev_prefix_len, page_aligned_len); tree ownership
after insert = page_aligned_len. New protected := len(new_indices) ==
page_aligned_len (assert L1213). The release-time free range
[protected_new, owned) therefore never intersects the tree: [0, protected_new)
is exactly the tree-owned span, protected_new itself page-aligned.

Shrink analysis (old_protected - protected_new):
1) old protected from a PRIOR unified checkpoint (L1238): both page-aligned →
   shrink is a multiple of page_size. Assert L1210
   `protected <= len(new_indices) + page_size - 1` → shrink < page_size →
   shrink == 0. No-op.
2) old protected from unaligned source (tree owns page_floor(old)):
   shrink = old - page_aligned_len. Tree-owned boundary page_floor(old) <=
   page_aligned_len (tree can't own less after insert than before — insert is
   append-only per request-key prefix, see L1126 radix_key built from token_ids
   [0, effective_cache_len) with prev_prefix_len = old protected). Freed rows
   [page_aligned_len, old) = only the partial-page tail r, request-owned by
   invariant I. Safe.
3) Assert-bypass question (shrink >= page_size via prepare_for_caching_req cut):
   - SWA (swa.py:980-1009): returns branching_seqlen only when > protected;
     mamba-enabled returns None. No cut below protected.
   - Mamba (mamba.py:564-): extra_buffer returns mamba_last_track_seqlen. Each
     unified checkpoint is itself floored by the mamba boundary (min at L1120),
     and track boundaries only grow, so any TREE-derived protected <= current
     track boundary → no shrink from mamba. For MATCH-derived protected, the
     mamba boundary is seeded at the matched node (track seqlen = matched
     boundary >= protected) — invariant M: track >= tree-derived protected.
     If invariant M is ever broken by a new producer, the assert L1210 CRASHES
     (DoS-at-worst, fail-stop) instead of silent UAF — acceptable failure mode.
   - floor_cache_len: base no-op (base.py:450); no component override found
     that cuts below protected.
4) cleanup_after_caching_req: FULL base.py:689 no-op; swa.py:1032 frees only
   request-owned out-of-window SWA ring slots; mamba.py:644 frees mamba state
   (request-owned / donated insert copies). None touch device-KV tree rows —
   consistent with protected_new being the exact tree boundary, nothing to
   release.

Reference comparison: plain radix_cache.py:527-541 sets protected =
len(new_indices) with NO `+page-1` assert — that is the locus radix-lead
confirmed. Unified's counterpart is tighter; recommend the radix fix keep
unified's assert + invariant I as the documented protocol.

Additional datapoint: lmcache_unified_radix_cache.py:763 already needed a clamp
`min(protected, tree_owned_len)` for external-flow ownership — evidence the
"protected must equal tree-owned boundary" invariant is load-bearing; lmcache is
out of our production scope but is the one place that violated-and-patched it.

### Repro attempt / test sketch (for radix-lead's regression test)
CPU-constructible shape IF the invariant breaks: build UnifiedRadixCache (harness
pattern: test/registered/unit/mem_cache/test_unified_hicache_transfer_lifecycle.py),
admit req A with prefix match P (tree owns [0,P), protected=P); force a checkpoint
where effective_cache_len floors to P - k*page (needs a producer that cuts — none
目前 reachable; the test would mock a component prepare returning protected-64 and
assert the L1210 assert fires, pinning fail-stop semantics). Negative test worth
shipping instead (no mock): dual-req shared prefix; req A checkpoints mid-decode at
unaligned protected then finishes; assert rows [page_floor(protected), protected) are
freed exactly once and tree slot refcount for [0, page_floor) unchanged; sibling
allocator double-free detector (PagedTokenToKVPoolAllocator debug_mode +
_debug_check_no_duplicate_pages) turns any regression into a test failure.
I could not construct a true-UAF repro without breaking an invariant that has no
breaking producer — that is the (positive) result of this trace.

---

## B1 FIX (branch agent/decode-offload-audit-fix, based on dev=V8)

NOTE: my earlier repro (audit/repros/repro_drain_crash.py) targets PRE-V8 main
(raise on ANY pending). V8 (disagg-lead, dev merge 94321a0a) made the drain
wait bounded (SGLANG_DECODE_OFFLOAD_DRAIN_TIMEOUT_S, default 30s) and raise
only on a TRUE WEDGE — old repro obsolete; kept for history.

Fix: catch RuntimeError at the only two call sites (swept all of python/ — no
other callers/variants; notify-ack path separate):
- scheduler.py update_running_batch (~4404): DEFER — batch.batch_is_full=False,
  return None; batch kept intact; no rows freed; retry naturally next iteration.
- scheduler.py pause_generation (~5744): REFUSE pause — _engine_paused reset to
  False, error logged, return without retracting; engine keeps serving; the
  operator/update flow may retry PauseGeneration. (Chosen over "stay paused
  half-retracted": that would let a weight update run over stale-weight KV.)

Regression tests (test/registered/unit/managers/test_scheduler_pause_generation.py):
- Reworked test_retraction_refused_while_any_rank_tracks_an_offload_copy ->
  test_retraction_deferred_... : wedge simulated via drain-timeout override
  (0.05s); asserts no raise, frees empty, pause refused / engine live, per-rank
  MAX consensus preserved.
- NEW test_kv_full_wedge_defers_and_retries_after_copy_completes: wedged call
  defers (None, batch intact, no free); ack requeued; retry retracts and frees
  only with copy.done True.
Results: 28 passed + 8 subtests (pause suite); drain V8 tests 6/6;
test_decode_hicache_tree_core 3/3. CPU venv, torch 2.14.1+cpu.

## B1 fix — disagg-lead review items closed
- logger.critical "PAUSE REFUSED: D2H wedge" + pending counts (commit 4ec9e7d7).
- Sweep re-verified: drain_before_retraction callers = scheduler.py x2 only;
  Scheduler subclasses inherit both patched methods (no variant overrides).
- PRE-EXISTING GAP surfaced (beyond B1 scope, flagged not fixed): tokenizer
  manager sets is_pause=True BEFORE fire-and-forget dispatch to the scheduler
  (tokenizer_manager.py:2282-2286); no ack. If the scheduler refuses the pause
  (B1 wedge path), the tokenizer still reports paused, and the weight-update
  contract gates on tokenizer-side is_pause (tokenizer_control_mixin.py:471,
  513-519, 576, 608) — so a weight update CAN proceed thinking the engine is
  retracted while the scheduler kept the running batch. Same hazard shape V8's
  option-(a) critique targeted, one layer up. Operators: a refused pause now
  logs PAUSE REFUSED critically; do not proceed to weight update without a
  fresh successful pause (no programmatic ack exists today). Follow-up fix
  needs a pause ack/negative-ack channel — out of B1 scope.
