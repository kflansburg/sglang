# Decode-side disaggregation audit findings (disagg-decode)

Scope: decode.py, decode_schedule_batch_mixin.py, decode_kvcache_offload_manager.py, decode_hicache_mixin.py (+ scheduler seams). Repo: latest upstream main.

Status legend: [CAND] candidate, [CONF] confirmed against code, [CLEAN] checked, no bug.

## Deferred KV release (pop_transferred / _defer_release / resolve_deferred_releases), decode.py:2727-3027

- Upstream has a full deferred-release design: on KVPoll.Failed, if `enable_deferred_kv_release` and the receiver `ensure_abort_notified`, the request is held in `_deferred_releases` and its pages are NOT freed until `kv_mgr.is_abort_release_safe(room, required_acks)` (drain ack from prefill) or `deferred_kv_release_timeout` elapses (frees anyway, warning logged). Host-staged requests never time out — held until drain ack ("A timeout cannot prove that a remote write has stopped"). This addresses the known "free KV while RDMA write in flight" bug for device destinations only partially: after timeout it still frees with in-flight writes possible. Check: what is deferred_kv_release_timeout default, and is timed-out free actually safe for Mooncake? The earlier-audit class "freeing KV after 30s timeout while RDMA write may be in flight" may still be LIVE for device destinations (only a warning); host path is safe. Verify timeout value + mooncake write guarantees. [CAND-1]

- `release_memory_occupation` (decode.py:3019) does `self.queue.clear()` and `self._deferred_releases.clear()` WITHOUT releasing the pre-allocated KV of the queued reqs. If this is pool teardown ("Pool is being torn down") the leak doesn't matter; but verify it can't be called mid-serving (resume_memory_occupation is a no-op that allows new transfers, implying release can happen and service continue — then queued reqs' KV + metadata buffer indices leak). [CAND-2]

## offload manager (decode_kvcache_offload_manager.py)

- `_release_finished_req` guards `req_pool_idx in (None, -1)` before freeing; free path = tree_cache.free_kv_row(protected_len..allocated) + unlock + req_to_token_pool.free + mark_kv_released. Called from both `_check_offload_progress` (when last inflight offload lands) and `finalize_release_on_finish`. Guard makes double-release safe IF req_pool_idx set to None by req_to_token_pool.free — verify ReqToTokenPool.free actually sets req.kv.req_pool_idx = None, and that no OTHER frees happen for the same row elsewhere. [to verify]
- `offload_kv_cache` device_indices D2H read vs device-slot free: comment says prefill-aligned GPU slots freed only at finish. But `drain_before_retraction` raises RuntimeError if any D2H in flight — check retraction call site handles the exception (catches and refuses retract) rather than crashing the scheduler loop. [to verify]
- `_check_backup_progress`: pops `ongoing_backup[ack_id]` then frees host pages. If backup ack arrives for an ack_id not in ongoing_backup -> KeyError. Also: on abort of a req with ongoing backups, host pages stay until ack — is ongoing_backup ever leaked if storage write fails? [to verify]

## Update pass 2 (confirmed/refuted)

- CAND-1 CONFIRMED (design-level, known class still live for device destinations): deferred KV release is enabled by default (SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE=True, TIMEOUT=30s). On KVPoll.Failed with abort_notified, pages held until prefill ABORT_ACKs (is_abort_release_safe, required_acks = #bootstrap_infos) OR 30s deadline — then resolve_deferred_releases calls _do_release → release_kv_cache → pages back in allocator immediately (decode.py:2989-3008, comment 'releasing anyway'). If the ABORT never reached prefill (partition/crash) or the prefill ignores it and its Mooncake RDMA write drains after 30s, those writes land in pages that may already be reissued to a new request → cross-request KV corruption. Host-staged destinations are held indefinitely (correct). Sibling check: prefill side must treat ABORT as authoritative-stop; worth verifying prefill sender kills in-flight mooncake transfers on ABORT receipt (prefill agent's file). Verdict: same class as earlier-audit '30s timeout free with RDMA in flight' — upstream mitigates via drain-ack but the timeout escape hatch re-opens the window. Impact requires failed-abort racing slow RDMA, so rare but real.
- CAND-2 mostly refuted: release_memory_occupation (decode.py:3019) is only called from weight_updater with assert is_fully_idle (weight_updater.py:341) followed by flush_cache. Dropping queue+deferred entries there is intended. RESIDUAL: assert uses is_fully_idle(ignore_waiting=_engine_paused) — verify is_fully_idle counts transfer-queue + deferred entries as non-idle; if a deferred release can be pending while 'idle', its KV row + metadata idx leak permanently (req gone from all queues). Low priority, flag for challenge.
- decode abort_request cross-check (scheduler.py:5602): transfer-queue abort leaves req in queue; kv_receiver.abort() sets conclude_state=Failed → later pop_transferred sees Failed → deferred release path. Consistent, no leak. Waiting-queue decode abort: release_kv_cache(checkpoint=True) inserts req KV into tree as committed — fine since transferred KV is valid. host_staged abort: prepare_abort only; pop_transferred Success+FINISH_ABORT releases. Consistent.
- Grammar: decode-side grammar accept handled in process_prebuilt; on grammar ValueError uses to_finish → single release path via update_finish_state (comment in code says this was the fix). No double release seen. [CLEAN]
- retract vs SWA offload guard (KNOWN BUG): current upstream has drain_before_retraction called in BOTH retract sites (scheduler.py:4399, 5727) BEFORE retract_decode/retract_all; it refuses (RuntimeError) while any D2H offload copy reads device KV, all-reduced MAX across TP. retract then uses backup_kv_cache (synchronous get_cpu_copy) with offload path, or offload_kv=False rebootstrap path. Mechanism looks fixed upstream; only question: who catches the RuntimeError? If nothing catches, scheduler crashes instead of deferring retraction (availability bug, not memory). TODO: check call-site try/except.

## HiCache mixin (decode_hicache_mixin.py)
- _try_hicache_queue_load_back failback: on partial coverage failure sets FAILED; pop_transferred Failed path calls _clean_hicache_prefetch_resources (finish ABORT + unlock restore_lock) → then release path. Lock/unlock pairing looks owned. Verify: init_load_back may allocate device indices for the load; on FAILED are those returned to the tree (cache-owned via load_back_is_cache_owned=True → tree keeps them as cached nodes; no leak but check no dangling lock).
- _commit_hicache_local_restore_to_req: unlocks old req.lock, swaps to restore_lock, sets hicache_restore_lock=None to avoid double unlock. Clean.
- init_load_back returns None in buffer mode → bare tuple unpack would TypeError; production uses page_first_direct (not buffer mode) so unreachable in production, but a latent crash for buffer-mode users. Low.

## TODO
- _pre_alloc / pop_preallocated paths (prealloc KV ownership)
- retract + SWA offload guard (drain_before_retraction call site)
- admission abort & grammar double release on decode side
- decode_hicache_mixin + decode_schedule_batch_mixin
- scheduler.py seams: ladder/timeout of deferred releases (is resolve_deferred_releases called every loop? what if event loop hops?)
