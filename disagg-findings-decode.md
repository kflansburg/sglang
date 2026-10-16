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

## TODO
- _pre_alloc / pop_preallocated paths (prealloc KV ownership)
- retract + SWA offload guard (drain_before_retraction call site)
- admission abort & grammar double release on decode side
- decode_hicache_mixin + decode_schedule_batch_mixin
- scheduler.py seams: ladder/timeout of deferred releases (is resolve_deferred_releases called every loop? what if event loop hops?)
