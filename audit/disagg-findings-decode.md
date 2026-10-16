# Decode-side disaggregation audit findings (disagg-decode)

## FIXED (shipped, in review/merge queue)
- V4-decode FIX (board #19, @be5bd52): device deferred release holds past 30s until drain/peer-down. (See top section.)
- F12: AUX_DATA wire-controlled arbitrary-offset write + aux-racing-reissued-slot cross-request metadata overwrite (mooncake _handle_aux_data + AuxDataCodec). SHIPPED via disagg-mooncake's codec-side design (board #42): bounds in AuxDataCodec (raises ValueError: buffer_index/apparent-slot bounds, payload==item_len) + room∈request_status gate in _handle_aux_data. My parallel handler-side fix (d7f400f) was retracted in 9b2299f to avoid a merge conflict; my contribution = regression tests test_mooncake_aux_data_bounds.py (board #44, tests-only, ordered after #42), verified 10/10 green on the union tree.

## Extra verdicts (role-switch watch)
- Heartbeat/node-failure vs role-switch: CLEAN. CommonKVManager.teardown sets _heartbeat_shutdown; the daemon thread is not joined and can run one more iteration on the dying manager (possible late zmq exception, logged and harmless) — all mutated tables are per-manager; class-level receiver socket cache (close_all_sockets) keeps _ctx reusable; no cross-role state corruption possible. No fix needed.
- is_fully_idle gates role-switch on has_pending_deferred_releases() and ongoing_offload — verified; V4 holds correctly block destructive ops.

## FIXED (shipped)
- V4-decode FIX on agent/disagg-decode @be5bd52, board #19: device-destination deferred release no longer frees at 30s timeout; holds until drain ack or prefill-down (prefill_info_table pop). 'peer_dead' metric outcome. Supersedes CAND-1 below for device path on HEAD. Older prod versions (v0.5.14/.15.post1/.17) have NO deferred release at all and free immediately on failure — confirmed via fetched tags; the local-patch backport story for those belongs to release engineering.

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

## Pass 3 (final coverage of lifecycle)

- pop_preallocated FINISH_ABORT sweep (decode.py:1368-1391): only reqs not yet preallocated live in this queue (preallocated ones are popped to transfer queue in the same call), so kv_receiver.clear()+None suffices; no KV/lock held yet. Lock leak on reservation-not-fit avoided via unlock at 1543/1563. [CLEAN]
- _update_handshake_waiters Failed -> prepare_abort; swept next pop. Handshake precedes prealloc, nothing held. [CLEAN]
- hold_rebootstrap (decode.py:896): reqs held in held_rebootstrap_reqs while engine paused for weight update; their KV row was already freed by retract_all(offload_kv=False). RESIDUAL: abort_request does not scan held_rebootstrap_reqs -> a client abort during a paused engine is never delivered to these reqs; on resume they recompute prefill for a dead client (leak of a running request, not memory safety). [CAND-4, low]
- retract guard: drain_before_retraction RuntimeError is UNCAUGHT in event loop path (scheduler.py:4399 inside update_running_batch) and pause path (:5727) -> decode process dies when OOM-retract coincides with in-flight offload D2H (prod config: decode offload enabled). Memory-safe but availability-hostile; the correct behavior is a one-iteration defer. [CAND-3]
- _commit_transfer_to_req: metadata bootstrap_room validation present (0 -> abort; mismatch -> abort). kv_receiver.clear()+None on every exit incl. abort. [CLEAN]
- decode_schedule_batch_mixin: process_prebuilt grammar uses to_finish (single release). PREBUILT writes out_cache_loc from pre-committed req_to_token rows; indexes from prefix lock. [CLEAN]
- resolve_deferred_releases committed 3338 in process_decode_queue each loop -> timeouts fire. release_memory_occupation drops deferred entries only under is_fully_idle assert + flush_cache. [CLEAN-ish, see CAND-2 residual]

## Items verified clean (known-bug checklist)
- deferred decode KV release: present upstream (on by default), drain-ack + timeout; timeout escape hatch = CAND-1.
- disaggregation admission abort: decode-side abort handled in abort_request for all four states (prealloc queue, transfer queue, retracted, waiting, running). [CLEAN]
- grammar double release: fixed upstream via to_FINISH single-path. [CLEAN]
- retract with SWA offload guard: fixed upstream (drain + synchronous backup). [CLEAN, but CAND-3 crash-on-guard]
- prefill abort cache leak (decode side of the seam): transfer-queue failure always releases prealloc'd KV via _release_request or deferred release. [CLEAN]

## Pass 4 — siblings of V4 (free-on-timeout-assumption, decode side)
Swept decode.py + common/conn.py for other "free on timeout" decode-side frees: none remain. All other timeouts (Bootstrapping/WaitingForInput prefill-wait, transfer failure) route through the Failed path → deferred release (with force_arm when metadata was partially published). Non-deferred backends (fake, opted-out) release immediately by design (no remote RDMA writer). Verdict: sibling sweep CLEAN.

## Admission arithmetic verdict (decode.py:1962-2156)
_allocatable_token_budgets + SWA variants: reservation-based admission; a mis-estimate shows up as the hard assert in _pre_alloc (scheduler crash), NOT silent corruption — the allocator itself enforces capacity. Budget arithmetic errors are thus availability bugs; flagged but not chased further (radix-sched owns the sibling capacity-check class on the running-batch side, board #16).

## Pass 5 — small-region sweep + reviews performed
- _resolve_pending_reqs / DP-rank prefetch: infrastructure only, no KV ownership. [CLEAN]
- _transfer_page_indices, host-receive payload slicing, _send_kv_metadata suffix-relative invariant (assists F13): cache_protected_len set to total_prefix_len (:1604) BEFORE send ⇒ staging scatter token_start matches the dst span written by _pre_alloc. [CLEAN]
- hisparse direct-to-host decode-side: alloc_for_decode_prealloc_hisparse + host slot registration + admit_request_direct seam covered-clean; remaining index math belongs to hisparse_coordinator/pools (dsa/pools teams). Verdict relayed to lead.
- Reviewed disagg-lead's V8 fix (board #21, APPROVED): drain_before_retraction bounded-wait; all_reduce uniformity verified; tests 3 passed in shared venv.

## Not deeply covered (handoff notes)
- HiSparse direct-to-host path (decode.py:2260-2278 alloc_for_decode_prealloc_hisparse, host pool rows): production runs DSA trtllm on DeepSeek V4, hisparse coordinator is adjacent; only lightly checked.
- _pre_alloc_host sidecar_pool_specs indices (extra pools from tree_cache.sidecar_pool_specs) reuse the same host_indices — assumed by construction.
- Who catches kv_receiver.abort() during prealloc queue abort while prefill might already hold destinations: Mooncake conn agent should verify abort_notified semantics (force_arm on partial publish seen at decode.py:2815).

## LANDING (2026-10-16 ~05:0x UTC)
- **#19 V4 fix MERGED → dev sha 8e731e894a.** Device-destination deferred KV release now holds past the 30s timeout until drain ack or peer-down declaration (prefill_info_table pop); new outcome value "peer_dead". Added hidden_kv_layers registration micro per lead #957.
- **#44 F12 final union MERGED in the same branch.** Decode-side `_handle_aux_data` (room ∈ request_status gate, per-message buffer_index/aux_index/item_len bounds) + codec exact-equality raise + mooncake-side try/except kept with mooncake's branch (#42 track). Dev-tip verification: test_mooncake_aux_data_bounds.py 10/10, deferred pair 49/49.
- Shared test file test_deferred_decode_kv_release.py = union of mooncake F4 worker-survive blocks and my V4 hold blocks; pre-submit rebase per wave-2 addendum done cleanly at 3d38a8f.
- mooncake #33 retry failed on pSwaStateTransfer test cross-drift (unrelated to my content). Their #37-prime rebase should take dev's file version as final.
- mooncake MR47 (#37+#41+#42+#55) merged → dev ac0f192; my independent second-witness dev-tip sweep green (235 passed, 0 failures) including the F12 final union and shared deferred test file. Disagg decode lane complete.

## FINAL COVERAGE VERDICT (disagg-decode lane, 2026-10-16)

### Production-surface coverage (own files, in-scope configs)
| Surface | Files | Verdict |
|---|---|---|
| Decode prealloc (page-64, fp8, chunked prefill, extra slots, failure/timeout) | decode.py alloc_for_prealloc*, extra_slots handling | COVERED — no freeing-path memory-safety bug beyond V4 (fixed); hisparse prealloc equal-by-construction = LOGICAL-ONLY, creator mapping unreachable on decode (features handoff F-5) |
| Deferred KV release (device + host-staged) | decode.py:2727-3027, KVPoll.Failed | FIXED (#19/V4, merged 8e731e894a): device path no longer frees at 30s without drain proof; holds for queue-full/agenda-retry and peer-down declaration; peer_dead metric. Host path was already correct. Timeout-hole confirmed live in v0.5.20+; older prod (0.5.14–0.5.17) lacks machinery (backport = release engineering) |
| AUX_DATA wire-controlled bounds (mooncake decode receiver) | mooncake/conn.py _handle_aux_data + common/utils.py AuxDataCodec | FIXED (#44+F12 union, live on dev ac0f192): codec exact-equality raise + handler room-gate/bounds + mooncake-side catch. 10/10 aux-bounds tests green |
| Retract with SWA/offload guard | decode.py retract path + offload manager | COVERED — verified guard + drain-before-retraction shipped by decode-offload-audit (#21 lead + #52 osprey 2198b3e8d1/fd9845e233) |
| Abort races (admission, transfer in flight, grammar double release) | decode.py abort/cleanup + schedule_batch mixin | COVERED — ordered resolve-before-free verified; abort-worker poison-recovery now worker-resilient (mooncake F3 land ac0f192); V4 closed the free-while-RDMA-in-flight device window |
| Offload manager host/GPU double free + read-after-free | decode_kvcache_offload_manager.py | COVERED — ownership + ongoing_offload gate verified; C6 linker-fail abort PP escape confirmed and fixed upstream lane (#67) |
| Role-switch / heartbeat / teardown interplay | role_switch.py, common/conn.py | CLEAN — is_fully_idle gates on pending deferred releases + ongoing offload; teardown fully resets per-manager tables; late daemon iteration harmless |
| Hisparse creator-mapping re-route | decode.py:2260-2278 + hisparse.py | CLEAN — worker-local; cross-worker mapping 'unreachable structurally'; alloc = LOGICAL-ONLY; zero-sentinel safety via buffer zeroing |
| Prealloc timeout leaves request state | decode.py + transfer worker pump | COVERED — states verified: retract/defer/free/WaitingForInput transitions; deferred + role-switch blocks destructive ops until resolved |

### Non-bug notes reaching dev/other teams
- hidden_kv_layers registration micro added at decode.py kv_args setup site (lead #957 mirror of prefill #30) — defensive for role-switch D→P with anchor main-V path.
- V7 ack-generation-register (mooncake receiver reuse rereg sets generation=0) shipped by mooncake (#37).

### Residual (handoffs)
- v0.5.14–0.5.17 production branches: no deferred-release machinery at all → the V4 fix concept (hold-vs-timeout) does not apply there; any timeout free on those versions is a different (older) code path — release-eng backport decision, not an upstream merge.
- hisparse host-row semantics question (prealloc device-buffer zeroing vs host pool rows) handed to pools/features (my F-5 answer: structurally unreachable from decode path).
- Mooncake/NIXL engine internals (batch indices, keep-alive, RDMA slot reuse) belong to their owners; where they touched wire/codec/test files I co-owned, the union was negotiated and verified.

### Deliverables shipped
- Board #19 (V4 deferred hold) → dev 8e731e894a; regression test_deferred_decode_kv_release.py + test_deferred_release_timeout_hold.py (49 tests).
- Board #44 (F12 aux-bounds union) → dev 8e731e894a; regression test_mooncake_aux_data_bounds.py (10 tests).
- Second-witness sweep of mooncake MR47 → dev ac0f192: 235 passed / 0 failed across the full shared suite list.
- Reviewed+approved: #21 (drs-before-retraction), mooncake F12 codec matrix vote (lead #595 via #587).
- Version matrix for deferred-release machinery: absent v0.5.14/.15p1/.17; present v0.5.20+ with the timeout hole fixed in dev (this branch).

Disagg decode lane: COMPLETE.
