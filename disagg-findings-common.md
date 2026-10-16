# Findings: disaggregation/common (staging_buffer, staging_handler, dcp_pack, bootstrap, conn shared)

Branch: agent/disagg-prefill-common. Owner: disagg-prefill-common.
Status codes: CANDIDATE (reasoning shown, needs repro/second review), VERIFIED, REFUTED.

---

## F1. CANDIDATE — race: late staging CHUNK_READY scatters into KV pages concurrently with abort/teardown freeing them
- File: staging_handler.py:297 (handle_chunk_arrived) -> :229 (submit_chunk_scatter) -> :393 (_scatter_region); teardown at :180 (unregister_decode_req/release_room).
- Threads: decode_thread (arrival path) vs scheduler main thread (unregister/release + page free in scheduler abort).
- Interleaving:
  1. decode_thread: `handle_chunk_arrived(room)` -> `_room_to_receiver.get(room)` != None; all writers arrived.
  2. main thread: `unregister_decode_req(room)` pops room, `release_room` syncs scatter_stream and frees staging allocs; scheduler abort path then frees the request's KV pages back to the pool for reuse by another request.
  3. decode_thread: `submit_chunk_scatter` ran BEFORE the pop or concurrently: `_room_to_decode_req.get(room)` still returned decode_req -> `_scatter_region` reads `req_to_token_pool.req_to_token[req_pool_idx, token_start:token_end]` and launches scatter kernel writing into those page indices.
  4. If pages/req_pool row got freed and REUSED between (2) and the kernel execution... Note: release_room does `stream.synchronize()` before freeing staging allocs, which drains the kernel; but the KV pages written are the decode request's own pages — writing to them is only harmful after they are freed/reassigned. `stream.synchronize()` in release_room ensures the kernel finished before unregister returns; the question is whether the scheduler frees the KV pages BEFORE calling unregister_decode_req.
- What breaks: cross-request KV corruption (scatter lands in reassigned pages) IF any teardown path frees KV pages before unregister_decode_req drains the scatter stream. Needs decode.py abort ordering to confirm exploitability (parent: disagg-prefill, decode.py owner).
- Also: `_scatter_region` reads req_to_token row from a snapshot AFTER teardown began: token_start from `cache_protected_len + page_start*page_size` — if retraction/abort shrank the alloc, kv_indices may be 0s -> scatters into page 0 (reserved?) or wrong pages.

## F2. CONFIRMED-BY-READING leak — _wm_subscribers grows forever
- staging_handler.py:121 (_wm_subscribers) never pruned on room teardown; every watermark free broadcasts ZMQ to every dead prefill session (exceptions swallowed). Slow leak + wasted sockets per request.

## F3. CANDIDATE — decode-side staging is shared ring across ALL rooms; `handle_staging_req` allocates with no per-room quota: a malicious/buggy room can pin the whole ring (watermark never advances), deadlocking other rooms' chunks until 30s timeout. Availability; matches known class "admission abort".

## F4. REFUTED — compute_head_slice_params src>dst head OOB: in src_attn_tp>dst_attn_tp branch, src_heads_per_rank <= dst_heads_per_rank always (src_tp>dst_tp), so head_offset+num_heads*head_dim stays within stride. No wrap overflow.

## F5. CANDIDATE — `_gather_all_layers_torch`: per call allocates `page_idx_tensor` + expanded `gather_idx` (num_tokens*num_heads*head_dim*8 bytes int64!) — for 32768-token chunk, 8 heads, 128 dim = 32K*8*128*8 = 268MB int64 gather index on EVERY chunk transfer. OOM risk under concurrency; also `torch.gather(out=)` with overlapping? Not memory-unsafe per se, but huge transient alloc in transfer worker.

## F6. CANDIDATE — PrefillStagingContext.prefetch_requested keys (room, chunk_idx, session_id) grow forever per room; also `prefetched_rooms` never pruned (only bounded by room reuse 2^63; slow leak).

## F7. CANDIDATE — `free()` empty-ring branch sets `head=0` and `round+=1`, and watermark=(new round, 0). A stale WATERMARK comparison in prefill `handle_watermark_msg` uses tuple compare `(round,tail) > prev` — fine. But `is_watermark_ready` returns True immediately for alloc_round<=0 — if allocator rounds ever wrap to negatives, no. Skip; kept for completeness.

## F15. CANDIDATE (logic verified; config trigger identified) — get_mha_kv_ptrs_with_pp draft-model branch silently skipped when dst_total % num_kv_layers == 0 -> main K/V ptr mixup, silent garbled V
- conn.py:1556-1579.
- Branch selection:
  - if num_kv_layers == dst_num_total_layers: fast path.
  - elif `num_kv_layers < dst_num_total_layers and dst_num_total_layers % num_kv_layers != 0`: draft-model layout [K_main,V_main,draft_K,draft_V].
  - else: 'PP equal or 1' layout [K...,V...]: dst_v_ptrs = dst_kv_ptrs[dst_total+start_layer : dst_total+end_layer].
- Trigger case A (wrong V pointer set): prefill has NO spec decoding, decode HAS draft (EAGLE/DSPARK). e.g. main=4 layers: src_kv_ptrs=8 -> num_kv_layers=4; dst has main+draft=8 layers -> dst_total=8 (16 ptrs). 8 % 4 == 0 -> ELSE branch: dst_v_ptrs = dst_kv_ptrs[8:12] = draft_K pointers; dst_k_ptrs = dst_kv_ptrs[0:4] = main K (ok by luck). Result: main V data is RDMA-written into the DRAFT K buffers; main V and draft V never transferred -> decode serves stale/uninit V for every layer -> massively garbled output with no error. Only the modulo!=0 quirk saves other shapes.
- Trigger case B (silent slice truncation): else branch indexes dst_kv_ptrs[dst_total+start_layer : dst_total+end_layer]; with start_layer>0 (prefill PP>1) this can exceed len(dst_kv_ptrs); Python slicing truncates silently -> len(dst_v_ptrs) < len(src_v_ptrs).
- Propagation: _get_mha_head_slice_blocks (conn.py:1616) does entries = zip(src_k+src_v, dst_k+dst_v, ...) — zip drops the shortest side silently, so missing layers are never transferred and no exception is raised. Decode reads stale buffer leftovers from a previous request that owned those pages -> cross-request leakage / garbled output.
- Production relevance: staging path (SGLANG_DISAGG_STAGING_BUFFER) + spec-decoding draft models: listed as possible in our deployments; DeepSeek V4 Pro + DSPARK on v0.5.15.post1 is a candidate config if staging is enabled for non-MLA there.

## F16. CONFIRMED-DEAD-CODE-RISK — _get_mha_head_slice_blocks zip() silently truncates layer lists (sibling of F15): any ptr-count mismatch in get_mha_kv_ptrs_with_pp or get_mla_kv_ptrs_with_pp variants becomes silent partial transfer.

## F12. SEAM finding — AuxDataCodec.deserialize_data_to_buffer (common/utils.py:101): dst_addr = aux_ptr + item_len*aux_index with NO bounds or length validation of wire-provided aux_index/data. Callers: mooncake/conn.py:1815-1824 takes aux_index = int(msg[3]) from ZMQ metadata msg. A stale/duplicated AUX_DATA message after metadata-buffer slot reuse writes hidden_states/output tensors into a slot owned by ANOTHER request -> cross-request metadata corruption. Coordinate w/ disagg-mooncake: msg has no room-generation tag. Known-class sibling: 'KV integrity tagging'.

## F8. check_status race window (conn.py:564 check_status raises KeyError if room cleared between poll and check) is guarded by callers holding conclude_state; _clear pops request_status then a concurrent poll() via check_status[room] -> KeyError crash of scheduler thread? poll() reads conclude_state first, but update happens on other threads (apply_prefill_status from bootstrap thread). Only if clear runs on scheduler and poll on scheduler same thread -> safe. KEPT as watch item, not a finding.

## F13. CANDIDATE (needs decode-side invariant check) — staging CHUNK_READY page_start is request-suffix-relative (mooncake/conn.py:563 sends kv_chunk.index_slice.start), and _scatter_region does token_start = cache_protected_len + page_start*page_size. This is only correct if dst_kv_indices in the prefill's wire metadata EXCLUDES the decode cached prefix pages. If decode ever publishes full indices (or prefix changes after retraction), token_start overshoots the request's allocated req_to_token row; the slice then reads page indices of whatever occupies the row tail (stale indices from a previous owner are 0s or another request's pages) -> triton scatter writes KV into WRONG pages -> cross-request KV corruption, silent. Also unguarded: token_end > row length is never checked in _scatter_region (staging_handler.py:415). Production: page 64 + staging enabled.

## F9. minor — early-policy receivers: poll()/clear()/abort_notified read from scheduler thread; _run_prefill_recompute (executor thread) can call abort() concurrently when PD-rebootstrap retraction is used WITHOUT defer_decode_allocation policy: _lifecycle_lock is nullcontext then. conclude_state/aborts race benign-ish (double abort idempotent, record_failure under lock), but abort_notified flag check-then-set unsynchronized -> duplicate ABORT notifications to prefill (harmless by design).

## F17. watch — FastQueue has no sentinel/drain on shutdown; worker threads block forever on get() (known class 'Mooncake transfer worker keep-alive'); not my file's owner but FastQueue is mine: put-with-sentinel pattern absent. Ref: 'Mooncake transfer worker keep-alive' known bug class.


## F15 UPDATE — verified arithmetic (branch selection wrong whenever decode draft entries make dst_total divisible by num_kv_layers; and in the draft branch, v_ptr_offset = num_kv_layers*multiplier_ratio points into draft_K for the canonical [K_main,V_main,draft_K,draft_V] layout when start_layer=0). Both branches mis-place V for "prefill without spec, decode with EAGLE/DSPARK draft" + staging head slicing. Case C (PP2 stage, no draft) is correct. The draft branch condition reads inverted: it requires dst_total % num_kv_layers != 0, but exactly divisible draft layer counts are natural. See verification script output in commit history message. NOTE: head-sliced path only runs under SGLANG_DISAGG_STAGING_BUFFER / heterogeneous TP non-MLA.

## F18. CANDIDATE — DeferredBootstrap.fail(room) CREATES a tombstone for any room id (`_room()` setdefault), and once failed, register() for that room returns failure forever until the ~35s tombstone expiry. An ABORT frame is unauthenticated; any decode instance (or a stray/record-replayed ZMQ message) with a guessed/stale room id can poison a room before the legitimate decode bootstraps to it -> the prefill then fails a healthy transfer. Also fail-on-cleared-room: _handle_bootstrap_message ABORT path fails bootstrap but returns False so backend still processes. Impact: failed transfers, possible cross-request cross-talk via room id reuse. Known-class sibling: deferred bootstrap state machine.

## F19. CANDIDATE (handoff to disagg-mooncake) — decode._defer_release sets required_acks = len(bootstrap_infos); for MLA, bootstrap_infos includes is_dummy=True entries (one per target_tp_rank) that never send ABORT_ACK -> acks count can never reach required_acks -> deferred release only resolves via timeout (SGLANG_DISAGGREGATION_WAITING_TIMEOUT+...), at which point prefill RDMA write may still be in flight -> the '30s timeout frees KV while RDMA write in flight' class. If dummy ranks already excluded upstream, refute.

## Coverage notes (conn.py shared, staging, bootstrap, dcp_pack):
- StagingAllocator ring+watermark protocol: hand-verified for wrap/overcommit/empty-reset; prefill-side gate is consistent; no overlap proof bug found inside common/ (the O(num_allocs) alloc_order list is O(n^2) under load -- perf note only).
- DeferredAbortAck generation scheme: sound; room-reuse resets in update_status; acks counted in a set -> duplicate ACKs harmless.
- bootstrap.py DeferredBootstrap: open/complete/close owner checks OK; expiry cleanup OK.
- dcp_pack.py: bound checks present (fits + capacity); pack_offset+capacity semantics flagged (F-DCP-1 lowered: caller passes pack_offset_bytes=0 today; watch).
- common/utils.py FastQueue/pack/unpack: safe; AuxDataCodec OOB = F12.
- compute_grid_segments/staging_grid_tokens: arithmetic checked, aligned; no off-by-one found.
- gather/scatter kernels: masks guard tails; head-slice OOB refuted (F4); F5 OOM-class transient gather_idx int64 expansion (268MB @32k tokens x 8 heads x 128 dim) kept as candidate (memory blowup in transfer worker).

## F18 RESOLUTION (in this branch): register-first refusal landed in bootstrap.py fail() — an ABORT for a never-registered room is dropped+logged, no tombstone; known rooms keep cancel semantics. Tests adapted in test_deferred_bootstrap.py + dedicated refusal test. Rationale: every legit flow has open() (sender ctor) before any decode can ABORT the room (decode bootstraps only via prefill-registered bootstrap entries), so pre-cancel tombstones only ever consumed stale frames.

## FOLLOW-UP DESIGN (deferred per disagg-lead): room-integrity generation token for BOOTSTRAP/ABORT frames.
Key choice points: (1) minter: decode receiver, when it first registers to a room, alongside bootstrap_room; (2) wire placement: extra trailing frame on BOOTSTRAP and ABORT multipart messages (old peers ignore trailing frames -> wire-compatible rollout; same pattern as the optional generation frame already on AbortNotification); (3) prefill stores (room, token) at DeferredBootstrap.register and requires match on fail(); (4) cross-version: absence of token = legacy behavior (accept), presence-with-mismatch = drop + warn; (5) token source: 64-bit random per request (NOT the room generation counter, which increments per decode instance lifecycle and is guessable). Defer until fix wave clears.

## ASYMMETRY NIT (lead-approved, follow-up): AUX_DATA out-of-bounds -> dropped (decode thread lives; request dies by waiting_timeout), whereas KV geometry mismatch -> room Failed immediately. The aux path lacks room->receiver plumbing to fail fast. Note for later hardening: pass the room's receiver into _handle_aux_data or key aux buffers per room.

## F20. FIXED (this branch) — StagingAllocator.assign accepted required_bytes<=0: a wire-decoded negative page count (STAGING_REQ chunk_num_pages) would regress ring head -> same-round overlap of two live allocations -> cross-request staging corruption. Guard: raise ValueError on non-positive size (staging_buffer.py). Test: test_allocator_rejects_nonpositive_size.
## F21. FIXED (this branch) — negative chunk_idx wire aliasing: STAGING_REQ/STAGING_RSP chunk_idx<0 would index per-chunk lists from the END (Python negative indexing), silently overwriting another chunk's (alloc_id, offset, round, end) — prefill gathers into a region owned by a different allocation -> cross-request KV corruption; verified repro (set_chunk(-1) overwrites chunk 1). Guards: ValueError in StagingTransferInfo.set_chunk (+<0 drop at handle_staging_rsp, STAGING_REQ field drop in handle_staging_req), WATERMARK out-of-range drop. Tests: 6 in test_aux_codec_bounds.py TestStagingWireFieldGuards.
- Frame handler exposure note: the corrupted-frame precondition is a miswired/buggy peer or pointer corruption in ZMQ frames (unauthenticated intra-cluster fabric — known trust boundary). Guards convert silent corruption into drop+warn.
- VERIFIED live-reachability at v0.5.x? STAGING_REQ/RSP only exist on staging-enabled builds; production flag SGLANG_DISAGG_STAGING_BUFFER may be enabled for non-MLA. Version tags: run version-triage separately (staging module present HEAD; likely absent pre-0.5.16).

## PD TRUST-BOUNDARY CHECKLIST (lead sweep item 2) — wire-controlled fields in common/bootstrap.py + common/staging_handler.py (+ common/conn.py parsers + common/utils.py codec surfaces), each marked CHECKED (validated in code) or CONSTRAINED (provably bounded by protocol state) or OPEN:
| Frame / surface | Field | Verdict | Where |
|---|---|---|---|
| BOOTSTRAP (_handle_bootstrap_message) | room | CHECKED: unknown rooms refused (F18); else sender-owned | common/bootstrap.py |
| BOOTSTRAP | endpoint (ip, port) | CONSTRAINED: only used as ZMQ status-notify target; bogus -> send fails, no state | common/bootstrap.py + conn.py _notify_bootstrap |
| ABORT | room | CHECKED: same as BOOTSTRAP room (F18) + backend drains via generation ACKs | conn.py + bootstrap.py |
| STAGING_REQ | room | CONSTRAINED: must exist in room_receivers (populated at prealloc) else dropped; <0 dropped (F21) | staging_handler.py handle_staging_req |
| STAGING_REQ | chunk_idx | CHECKED: <0 dropped (F21); large idx -> ALLOC grows list to idx on DECODE receiver.chunk_staging_infos (bounded by prefill-announced chunk count? NOT bounded by dst page count within this fn — OPEN low: unbounded list growth if idx huge, DoS only) | staging_handler.py |
| STAGING_REQ | chunk_num_pages | CHECKED: <=0 dropped (F21); huge -> required>total -> ALLOC_OVERSIZED (bounded) | staging_handler.py |
| STAGING_REQ | session_id | CONSTRAINED: only routed back in RSP; string | staging_handler.py |
| STAGING_RSP | room, chunk_idx | chunk_idx CHECKED (F21) room: tinfo lookup miss -> warn+drop | staging_handler.py handle_staging_rsp |
| STAGING_RSP | offset/round/end | CHECKED-by-protocol: consumed by check_ready against allocator state via watermark; gather bounds = fits()+watermark; corrupt values -> watermark refuse (prefill defers) | staging_handler.py |
| WATERMARK | round/tail/session | CHECKED: <0 dropped (F21); monotonicity enforced by tuple compare (can only advance); session free-form (new sessions land in dict — OPEN low: unbounded session keys DoS) | staging_handler.py |
| CHUNK_READY | room | CONSTRAINED: room must be registered (staging_room check) | staging_handler.py handle_chunk_arrived |
| CHUNK_READY | page_start/num_pages | CLOSED w/ assumption (lead): token window relies on suffix-relative dst_kv_indices invariant (F13); in-pool by construction; NOT bounds-checked in _scatter_region | staging_handler.py _scatter_region |
| CHUNK_READY | writer_id | CONSTRAINED: fan-in counter only | staging_handler.py |
| AUX_DATA (codec, common/utils.py) | buffer_index/aux_index/len | CHECKED post-#42 union: exact-equality + bounds incl buffer_index + aux_data_lens fold | mooncake/conn.py + common/utils.py |
| kv status msg (parse_kv_status_message) | room/status/rank/reason | CHECKED: room must exist in request_status else dropped; status enum-compared; rank counted in set; reason string only recorded | common/conn.py |
| ABORT_ACK | room/rank/generation | CHECKED: generation must match armed generation; ranks in set | common/conn.py note_abort_ack |
| DeferredBootstrap register frame (decode->prefill BOOTSTRAP endpoint) | room/endpoint | CHECKED: register() rejects second endpoint for a room (no replace) | common/bootstrap.py |
Remaining OPEN-low items (documented, not gated): huge chunk_idx list growth on decode chunk_staging_infos; unbounded watermark session keys; CHUNK_READY duplicate frames can accelerate fan-in (scatter of partially-written staging region — integrity nit, trust-boundary).

## Fresh-eyes challenge: decode_kvcache_offload_manager.py (lead item 3) — CLEAN with 2 nits, assumptions listed & broken-or-held:
- A1 req identity: weak refs claimed; ongoing_offload/ongoing_backup tuples hold Req/rid strongly until ack — WeakKeyDict never silently disappears mid-flight. HOLDS.
- A2 retract arithmetic: after retraction, incremental_new can go negative; floor-alignment yields negative aligned_len != 0, causing empty-slice writes (start>end) — HARMLESS but should have been blocked by an explicit <=0 guard. NIT-1 (perf/noise).
- A3 retract state reset: OffloadedState.inc_len is never reset on retraction; after retract-into-offloaded-zone + regrowth, the regrown region's KV is never offloaded (coverage HOLES) and old host/storage entries linger under old hashes. Content-addressed hashing keeps correctness (new continuation has different token hashes; identical regrowth yields byte-identical KV). VERDICT: no corruption path found — covered content addressing is the safety net; still a coverage nit. NIT-2.
- A4 write_storage exception path (host pin leak / scheduler crash): cache_controller.write_storage is an unbounded queue put + id return — cannot raise. HOLDS (leak only if the storage background thread dies — hicache keep-alive class, their scope).
- A5 MIN all_reduce queue drain: MIN bounds pops to the slowest rank's queue len; no over-pop possible. HOLDS.
- A6 drain_before_retraction MAX all_reduce uniformity: covered by V8 review (disagg-lead/disagg-decode). Not re-challenged here.
Verdict assumptions to challenge later by someone else: content-addressed hashing covers misfires (A3); unbounded ZMQ in-memory backup queue never back-pressures (A4).
