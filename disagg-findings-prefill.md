# Disaggregation audit — prefill side (prefill.py, utils.py, kv_events.py, checksum.py, common/)
Owner: disagg-prefill. Repo @ origin/main as of 2026-10-16.

## Coverage checklist
- [x] bootstrap queue lifecycle (add/pop/abort/failure)
- [x] abort paths: bootstrap queue, waiting queue, chunked req, inflight queue
- [x] inflight queue: send completion vs radix unlock ordering
- [ ] KV send completion vs prefix eviction during Transferring (deep: conn.py sender thread)
- [ ] chunked prefill: chunk boundary off-by-one, start_send_idx monotonicity
- [ ] grammar double release
- [ ] checksum.py: tagging gaps
- [ ] kv_events.py
- [ ] utils.py: MetadataBuffers, ReqToMetadataIdxAllocator
- [ ] SWA/mamba/DSA payload index translation
- [ ] staging buffer lifetime

## FINDING CANDIDATES

### C1 (lead's candidate, scheduler.py:5585 bootstrap-queue abort) — APPEARS SAFE on latest
Non-pp abort of a bootstrap-queue req: `_release_aborted_request` + `sender.abort()`
sets `conclude_state=Failed`; the req stays queued but next `pop_bootstrapped` polls
Failed -> `handle_bootstrap_failure` (prefill.py:1309) which releases KV (guarded by
holds_kv), frees metadata buffer, prepare_abort. Double-abort of sender is guarded
(`conclude_state` sticky Failed; abort() re-records failure message, harmless).
Metadata buffer: bootstrap-queue reqs usually have metadata_buffer_index == -1
(allocated at pop). No free of an index another req owns. VERDICT: not a bug upstream;
window is bounded by one scheduler step.

### C2 — inflight-queue abort has no finished_reason until next poll; safe by poll conclusion
abort_request (scheduler.py:5597) inflight branch only calls sender.abort(). Next
process_disagg_prefill_inflight_queue polls Failed -> handle_inflight_transfer_failure
releases KV with checkpoint=True (KV data itself is valid; decode-side failure does not
corrupt prefill's pages) and prepare_aborts. No double-release: holds_kv guarded.
Note: metadata buffer freed in the done_reqs loop (line ~1230) for all done reqs — ok.

### C3 — pending_chunk_rids cleared on all retire paths checked so far
send_kv_chunk(last_chunk=True) discards; _retire_aborted_prefill_result,
process_pending_chunked_abort, handle_bootstrap_failure all call clear_pending_chunk_send.
optimistic_release_and_requeue also clears. OK.

## Round 2

### C4 (OPEN, handed to conn.py/staging sub-audit): in-flight middle-chunk RDMA reads vs abort freeing pages
Path: chunked req, non-pending sender, overlap: middle chunks sent via
send_kv_chunk(last_chunk=False) in process_batch_result_disagg_prefill (prefill.py:1076).
Then AbortReq -> process_pending_chunked_abort (scheduler.py:3690): sender.abort()
marks Failed; clear_pending_chunk_send drops the rid WITHOUT waiting for posted
middle-chunk transfers; KV released shortly after via _retire_aborted_prefill_result
(release_kv_cache). If the transfer engine has not finished RDMA-reading the sent
pages, freed pages may be reused by other requests -> transfer reads NEW owner's data
-> cross-request KV leak to decode (which by then is itself aborting, so impact is
bounded but nonzero: decode may still have live grammar prealloc; also prefill-side
cache checkpoint=False so pages go straight back to allocator). Needed: confirm
whether CommonKVSender.abort() drains/cancels outstandings synchronously before the
scheduler proceeds. @disagg-prefill-common is tracing conn.py for this.
The same class applies to handle_inflight_transfer_failure (30s timeout abort while
write in flight on decode side) — decode-side KV lifetime after failure is decode's.

### C5 checksum stream ordering — appears safe
send_kv_chunk(last_chunk=True) (prefill.py:1455) computes adler32 over the full
req range [0, end_idx) on GPU AFTER copy_done.synchronize() in
process_batch_result_disagg_prefill; forward KV writes precede on the same stream.
Coverage caveat: checksum covers ALL pages incl. decode_prefix_len prefix (which was
not transferred in this req) — decode must checksum its OWN copies of those pages;
if decode's cached prefix pages were corrupted before this req, checksum compares
different bytes than prefill intended but that IS the point (integrity of whole
context). Not a gap. Health-check reqs skip checksum (value 0) — fine.

### C6 ReqToMetadataIdxAllocator.free — no double-free guard, but all call sites
(maybe_release_metadata_buffer, prefill.py:163) guard via req.metadata_buffer_index >= 0
then set -1. Single-threaded use. No bug found.

### C7 kv_events.py — event-publishing machinery only (BlockStored/Removed msgs to router);
not in KV data path; publisher thread only reads pre-built msgspec structs. Low risk,
not deep-audited for memory safety of KV (it has none).

### C8 grammar queue abort on PREFILL (grammar_manager.abort_requests ->
set_finish_with_abort) — enters bootstrap add() -> is_unadmitted_reject ->
retire_unadmitted_request: releases handle, streams, no sender created, no KV held
(origin_input_ids stubbed to 1 token, no cache match yet). Double release guarded by
release_mm_inputs_on_abort nil-ing multimodal_inputs. No double-release found here.

### C9 update_finish_state/finished_reason races: none; scheduler-thread only.

## Round 3 (2026-10-16 ~03:40)

### F15 VERIFIED numerically (sub-agent finding, re-derived)
get_mha_kv_ptrs_with_pp (common/conn.py:1556-1614). Called from mainstream
mooncake/conn.py:1104 (non-MLA GQA/MHA, e.g. Gemma), nixl/conn.py:945,1845,
ascend, staging head-slice.
Trigger: prefill WITHOUT draft, decode WITH draft, D % M == 0, D > 0
(D = decode draft layer count, M = main layers, per-PP-stage view).
Branch miss -> dst_v_ptrs = dst_kv_ptrs[dst_total+start:dst_total+end] =
draft_K block. Lengths match (all len M) so layers_params/zipping do NOT catch
it. Consequences: (1) main V RDMA'd into draft K buffers (garbled speculation),
(2) decode main V NEVER written -> serves previous request's V pages =
silent cross-request KV leakage. D<M (normal EAGLE) escapes only arithmetically.
PP>1 + draft: else-branch slice can overrun the ptr list -> zip truncation.
Fix plan (board #30): explicit draft handling via num_draft_entries on both
sides + fail-loud length asserts before pairing.

### V4/F9 fix implemented (board #31, agent/disagg-prefill)
(a) decode-side CommonKVReceiver._send_abort_notification: per-rank failure
tracking + poll-driven bounded retry (0.5..8s, 6 attempts), ERROR on give-up;
pages then stay held per decode's V4 hold-forever path.
(b) prefill-side CommonKVManager._send_abort_ack (same class, from mooncake's
review note): failed drained ACKs queued in _pending_abort_acks with the same
retry discipline; poisoned rooms' pending ACKs dropped.
Regression test: test/registered/unit/disaggregation/test_abort_notification_retry.py.

### Known-class test verification (lead's request)
- deferred decode KV release: test_deferred_decode_kv_release.py EXISTS; code
  guard = drain-ack + poison machinery (common/conn.py register_deferred_abort_room,
  note_abort_ack, poison_deferred_ack_room) verified present.
- admission abort: test_admission_abort_not_enqueued.py EXISTS; prefill door
  is_unadmitted_reject -> retire_unadmitted_request verified in PrefillBootstrapQueue.add.
- prefill abort cache leak: test_prefill_abort_result_cleanup.py EXISTS; code
  guard _retire_aborted_prefill_result + release_aborted_prefill_waiting_req
  verified. Sibling gap: (a) aborted BOOTSTRAP-queue req with metadata buffer
  pre-reserved? bootstrap queue reqs don't reserve until pop -> no sibling. 
  (b) SWA variant of abort release: release_kv_cache handles swa_evict_floor
  via tree cache; not re-verified deeply (mem_cache team's area).
- grammar double release: covered in test_prefill_abort_result_cleanup.py per
  lead; code path verified by me (C8): single to_finish path, mm release
  idempotent (None-ing). No sibling found in my files.
- Mooncake transfer batch indices / keep-alive: disagg-mooncake territory; their
  F-tests exist (test_mooncake_transfer_batching.py).
- KV integrity tagging (checksum): test_kv_checksum.py EXISTS; checksum.py
  verified: compute covers full seq incl. decode_prefix_len; health-check
  reqs use value 0. Siblings: XLA/other backends call set_kv_checksum? Only
  prefill.py:Send_kv_chunk calls set_kv_checksum -> decode reliance on it is
  the config bit disaggregation_enable_kv_checksum; F12 (AuxDataCodec wire
  aux_index OOB) is the real sibling in this class -> flagged to mooncake.
- paged allocator capacity check: not in my files (core-pools team).
- NIXL reload invalidated peer: refuted upstream (nixl's re-trace, lead accepted).
- DSA tail req_pool index: get_dsa_tail_state_indices (utils.py:1208) checks
  pool_size>1, tail>=pool; n_valid wraps correctly. Index list
  [req_pool_idx, start_phys, first_n, 0, second_n, tail_size] bounded by
  construction (start_phys < tail_size; first_n+second_n = n_valid < pool_size
  <= tail_size). No OOB found. Sibling: disagg_kimi (K3) QSA pending state
  indices = req_pool_idx only; req.kv.req_pool_idx is int (validated non-None).
- disagg retract with SWA offload guard: decode-side, verified by disagg-decode.
- HiCache host reclaim order / host pool free-list double free: hicache team.
- unified pool move_kv_cache OOB: unified memory gate (utils.py:1418)
  verified clearing paths (C3).

### C4 residual question to lead: does production set
SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE=1? If OFF, abort of a chunked
req can free prefill pages while a mooncake worker is mid-RDMA-read on a
dequeued middle chunk (worker drops only not-yet-started chunks on Failed).
Impact bounded (room Failed -> decode discards) but late reads of reissued
pages still possible on prefill side's own caches... actually only reads, so
worst case is wrong bytes sent for an aborted room = harmless. Withdrawn
unless someone shows decode-side reuse; the real teeth are F8/F9 (fixed).

## Round 4 (2026-10-16 ~04:00)

### Fixed & in review
- F15 (board #30): get_mha_kv_ptrs_with_pp rewritten; anchor = kv_args.hidden_kv_layers
  (populated in prefill.py from model_config.num_hidden_layers). NOTE for backports to
  v0.5.14-17 (before hidden_kv_layers existed): plumb the draft count at the
  registration site from dst ptr counts instead. All prod versions .14..HEAD affected.
- F9/V4-prefill (board #31): abort-notify retry + drained-ACK retry queue.
- F12 codec side (board #43): AuxDataCodec bounds check. Handler side -> mooncake.

### pp>1 abort double-finish (lead's open item a) — LOW, no memory unsafety
pp>1 abort marks finished_reason on reqs that stay queued; pop paths then conclude
via handle_bootstrap_failure which re-runs prepare_abort (overwrites reason) and
re-streams output. tree_cache.finish(ABORT) re-entrant: release_aborted_request is a
no-op for plain radix; unified's is pop-if-present (verified style). Result: duplicate
abort responses to tokenizer (tolerated; tokenizer dedups per-rid retire) and a
slightly wronger finish message. Not memory-unsafe; recommend later dedup via
"if req.finished_reason is not None: skip re-prepare" in handle_bootstrap_failure.

### Env/test status
Shared env picked up via sandbox_reset. test_utils import breaks collection
(datasets missing; lead adding). My three new test files avoid test_utils;
all pass locally: mha 7, abort-retry 4, aux-codec 5.

### checksum deep-check (continuation of C5) — VERIFIED consistent
- SWA state: checksum's translate_loc_from_full_to_swa == payload's
  translate_swa_indices_for_transfer (allocator/swa.py:260 delegates) — same mapping.
- Range symmetry: decode verifier (decode.py:3198) calls the SAME
  page/state_indices_for_request helpers with seq_len = full input len; prefill
  computes with end_idx = min(extend_end, len) == len at last_chunk. Symmetric. OK.
- Strengths/coverage: catches transport corruption + prefix-cache-content drift;
  health-check reqs tagged 0 and skipped both sides; expected==0 also skips
  (non-checksum backends/health checks indistinguishable — acceptable).
- F18 verdict: LOW/by-design (see #disagg post) — tombstone-before-open is
  intentional; wire-generation token is the real fix; deferred to hardening list.
