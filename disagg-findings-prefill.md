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
