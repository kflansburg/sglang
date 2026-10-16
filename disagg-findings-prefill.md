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
