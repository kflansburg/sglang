# Disagg audit — scheduler integration seams (disagg-lead)

Files read: python/sglang/srt/managers/scheduler.py (disagg branches), cross-refs into decode/prefill queue names.

## Seam map (scheduler.py)
- `_add_request_to_queue` (~3392): NULL->waiting_queue; PREFILL->_prefetch_kvcache+disagg_prefill_bootstrap_queue.add; DECODE->disagg_decode_prealloc_queue.add(req, is_retracted).
  - NOTE: PREFILL branch calls _prefetch_kvcache BEFORE bootstrap_queue.add; aborted-before-bootstrap reqs need finish(ABORT) + waiting sweep to clear prefetch (retire_unadmitted_request handles; abort_request PREFILL branch calls _release_aborted_request + sender.abort).
- `_release_aborted_request` (~3457): release_mm_inputs_on_abort + tree_cache.finish(req.cache_request_handle, ABORT) + session abort. Does NOT touch KV indices (expects queue-specific handlers).
- abort_request (~5516):
  - waiting_queue pop: DECODE branch -> discard_kv_cache_backup(host_pool) if host_receive_threshold<1 + release_kv_cache(checkpoint=True); PREFILL -> release_aborted_prefill_waiting_req(req); mamba+non-disagg -> release_kv_cache(checkpoint=False).
  - PREFILL bootstrap queue abort (~5587): iterate disagg_prefill_bootstrap_queue.queue, _release_aborted_request + req.disagg_kv_sender.abort() (+pp prepare_abort) but does NOT remove from queue — relies on queue pop filtering finished. VERIFY in prefill.py (owner: disagg-prefill): (a) aborted req popped without double release; (b) no finished_reason set on non-pp path? Actually _release_aborted_request doesn't set finished_reason; pp branch sets prepare_abort. If finished_reason never set on non-pp, bootstrap queue may still try to bootstrap an aborted req -> two abort() calls or KV send for aborted req (cache leak). CANDIDATE — hand to disagg-prefill.
  - DECODE transfer queue abort: host_staged -> prepare_abort + `continue` (keeps host dest alive until prefill stops writing — relies on drain-ack, check CommonKVReceiver._send_abort_notification). Non-staged -> kv_receiver.abort() arms drain-ack first (comment says so).
  - retracted_queue abort: discard_kv_cache_backup + send_output. Queue rebuilt. OK shape; verify discard_kv_cache_backup frees exactly once even when backup partially staged (owner: disagg-decode).
- pause_generation retract (~5658): decode waits include reqs with kv.holds_kv staged into retract_reqs; PREFILL chunked_req tmp_end_idx reset; note PREFILL branch skips merging last_batch (comment: avoids leak since prefill event loop never calls update_running_batch).
