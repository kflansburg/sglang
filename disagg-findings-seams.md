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

## Verified-bug ledger (as of 03:22)
V1 nixl reload race — REFUTED at code level (complete peer sets ⇒ same worker; gate nixl/conn.py:3261, key 2956-2960). Fix = invariant comment/assert.
V2 nixl send_aux count-only assert. FIXING (nixl).
V3 nixl _await_handles unbounded healthy wait ⇒ bounded 300s (env tunable) with distinct failure. FIXING (nixl).
V4 COMPOUND: best-effort one-shot AbortNotification (common/conn.py:2608-2629) × decode device-path deferred release frees at 30s w/o drain (decode.py:2989-3008). FIXING (prefill F9 retry/fail-all; decode F8 device parity w/ host path).
V5 mooncake silent truncation transfer_worker (conn.py:2556-2573) ⇒ fail loud. FIXING (mooncake).
V6 mooncake registration ret ignored ⇒ fail-fast. FIXING (mooncake).
V7 mooncake worker death no supervision ⇒ supervision/keepalive. FIXING (mooncake).
V8 decode drain_before_retraction raise in scheduler OOM path (decode_kvcache_offload_manager.py:302; scheduler.py:4399) — CANDIDATE (availability).
V9 mooncake maybe_send_extra silent truncation on drift except QSA/SWA_RING/DSV4_REQUEST_STATE — pending which states are production.

CLOSED-clean: bootstrap queue abort chain; inflight queue abort chain; grammar double-release decode; retract+SWA offload guard; nixl defer-ack accounting (F4); is_abort_release_safe truthful nixl-device/unreachable nixl-host; mooncake: batch indices pairing guard, Failed-terminal, drain-ack protocol, peer re-reg ordering, 0-len/dedup registration, int32 wire dtype.

## Cross-team handoffs (incoming)
- pools/dsv4 F2: DSA compress-state PD transfer indices (deepseek_v4_compress_state.py:87-132) use wire-controlled req_pool_idx/seq_len unclamped → OOB row math (crash; under graphs possibly cross-request read). All prod tags since v0.5.15. Part of THEME: PD trust boundary = wire-supplied index clamping — instances: this, F13 (staging_handler._scatter_region token window unbounded), F12 (aux_index*item_len deserialize unchecked). Grouped fix sweep planned post fix-wave; owner disagg-lead.

## F15 verdict (new HIGH)
get_mha_kv_ptrs_with_pp (common/conn.py:1549-1582): draft-branch dispatch via divisibility is wrong when (num_main+num_draft)%stage_layers==0 (self-draft-like or pp combos) → else-arm writes main V into decode draft_V region; V_main never written → stale cross-request reads. EAGLE-1-layer pp=1 accidentally safe ((L+1)%L≠0). Fix: explicit draft count from kv_args.num_draft_entries/draft_kv_layers. Owner: disagg-prefill (+spec-auditor liaison). Version mapping pending.

## F12 verdict (VERIFIED, trust-boundary write primitive)
Handle: mooncake/conn.py _handle_aux_data (1811-1829) → AuxDataCodec.deserialize_data_to_buffer (common/utils.py:101-109). Wire-controlled room/buffer_index/aux_index/len: no room-membership check, no buffer_index range check, no aux_index capacity check, no len(data)==item_len check → write into arbitrary slot (OOB heap or another request's aux/metadata region). Same site exists in mori/conn.py:1950 (out of prod scope). Intra-cluster trust, but stale-message generation mixups make it reachable without an attacker. Fix: validate room ∈ transfer_infos, buffer_index < len(aux_data_ptrs), aux_index < capacity (plumb aux capacity into kv_args/manager), len(data) == item_len else reject. Owner: disagg-prefill (after F15). Pair with wire-integ K-classes.
