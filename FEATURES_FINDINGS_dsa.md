# DSA/Sparse + DP-attention audit findings (dsa-auditor)

Scope: deepseek_v4 pools, compress state, dsa_cache_layer_split, dsv41_request_window, qsa, hisparse, DP pool divergence.
Branch: agent/dsa-auditor (off origin/main 3420bae).

## Known-bug-class verification at HEAD

### K1. DSA tail req_pool index OOB — STATUS: FIXED at HEAD
- `_compress_tail_k/_score` tail buffers sized `req_pool_size = max_running_requests + 1`
  (DSATokenToKVPool._init_kpool_compress_tail_buffers, memory_pool.py:5300).
- All three construction sites now pass the real req-pool capacity:
  - kv_cache_configurator.py:1694 `_build_dsa_kv_pool(..., max_running_requests=req_to_token_pool.req_to_token.shape[0])`
  - kv_cache_configurator.py:2333 hybrid MLA/SWA DSA branch: `max_running_requests=(req_to_token_pool.req_to_token.shape[0])`
  - `tail_extra_slots` = num_draft_tokens only affects ring width (dim1), not req count (dim0).
- Ring offset math `start_phys = (seq_len - n_valid) % tail_size`, first+second_n <= n_valid < tail_size -> in-bounds.
- get_dsa_tail_state_indices (disaggregation/utils.py:1208) only emits (req_pool_idx, start, n) — no bounds check
  but consumer indexes sized buffer; OK given the above fix.
- MUST verify production tags v0.5.14..v0.5.20 still have the unfixed version. TODO.

### K2. DSV4 request-scoped pool sizing vs decode prealloc slots — STATUS: FIXED at HEAD
- DeepSeekV4TokenToKVPool now takes `num_req_slots=req_to_token_pool.req_to_token.shape[0]`
  (kv_cache_configurator.py:1805), covering DecodeReqToTokenPool prealloc slots
  (`_alloc_size = size + pre_alloc_size + 1`, disaggregation/decode.py:189).
- C4 unified state ring, C128 ring, ratio-2 pair ring all sized off num_req_slots:
  deepseek_v4_memory_pool.py:1003,1008,1012,1634.
- QSA pending ring: `num_request_slots=req_to_token_pool.req_to_token.shape[0]` (kv_cache_configurator.py:2369).
- clear_c4_req_states registered as on_alloc_rows hook per-slot reset (kv_cache_configurator.py:1838) —
  unified_kv only, which is the only config with request-scoped C4 state. No explicit free needed.

### Compress-state lifecycle (alloc/free symmetry) — clean by design so far
- Request-scoped state (c4 unified / c128 / pair) is per-req-slot rings reset on alloc
  (clear_c4_req_states hook) or on PD receive (decode.py:1706 before transfer).
- No free() that could double-fire found in pool code path itself. Abort paths release req slot only;
  next alloc resets. No leak/OOB found yet. TODO: check retract/abort of grammar & spec paths (spec-auditor
  owns spec; watch clear_unaccepted_c128_draft_states).

### dsv41_request_window (encoder replay only; not in prod list) — partial
- zero_row/sink_row at rows, rows+1 with buffer sized rows+page_size: in bounds for page_size>=2.
- commit writes tags for sink_row too (dst=sink_row when commit_mask False) — benign scratch row.
- history_src masked to zero_row when invalid; pos%capacity always non-negative in torch. OK.
- tags staleness explicitly validated in buffer() (RuntimeError). OK.
- NOTE: RequestWindow only active with enable_encoder_swa_bounded_replay (not prod) — low priority, verdict OK.

### dsa_cache_layer_split (GLM DSA prefill-CP only)
- Staleness of remote_buffer for index keys is handled: dsa_indexer calls
  invalidate_index_buffer_for_layer on EVERY rank before writing its own index data (dsa_indexer.py:649 etc.),
  forcing re-broadcast on next read. KV path uses pending/finalize broadcast with stream sync.
- No double-free/leak found in layer-shard bookkeeping; non-owned layers are 0-row tensors.
- Prefill-CP GLM only; not in DSV4 prod path. Verdict: no prod-facing bug found, moderate confidence.

### DP attention pool divergence — control-plane only
- Each DP rank owns independent pools; a request is routed to exactly one DP rank (routed_dp_rank).
- dp_attn sync only shapes batches (prepare_mlp_sync_batch_raw), no cross-rank slot sharing.
- No divergence mechanism found in pool layer. Verdict: nothing found (will note for challenge review).

## Open TODO
- [ ] Check production tags v0.5.14/15.post1/K3-branch/17/20 for K1/K2 fixes.
- [ ] HiSparseC4DevicePool translate maps (full->hisparse device index mapping) bounds; enable_hisparse
      rejected with trtllm uniform_fp8 (deepseek_v4_memory_pool.py:1461) so not prod for DSV4-trtllm.
- [ ] hisparse_memory_pool.py (separate older MLATokenToKVPool subclass) quick pass.
- [ ] clear_unaccepted_c128_draft_states bounds (spec overlap; coordinate with spec-auditor).

## DP-attention max_running_requests sizing consistency (lead C2, msg #68)
EVIDENCE:
- Admission cap == pool sizing: scheduler.max_running_requests = tp_worker.get_worker_info()
  = model_runner.max_running_requests = kv_cache_configurator.configure().max_running_requests
  = resolve_max_num_reqs() output (scheduler.py:1187, model_runner.py:967). One source of truth.
- resolve_max_num_reqs (kv_cache_configurator.py:2797): requested_per_worker = user // attn_dp_size (floor),
  then min with token_capacity//2 and mamba cap. Resolved <= requested_per_worker always => any pool sized
  by raw user//dp(+1) is >= admission; no OOB direction.
- All // attn_dp_size consumers (2956 replayssm bytes, 2980/2988/3005/3044 mamba byte reservations,
  pool_configurator.py:930, moe_hook.py:593, lora_manager.py:1192, pd_disaggregation_hook.py:136
  decode extra slots) use the RAW user value divided by dp, which oversizes == safe direction.
- user=0 edge: --max-running-requests < attn_dp_size gives requested_per_worker=0 -> boot assert
  (tp_worker.py:413), fail-loud, not corruption.
- --max-running-requests semantics: treated as GLOBAL across attention-DP workers (divided everywhere);
  help text ("maximum number of running requests") is ambiguous but consistent with global reading.
  Recommendation: doc fix only.
- VERDICT: no memory-safety bug. C2 refuted for OOB. Adjacent byte-reservation undersizing under
  replayssm+dp>1 is mamba budget math (OOM risk only) -> hand to mamba-auditor.

## Shipped tests coverage (lead msg #89)
- Covered: C4 lifecycle on alloc hook incl. decode prealloc pool hook (test_dsv4_c4_state_lifecycle.py),
  compressed pools exact sizing (test_dsv4_compressed_pools.py), hisparse slot translation,
  layer-split broadcast, qsa pool.
- GAPS (no unittest found): DSA compress-tail ring with prealloc-sized req pool (K1 regression only
  protects via constructor arg sites; a test would pin it), transfer-index fns boundary/cross-ring
  (my harness logic), abort-path request-state reset (PD decode clear_request_scoped_state),
  DP per-worker admission == pool-size invariant (single pytest asserting
  max_running_requests == req_to_token_pool.shape[0]-1-pre_alloc sizing relations).
