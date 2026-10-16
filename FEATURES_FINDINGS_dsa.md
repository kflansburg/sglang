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
