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

## Production-version check (upstream tags fetched from github.com/sgl-project/sglang)
- K1 DSA tail req_pool OOB:
  - v0.5.14/v0.5.15.post1: NO kpool-compress tail feature at all (no _compress_tail_k/kpool_use_compress in mem_cache).
  - Feature introduced sized via req_to_token.shape[0]; v0.5.17 memory_pool.py:1241 uses self.req_to_token.shape[0].
  - v0.5.20 memory_pool.py:4901 _init_kpool_compress_tail_buffers takes max_running_requests+1 BUT the only caller
    (_build_dsa_kv_pool chain) is fed req_to_token.req_to_token.shape[0] at kv_cache_configurator.py:1263 and :1873. OK.
  - HEAD: same as v0.5.20. VERDICT: all prod versions safe.
- K2 DSV4 request-scoped ring sizing (num_req_slots):
  - v0.5.14: model_runner_kv_cache_mixin.py:488 passes shape[0]. OK.
  - v0.5.15.post1: same at :694. OK.
  - v0.5.17: kv_cache_configurator.py:1010 shape[0]. OK.
  - v0.5.20/HEAD: :1354 shape[0]. OK.
  - Kimi K3 branch base commits 87dd4596/bd4828e2 not present on public upstream; base ~v0.5.16 sits
    between fixed tags; high confidence OK, flag for internal-repo double check.
- QSA pending ring: QSATokenToKVPool not in v0.5.14/15.post1/17; at HEAD sized to shape[0] (configurator:2369). OK.

## Transfer-index boundary harness (CPU, numpy)
- c4_state_transfer_indices + request_scoped_state_transfer_indices: all invariants PASS
  (bounds vs num_req_slots*ring/pages_per_req; boundary seq=0 / seq%ratio==0 empties; cross-ring 8<->16
  positional consistency; bad ring rejected). Verdict: correct at HEAD.

## CORRECTED production-version matrix (after features-lead challenge #216)
Challenge claimed v0.5.14/15.post1 were AFFECTED because kv_cache_configurator.py (with the caller-side
fix) only exists in v0.5.17+. DIRECT TAG EVIDENCE REFUTES this: the fix in .14/.15p1 lives in
model_executor/model_runner_kv_cache_mixin.py:
- v0.5.14:488 + v0.5.15.post1:694 both pass num_req_slots=self.req_to_token_pool.req_to_token.shape[0]
  with the PD-prealloc comment. K2's SWA-ring sizing was already fixed there.
Moreover the RISKY FEATURES are version-gated:
- K1 kpool-compress tail (DSATokenToKVPool._compress_tail_*): ABSENT in v0.5.14 AND v0.5.15.post1
  (no _compress_tail/kpool_use_compress in memory_pool.py). First appears v0.5.17 sized via
  req_to_token.shape[0]; v0.5.20/HEAD sized via ctor arg fed shape[0]. -> no prod version affected.
- K2 request-scoped PD transfer (DSV4_REQUEST_STATE / request_scoped_state_transfer_indices /
  CompressStatePool.request_scoped): ABSENT in v0.5.14, v0.5.15.post1, v0.5.17; PRESENT only in
  v0.5.20+HEAD with correct sizing from introduction. -> no prod version affected.
- DeepSeek V4 Pro on v0.5.15.post1: NOT exposed to K1/K2 (features absent; SWA-ring class already fixed).
- Historical note: the same class DID ship and was fixed pre-v0.5.14 ("sizing as max_num_reqs+1 overflows
  (length out of range)" comment at v0.5.14 deepseek_v4_memory_pool.py:489).
- Kimi K3 branch (87dd4596/bd4828e2): still unverifiable from public upstream. Both K1/K2 features
  post-date its ~v0.5.16 base upstream, but the branch may carry cherry-picks; needs internal-repo check
  for (a) num_req_slots caller arg, (b) DSATokenToKVPool tail sizing, (c) _compress_tail_k req dim.

## FINAL VERDICT for assigned scope
- No new memory-safety bug confirmed in my scope at HEAD. Both known classes (K1, K2) are fixed upstream
  and fixed in every reachable production tag. Two unconfirmed hardening candidates (c128 cleanup kernel
  row-mask; hisparse -1 divergence) + 4 test gaps forwarded to features-lead.

## C128 cleanup kernel candidate (msg #216 follow-on) — VERDICT: not a bug at HEAD, hardening noted
- `clear_unaccepted_c128_draft_states` (kernels/ops/attention/dsv4/c128_cleanup.py) writes
  `state + (req_pool_idx*ring_size + slot)*2*half` with no row-bounds mask.
- Only caller: eagle_worker_common.py:746 with batch.req_pool_indices of the running batch — bounded by
  the K2-sized pool (ring rows = num_req_slots = req_to_token.shape[0]). Slot = (seq_len+draft_offset)%ring < ring.
- Order: runs post-verify forward, pre commit; slot algebra matches the draft C128 compressor's ring
  addressing (pre-verify seq_lens + draft offset). No OOB or stale-write at HEAD.
- Hardening suggestion (NOT applied; no confirmed bug): a defensive row mask or debug assert on
  row < pool._size, since the kernel is invariant-dependent.

## Regression tests added
- test/registered/unit/mem_cache/test_dsv4_request_state_transfer_indices.py (CPU suite):
  transfer-index bounds for all boundary seq lens; ring-region locality; empty/boundary no-transfer;
  8<->16 ring positional skew (EAGLE/MTP PD); bad-ring validation; CompressStatePool.transfer_indices
  dispatch (ratio-2 odd-only, offline matches module fn, page-scoped rejects). numpy logic verified
  locally; torch-dependent parts follow the shipped CPU-CI pattern.

## TEST DSA-1 (board #23) — delivered
test/registered/unit/mem_cache/test_dsv4_request_state_abort.py (CPU suite):
- clear_request_scoped_state offline c128 ring: slot-local reset, clear() semantics, other slots + padding untouched.
- ratio-2 pair state reset; all request-scoped pools reset together; page-scoped untouched.
- ONLINE_C128 3*head_dim row layout ([score|kv|rope]) via mocked ONLINE_C128 flag.
- prealloc slot > max_running_requests clears in-bounds.
- Abort/recycle invariant: poisoned slot freed WITHOUT compress-state clear (deferred-clear design),
  DecodeReqToTokenPool front-alloc/tail-free ordering verified emulated ([1] free→[2,3,4,1]), hook clears
  on re-alloc.
- Construction-site pins (test-gap #1): num_req_slots=req_to_token.shape[0] at DSV4 ctor site;
  DSA tail sites feed req-capacity; pool dim0 = max_running_requests+1 (contract that callers must
  feed capacity, not admission cap).
Numpy-emulated logic verified locally; torch parts follow shipped CPU-CI pattern.

## Mamba replayssm/dp byte-math review (for mamba-auditor, msg #235)
RESERVATION (kv_cache_configurator.py _handle_max_mamba_cache):
- replay_req_slots = user_max_running // attn_dp_size + 1  (:2956, GDN replayssm fixed bytes)
- intermediate reservations use user//dp and mamba caps (:2980,:2988,:3005,:3044)
- pool_configurator.py:930 SWA worst-case slots uses user//dp
ACTUAL ALLOCATION:
- MambaPool spec buffers: spec_state_size+1 rows (memory_pool.py:601/612/767/884/1038; rings :767).
- non-decode hybrid req pool: mamba_spec_state_size = resolved max_num_reqs (kv_cache_configurator.py:1558)
  -> actual <= reservation (resolved <= user//dp). OVER-RESERVED, safe.
- PD-decode hybrid req pool: HybridMambaDecodeReqToTokenPool passes
  mamba_spec_state_size = size + pre_alloc_size (disaggregation/decode.py ~:330) and
  effective_mamba_size = max(mamba_size, (size+pre_alloc)*slots_per_req)  -> CORRECT per-slot sizing.
  So on PD decode with GDN replayssm (+spec), ring rows = size + pre_alloc + 1 while the replayssm
  FIXED reservation counts only user//dp + 1: UNDER-RESERVED by pre_alloc_size rows * ring_bytes_per_req.
  Hygiene/OOM-risk only (budget carve too small -> mamba pool or kv budget overshoot at boot), not a
  memory-corruption bug. Affected config guess: Kimi K3 decode workers with enable_linear_replayssm_spec
  + GDN + disaggregation_decode_extra_slots > 0 + dp such that resolved == user//dp.
- moe_hook.py:593 uses max(1, user//dp); fine.
SUGGESTION to mamba-auditor: make replay_req_slots == actual spec_state_size basis
(get req_to_token row count or size+pre_alloc), and add a boot-vs-reservation assertion.

## BUG-HISPARSE-Q1 (board #51): radix-hit prefix NONZERO C4 mapping -> alloc_device_buffer clobber
Decisive code evidence (2026-10-16):
- V4 alloc_extend writes mapping only for extension slots (hisparse.py:560-616; paged alloc_extend
  is extension-only, allocator/paged.py:204-280). Prefix compressed slots retain creator's values.
- Tree-owned compressed-slot mapping cleared only at eviction (hisparse.py:629-635 free_compressed).
- alloc_device_buffer input covers req_to_token[..., :allocated_len] incl. prefix
  (hisparse_coordinator.py:940-946); :497 zeroes ALL; surplus frees recycle shared pages.
- Remaining narrow Q: worker context where device-mapped radix-hit prefix meets alloc_device_buffer
  (non-disagg hisparse, or decode-side radix reuse). Disagg host-staged admit may be all-zero.
- Guard-port decision: PORT V4 pure_surplus to non-V4 (pools-lead coordinates, alloc-auditor
  implements, dsa-auditor reviews).
- Merge queue: #16 submitted by features-lead for #27/#28; local queue-venv run: 21 passed, 0 failed.

## Seam verify: forward_mha.py:691 non-aiter fp8 DCP widened page_table (allocplan handoff) — REJECTED
- page_table_1_flattened: single producer dsa_backend.py:1254 from raw physical req_to_token
  page table (:1071). DCP widening (localize_dcp_page_table, :3846) applies to page_table_1
  only, at call time inside the sparse flow. Metadata flatten stays physical.
- Consumers (:2330/:2379, forward_mha fp8 dequant) all read the same metadata field. Consistent.
- aiter filter/translate is ROCm raw-layout compensation (576B vs 656B comment forward_mha.py:664-685).
- Reported to unified-auditor (#756).
