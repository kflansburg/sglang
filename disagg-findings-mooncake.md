# Mooncake transfer-path audit findings (disagg-mooncake)

Scope: python/sglang/srt/disaggregation/mooncake/{conn.py, utils.py},
transfer-engine wrapper python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py.
Production config assumed: multi-NIC RDMA (8x mlx5), page-size 64, fp8_e4m3, MLA.
Reference commit: 3420bae (origin/main at audit time).

## Status of previously-known bug classes (verified in this tree)

- **(1) batch indices misaligned across layers/requests**: `set_transfer_blocks`
  in `_send_kvcache_generic` pairs `group_concurrent_contiguous` groups
  index-by-index with equal group lengths (ValueError if arrays differ in
  length — loud, not silent). Traversal over layers uses `layers_params`
  built from layer-id pairs when published. Looks correct here.
- **(2) transfer worker keep-alive**: worker threads are created per queue
  and the exception path raises RuntimeError which KILLS the daemon thread;
  requests then hang until the 300s waiting timeout. No supervisor/restart.
  The None-sentinel handles clean teardown only. Still a silent-death mode
  (see F4).
- **(3) freeing KV after ~30s timeout with RDMA in flight**: default config
  has SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE=True and Mooncake
  `supports_deferred_decode_kv_release = True` → drain-ACK protocol gates
  release. `_poison_unsettled_send` covers the `batch_transfer_sync`
  wall-clock-timeout-returns--1-while-batch-runs case by poisoning the ACK so
  decode falls back to the 30s deferred-release timeout. This is the known
  residual risk the design accepts; the poison is in place on every ret!=0
  path I traced (kv send, state send, aux send). OK upstream.
- **(4) partial batch counted as done**: `batch_transfer_sync` is all-or-
  nothing per call; `_await_transfer_futures` drains running futures before
  returning ret when deferred release is on; conclude_transfer only on full
  Success. OK. One exception: see F2 (silent tail truncation).
- **(5) registration/deregistration races**: see F3 (teardown) and F1
  (register failure ignored).
- **(6) per-IB-device routing**: single engine instance per process,
  ib_device passed at engine init; sender/receiver both address by raw
  pointer, engine picks NIC by peer topology. No Python-side routing table.
  N/A at this layer.

## Findings

### F1. batch_register/batch_deregister failures are swallowed; conn.py never checks the return
- `mooncake_transfer_engine.py:166-198`: `batch_register` logs at DEBUG
  ("Mooncake batch memory registration failed.") and returns -1; on exception
  it also returns -1 unless the method is missing.
- `mooncake/conn.py:340-344` `register_buffer_to_engine()` calls
  `self.engine.batch_register(list(ptrs), list(lens))` and IGNORES the return
  value. Server then runs with KV/aux/state buffers NOT registered with the
  transfer engine. Every subsequent RDMA to/from them fails at transfer time
  (loud, eventually, via session failure) — but on the DECODE side a failed
  registration means prefill's writes fail per-transfer while bootstrap
  succeeds; requests churn through 300s timeouts instead of failing startup.
- Severity: availability/latent-misconfig; not direct corruption (engine
  rejects unregistered regions — that is Mooncake-transfer-engine behavior;
  verify). Same for staging `_register_staging_memory` (conn.py:457-458).

### F2. transfer_worker silently truncates kv_chunk.prefill_kv_indices and mutates the SHARED chunk across sibling reqs
- conn.py:2556-2573: when `len(chunked_dst_kv_indice) < len(kv_chunk.prefill_kv_indices)`
  (page-size>1 mismatch that "should never happen") the code logs a warning
  and MUTATES `kv_chunk.prefill_kv_indices` in place. Two hazards:
  a) The truncated request still concludes Success; decode believes the tail
     pages hold valid KV, but they were never written — those slots contain
     stale bytes from whatever request previously owned them
     (cross-request context leak, silent; only a warning logged).
  b) The mutation persists for subsequent reqs in the same
     `reqs_to_be_processed` loop (multi-destination rooms: MLA replica /
     required_dst_info_num>1). Req 2's `chunked_dst_kv_indice` is sliced with
     the original index_slice while prefill_kv_indices is already truncated
     → length mismatch → `group_concurrent_contiguous` raises ValueError →
     transfer_worker dies (prefill instance "dead" until restart).
- With page-size 64 (production) the pre-conditions can arise whenever the
  decode-side destination page list is shorter than the prefill chunk (the
  exact trigger is upstream in metadata assembly — disagg-decode agent).

### F3. teardown() deregisters memory and closes sockets even when worker joins time out
- conn.py:445-448: `for t in self._worker_threads: t.join(timeout=3.0)` —
  the timeout is accepted silently. If a transfer worker is stuck inside
  `batch_transfer_sync` (RDMA stall), the thread is STILL ALIVE and may still
  be reading source KV / writing through the engine when teardown proceeds to
  `deregister_buffer_to_engine()` (conn.py:497-500) and `_zmq_ctx.destroy()`.
  Deregistering a memory region while the engine has in-flight RDMA on it is
  the classic post-timeout race (class 3/5) on the role-switch path.
  Role-switch is currently not in the production surface (verify), so
  priority medium.

### F4. transfer_worker death is silent → transfers stop, rooms hang to 300s timeout
- conn.py:2806-2812: any unexpected exception raises RuntimeError, killing
  the daemon thread. Nothing restarts it; `add_transfer_request` keeps
  enqueuing to a queue nobody drains (unbounded growth → memory leak of
  TransferKVChunk + their numpy index arrays), and every room on that shard
  waits for SGLANG_DISAGGREGATION_WAITING_TIMEOUT (300s). The error message
  says "Prefill instance ... is dead" but the process keeps serving.
- This is the residual of known class (2): partially addressed (teardown
  sentinel) but no liveness supervision.

### F5. decode-side register: batch_register ignored + kv_item_len==0 published for SWA-only pools
- conn.py:3227ff `_register_kv_args`: `kv_item_len = kv_item_lens[0] if ... else 0`
  — sends 0 for pools without full-token KV. `send_kvcache_slice` uses
  `dst_kv_item_len` unchecked in that case? (slice path only used when
  non-MLA and TP mismatch; SWA pools go through state path). Also aux
  `dst_aux_ptrs` pairing is positional and trusts peer lists; a version-skewed
  peer publishing fewer aux ptrs → `prefill_aux_item_lens[i]` IndexError →
  worker death (F4). Trust boundary: peers are other SGLang ranks, but mixed
  versions are explicitly contemplated elsewhere ("Mixed SGLang versions may
  wait..."). Low/medium.

## Areas verified with no issue found
- `_registerable_regions` dedups (ptr,len) to avoid double registration;
  0-length DSA-elided regions skipped — correct.
- `_await_transfer_futures`: drains all futures before returning failure when
  deferred release enabled (required for drain-ack soundness) — correct;
  early-return without drain when disabled is the documented legacy risk.
- `update_status`: Failed is terminal; no Success-after-Failed promotion.
- Deferred abort ACK: generation tracking, poisoned rooms, per-endpoint
  targets, ack dedup on decode — sound design; concurrency handled by
  copy-then-replace with retry.
- `_send_mamba_state`, `_send_kvcache_generic` PP slicing via
  get_mla_kv_ptrs_with_pp and layer-id pairing with elision handling.
- CHUNK_READY / staging watermark flow: staging buffers registered via
  `_register_staging_memory` (same F1 caveat), slots freed by decode
  watermark ACK before reuse.

## Version coverage (fetched upstream tags)

- v0.5.14 (prod Kimi/Gemma), v0.5.15.post1 (GLM/DSV4 Pro), v0.5.17 (DSV4 Flash):
  - F2 truncation block present in ALL (`len(chunked_dst_kv_indice) < len(...)`
    at v0.5.14:1244, v0.5.15.post1:1298, v0.5.17:1621, main:2558).
  - None of them has deferred decode KV release / drain-ACK / poison: on a
    per-layer transfer failure the code `f.cancel()`s and RETURNS IMMEDIATELY
    (v0.5.14:676-682, v0.5.17:759-765) while running siblings keep writing.
    Combined with decode freeing slots on failure/abort, this is the known
    "RDMA write into freed/reallocated decode pages" class, live in every
    production version. Mainline fixes it ONLY when
    SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE=true (default true),
    where _await_transfer_futures drains and the drain-ACK protocol holds the
    pages. Residual on main: deferred-release disabled => legacy early return.
  - register_buffer_to_engine ignores batch_register() return (F1) in all
    versions; older versions also lack the 0-len/dedup guard
    (_registerable_regions added later) so DSA-elided 0-length regions could
    abort registration on v0.5.14/15 (Mooncake rejects zero-length regions).
- Kimi K3 branch (base 87dd4596/bd4828e2 ~v0.5.16): not in upstream tags;
  TODO fetch from Kimi fork if needed — expected same as v0.5.15.post1.

## Engine-wrapper notes
- `batch_register` swallows exceptions -> -1, and raises only when the symbol
  is missing. Caller in conn.py ignores -1. A failed registration (RDMA NIC
  hiccup at startup, 8x mlx5) leaves the whole process serving with
  unregistered KV buffers: every transfer errors at transfer time and requests
  cycle through session-failure/300s timeouts instead of a startup error.
- `batch_transfer_sync` returning -1 after its internal wall-clock timeout
  leaves the batch running (comment in conn.py:2415-2422). Mainline poisons
  the drain ACK on every ret!=0 path -> decode 30s deferred-release timeout is
  the only guard. Any path that returns ret without poison is a hole; I
  verified kv/state/aux send paths in transfer_worker all call
  _poison_unsettled_send on failure. OK.

### F6. SWA/DSA state page-list length mismatch silently truncated (stale-state read)
- conn.py:~2065-2084 (`maybe_send_extra`, generic-kvcache state branch): when
  `len(src_indices) != len(dst_indices)` for paged SWA/DSA state the code logs
  a warning and truncates the LONGER list. Comment says "Paged SWA/DSA
  tolerate a 1-page drift" but arbitrary drift is accepted identically. If the
  decode page list is short, the unmatched prefill pages are never written,
  yet the room still concludes Success -> attention runs over stale SWA state
  pages (prior-request tokens) without any failure signal. Same hazard class
  as F2; exact-match types (QSA/SWA_RING/DSV4_REQUEST_STATE) raise instead —
  SWA and DSA do not. With SWA production config (--swa-full-tokens-ratio
  0.075-0.1) a drift >1 page means a real accounting bug upstream, masked here.

### Additional race reviewed: peer re-registration mid-transfer
- transfer_worker reads `decode_kv_args_table[session]` and
  `state_strides_validated` unlocked while the bootstrap thread replaces them
  in `_publish_peer_registration` under session_lock. Field order inside the
  lock (discard validated -> set rejection -> assign table) plus GIL atomicity
  means the worker can never observe (new table + stale validated-present);
  the unsafe combination is excluded by construction. Sound, fragile but OK.

### Wire-format dtype check
- Receiver `send_metadata` ships `kv_indices.tobytes()`; prefill parses
  np.int32. Caller (_send_kv_metadata in decode.py) converts via
  `_transfer_page_indices(...).astype(np.int32)` at decode.py:2177. Safe as
  long as every caller goes through that helper — noted for the decode agent.

## Update (fixes landed on agent/disagg-mooncake)
- F2 FIXED (commit "fail loud on dst/prefill KV index count mismatch"):
  conclude_failure + break, chunk never mutated. disagg-decode's trigger
  hunt found NO normal path producing the drift (page counts equal by
  construction both sides); the fix converts any *future* bookkeeping drift
  into a request failure instead of corruption. Synthetic unit tests shipped.
- F1 FIXED: register fail-fast raise; deregister logs error.
- F4 FIXED: per-chunk fault isolation in transfer_worker + dead-shard
  enqueue refusal in add_transfer_request.
- F6 FIXED: drift cap 1 page for paged SWA/DSA/BLOCK_SCALE(_SWA); exact-match
  types unchanged. Production relevance: SWA (silent before) IS in production
  (--swa-full-tokens-ratio 0.075-0.1); QSA/SWA_RING/DSV4_REQUEST_STATE
  already raised. Backport note (version-triage): v0.5.15.post1/.16 have a
  SECOND unconditional truncation site in the state branch (~:150-151)
  needing the same cap; also MINIMAX_INDEX_K truncation below maybe_send_extra
  (~:2135-2160) still truncates silently on HEAD (MiniMax not in prod configs).

## Mooncake counterpart of refuted nixl F1 (peer/handle invalidation+reload)
- No analog: Mooncake sessions are address-derived (host:rpc_port), the
  engine dedups by (hostname, gpu_id, ib_device); the ONLY peer metadata
  reload is _publish_peer_registration (kv args re-registration), which is
  ordering-safe: decode_kv_args_table[slot] replace happens inside
  session_lock AFTER state_strides_validated discard, so an unlocked reader
  (transfer worker) can never observe (new table + stale
  validated-present); GIL keeps single-op reads atomic. REFUTED for mooncake.
- Residual (leak, not corruption): failed_sessions, session_failures,
  decode_kv_args_table, state_layout_rejections, state_strides_validated are
  keyed by session id and never expire across decode-instance restarts.
- Residual (abort-ack path, being fixed by disagg-prefill in F9 follow-up):
  _send_abort_ack best-effort one-shot; transient zmq failure loses the ACK
  -> decode device path frees at 30s regardless (V4 compound; disagg-decode's
  fix removes the deadline-release entirely).
