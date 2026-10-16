# DSV4 pool mechanics audit — dsv4-pool-auditor (branch agent/dsv4-pool-auditor)

Scope: deepseek_v4_memory_pool.py (2142 lines, read fully), deepseek_v4_compress_state.py
(read fully), dsa_cache_layer_split.py (read fully), dsv41_request_window.py (read fully),
plus key call sites in deepseek_v4_backend.py / deepseek_v4.py / encoder_swa_replay.py.
NOTE: repo has ONE squashed commit; no upstream history available for "was the prior
DSA-tail bug fixed" diffing — verified structurally instead. Production tags fetched:
v0.5.14, v0.5.15.post1, v0.5.17, v0.5.20 all contain deepseek_v4_memory_pool.py and
deepseek_v4_compress_state.py; dsv41_request_window.py is HEAD-only (encoder SWA replay).

## FINDINGS (ranked)

### F1. window_layout(): no contiguity precondition on req ids → index corruption + scatter OOB (MEDIUM-HIGH, HEAD-only feature)
File: dsv41_request_window.py:67-132 (`window_layout` / `_first_row_offsets`).
Groups are computed from *runs* of equal req ids, but all sizing (`history_rows =
groups * window` with groups = caller's num_groups) assumes one run per request.
If `req_pool_indices_repeated` is non-contiguous (same req id in 2+ runs — e.g. DP
padding, replay injection or batch composition interleaving), simulation shows:
  - `g_req = zeros(groups).scatter_(0, group, req)` indexes with group ids up to
    n_runs-1 ≥ groups → out-of-bounds scatter (CUDA device assert, or silent
    corruption under captured graphs).
  - old_loc reaches `n_runs*window - 1` → `indices` max 511 vs layout size 392 in a
    minimal sim → attention reads rows beyond the activated workspace (garbage KV →
    corrupted output); workspace activate() only guarantees layout.size rows.
Current callers (expand_prefill_casually / expand_extend_with_same_length in
deepseek_v4_backend.py:1447/2086/2226) build per-request contiguous runs so the
invariant holds *by construction today*, but it is undocumented, unvalidated, and
spans teams. Fix: validate cheaply (assert group[-1] <= groups-1 outside capture;
prefer sizing history_rows from actual runs) or document+enforce contiguity.
Affected versions: feature not in any production tag (request window is HEAD-only).

### F2. PD trust boundary: transfer index helpers have no bounds clamp (LOW, all prod tags)
deepseek_v4_compress_state.py:87-104 `c4_state_transfer_indices`, 107-132
`request_scoped_state_transfer_indices`: rows = req_pool_idx*ring + pos%ring with
req_pool_idx/seq_len from the remote (prefill) side / scheduler. An out-of-range
req_pool_idx yields indices past the state pool → decode-side index out-of-range
crash (at best; under graphs, captured code with table-driven indices could read
in-pool garbage for a *different* request = cross-request leakage). PD peers are
intra-cluster trusted, so this is crash-robustness, not an exploit; flag for the
disagg team's trust-model file. All prod tags (function present since v0.5.15;
v0.5.14 differs — PD state transfer added later).

### F3. HiSparseC4DevicePool.data_ptrs cached pointer table (INFO/LOW)
deepseek_v4_memory_pool.py:438-441 caches raw `data_ptr()` in a device tensor at
__init__. If torch memory-saver pause/resume reallocates kv_buffer, the table is
stale → transfers into freed memory. No in-repo consumer found (likely external
hisparse transfer registration); hisparse/HiCache team should confirm memory-saver
re-registration. Same cached-pattern exists elsewhere (memory_pool.py), so treat as
known systemic pattern. All prod tags.

### F4. Non-contiguous-request aliasing even when sized correctly (LOW, HEAD-only)
window_layout with num_groups := n_runs keeps everything in bounds, but a request
split across two runs gets two independent history windows while `commit` writes by
req*capacity (shared ring) → the two runs read each other's half-written ring state.
Same contiguity contract as F1; documented here as the correctness twin.

## CHECKED-CLEAN areas (verdicts)

- RequestWindow slot math (dsv41_request_window.py:126-257): rows = num_slots*capacity
  with num_slots = num_req_slots = max_num_reqs+1; zero_row/sink_row = rows/rows+1, both
  < rows+page_size (tags size rows+page_size, page_size≥2); commit dst = req*cap +
  pos%cap ≤ rows-1 for req < num_slots; dst sink_row within tags; reset() loc ≤ rows-1.
  VERDICT: in-bounds given scheduler contract req_pool_idx < num_req_slots (+1 dummy
  reserved). pool passes num_slots=self.num_req_slots so capacities align.
- Prior "DSA tail" class (prefill-graph tail block): deepseek_v4_backend.py:1645-1690
  padding scatter fills scatter_buf with out_cache_loc.shape[0], and the consumer
  deepseek_v4.py:3237-3240 allocates full = new_zeros((num_tokens+1, ...)) — the dummy
  row is IN BOUNDS and sliced off. VERDICT: the one-past-the-end dummy row pattern is
  correctly defended at HEAD (this is where I'd expect the old bug to have been).
  gather idx_buf pads 0 (in-bounds), loc_buf pads 0 (reserved dummy KV slot-0 writes,
  intentional; raw_out_loc==0 marks padding for compressors).
- CompressStatePool sentinel-row protocol (deepseek_v4_compress_state.py:225-281):
  -1 state_loc → last allocated row (size = logical + ring_size + 1); writes then
  re-cleared in set_state_by_state_loc; translate fns masked_fill negative→-1.
  VERDICT: sentinel protocol consistent; no double-free class possible (no manual
  free list in this pool).
- UnifiedKVPool rows = swa_pages + num_blocks*k_per_block[ratio] + page_size//ratio
  pad; SWA ring = req*ring + pos%ring < swa_pages for req < num_slots; compressed
  region padded by one page for the reserved page-0 indexing style. VERDICT: sized ok
  (asserts compress_rows % rows_per_page == 0 in _unified_page_views catch skew).
- clear_c4_req_states / clear_request_scoped_state / clear_unaccepted_c128_draft_states
  index req*ring + {0..ring-1} into pools sized num_req_slots*ring(size) — in-bounds
  under the req_pool_idx<num_req_slots contract. Python slicing in
  clear_request_scoped_state clamps even if violated (soft fail). VERDICT: clean.
- DeepSeekV4UniformFP8KVPool.set_key_buffer_fused: negative loc → python wrap to last
  row = reserved dummy page; consistent with reserved-page convention. VERDICT: note
  only — kernel-equivalent path (trtllm) must share this convention (kernel area).
- fp8/scale buffer sizing: 584 B/token V4 layout asserted exactly (448+128+8) with
  page padding ceil to 576B; indexer page bytes = page_size*(dim + dim/128*4);
  low-ratio indexer pools sized (full_size+page_size)//ratio (covers reserved page 0).
  VERDICT: consistent; relies on full_size+page_size being even (true for page 64).
- LayerSplitDSATokenToKVPool broadcast/scratch: pending-broadcast discipline
  (set_kv_buffer finalizes pending before write; get_* finalize+wstream wait) reads
  as race-safe; remote buffers sized (size+page_size) matching owned. move_kv_cache
  uses maybe_detect_oob guards. VERDICT: clean; `index_k[tgt]=index_k[src]` in-place
  aliasing on overlapping ranges is a torch-undefined ordering risk shared with the
  base class (radix team's move_kv_cache area) — noted, no DSV4-specific issue.
- dsv4_aborts: no free-list ownership in these pools (rows addressed, never freed) →
  double-free class N/A in my 4 files; alloc/free pairing lives in allocators
  (core-pools team) and abort paths (disagg team).

## Cross-checks handed off
- dsa-auditor: F1 contiguity contract (can expand_* ever produce non-contiguous req
  runs under DP padding / DSpark?); verify swa_buf.copy_(swa_out_cache_loc) length
  match in graph_tail path (backend 1749).
- disagg team: F2 trust-boundary policy for PD-supplied indices.
- hisparse team: F3 data_ptrs consumer + memory-saver re-registration.
