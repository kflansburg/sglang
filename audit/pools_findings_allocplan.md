# allocplan-auditor findings: allocation.py / kv_index_translator.py / kv_loc_plan.py + referenced helpers

Scope: index math (OOB, negative/overflowing KV locs, page-boundary off-by-one,
double counting, DCP widening round-trip). Production surface checked: page_size 64,
fp8/bf16 KV, chunked prefill 12288-32768, DP attention with attn DCP, SWA,
disaggregated prefill/decode admission/allocation path.

Legend: [BUG] confirmed, [LATENT] real hazard needing an unusual config, [CLEAN] verified.

## Ranked results

### 1. [LATENT] int32 truncation of DCP-WIDENED virtual KV ids — no bound check
`ReqToTokenPool.req_to_token` is **int32** (memory_pool.py:405). Virtual token ids
under attn DCP are widened: `index_buf_size = (size + page_size) * dcp_size - page_size`
(memory_pool.py:5214) and the static-pool DCP loc space is
`max_total_num_tokens * attn_dcp_size` (kv_cache_configurator.py:371, 1945-1949).
Every write path truncates int64 -> int32:
- allocation.py:648 `req_to_token_pool.write((idx, locs), out_cache_loc.to(torch.int32))`
- kernels `write_req_to_token_pool_triton` / `assign_req_to_token_pool` (implicit
  tl.store cast int64 -> int32).
If widened id space >= 2^31, ids wrap NEGATIVE in req_to_token. Downstream gathers
(req_to_token[row, col], kv_read_table kernel `tok < 0 -> sink clamp`, read kernels
indexing kv buffers with negative ids) either silently alias to slot 0 (corruption
masked as garbage attention) or read out of bounds. The unified pool guards its
PHYSICAL id space (`_assert_physical_id_bound`, unified_memory_pool.py:352), and
kv_read_table clamps negative VIRTUAL ids to the sink — but nothing guards the
static (non-unified) DCP pool's widened VIRTUAL space at config time.
Estimate of reachability: max_total_num_tokens * attn_dcp_size >= 2^31, e.g.
270M tokens at dcp 8. Not hit by today's deployments (pools <= ~100M), but
nothing fails loud before it. The offload path turns the wrap into a
CORRUPTION chain: `maybe_dcp_kernel_indices` positional stride + floor-div of a
wrapped negative widened id yields negative physical rows; torch semantics map
`kv_buffer[-k]` to end-of-buffer rows, so HiCache/offload silently copies the
WRONG request's KV and later restores it — cross-request context leakage.
Recommendation: assert
`max_total_num_tokens * attn_dcp_size < 2**31` where the widened loc space is
derived (kv_cache_configurator), mirroring `_assert_physical_id_bound`.
(No runtime patch landed; config-time guard is the right fix and belongs to the
configurator owner's file.)
Version check (git fetch tag from upstream): v0.5.14 has NO attn-DCP
(no `dcp_size` refs in mem_cache) so its widened-space factor is absent;
v0.5.17 (DeepSeek V4 Flash) has the same class: int32 req_to_token
(memory_pool.py:280) + `(size + page_size) * attn_dcp_size` bound
(memory_pool.py:4106) + no 2^31 guard. v0.5.20: same (int32 at :285, DCP refs
throughout). Affected: v0.5.17, v0.5.20(+), latest; not reachable on v0.5.14/
v0.5.15 (DCP absent there) unless a plain pool itself hits 2^31 tokens
(unrealistic, >4 TB KV).

### 2. [CLEAN] get_num_new_pages / alloc_extend / alloc_decode page math at page_size 64
- utils/common.py:4913 `get_num_new_pages`: decode rule `seq % ps == 1` is exactly
  `(seq-1) % ps == 0` of `alloc_decode_naive`/`alloc_decode_kernel`; extend rule
  `ceil(seq)-ceil(prefix)` matches both kernels' per-request page counts.
- kernels/ops/memory/allocator.py alloc_extend_kernel: 3-phase (suffix of old page,
  new full pages, new partial page); masked loads/stores all bounded by
  num_part1/2/3; part-2 page load index `offset//ps` < num_part2//ps (num_part2 is a
  multiple of ps); part-3 unmasked load of the last new page only occurs when
  num_page_start_loc_self >= 1 (proved: reaching part 3 implies a new page exists).
- alloc_extend fast path skips the exact page count when
  `extend_num_tokens // ps + bs + 1 <= free_pages` — a proven strict upper bound on
  the sum of per-request pages, so the kernels never index past free_pages.
- alloc_extend_naive dead-lane clamps (page_idx2/3) noted and correct.
- 20k-case pure-Python fuzz of the exact integer semantics (prefix 0..500,
  extend 0..300, ps=64) in test/srt/pools_fuzz/fuzz_alloc_extend_pure.py: all
  invariants hold (no dup locs, no negatives, free-page indices in range,
  decode rank == get_num_new_pages).

### 3. [CLEAN] get_last_loc (allocation.py:103-169)
- torch variant returns -1 for prefix_len==0 via torch.where; triton legacy and
  _safe kernels both mask loads with `prefix_lens > 0` (other=-1).
- No negative-index load is possible: masked lanes never dereference.
- HIP/NPU int32->int64 mixed-store miscompile is already routed to the _safe
  variant (documented), CUDA keeps legacy (correct store types match buffer).

### 4. [CLEAN] alloc_for_decode_default last_loc read (allocation.py:615)
`req_to_token[batch.req_pool_indices, seq_lens_gpu - 1]` indexes column -1 (row
wrap, reads LAST column) only if seq_lens == 0. Decode invariant: every request in
a decode batch has >= 1 committed token (prefill writes >= 1 token; retract keeps
committed prefix >= 1). Not enforceable inside the function but the invariant holds
across scheduler paths; a defensive clamp would mask real bugs elsewhere.

### 5. [CLEAN] KVLocPlan cols/sentinel handling (kv_loc_plan.py)
- `_cols` Tensor path: `torch.where(cols >= 0, ids[cols.clamp(min=0)], 0)` — -1 -> sink
  0; clamp-then-discard is safe; empty ids fail loud (IndexError), not corruption.
- slice path: `ids.view(bs, -1)[:, cols]` requires numel % bs == 0; .view raises on
  ragged windows (fail loud). `window_read_extent` uses numel // bs only for
  decode/draft_extend_v2 where windows are uniform [bs, k]; ragged verify uses
  draft_token_num — no undercount hazard reachable.
- `pad_with_sink` pads with id 0 (the reserved sink page in every allocator's
  clear()); sink id semantics verified in paged.py:clear and kv_read_table (-1/v2p<0
  clamp to entry 0).
- `read_table` growth: a second reader asking for `rows > previous` copies
  previous.ids and zero-fills lanes past the ORIGINAL batch; the first build always
  has rows >= bs (rows = max(bs, rows)), so real lanes are never zero-filled by a
  grow. Sink lanes are by design for padded graph buffers.
- `build_iteration_table` width: `min(max(ceil((max_seq+read_extent)/ps),1), row_pages)`
  clamps to req_to_token row width; `zero_tail` prevents stale columns in reused
  `into` buffers.

### 6. [CLEAN] DCP sentinel math in translate_dcp_read_ids (kv_index_translator.py:592)
torch floor semantics: -1 // dcp_size == -1 (sentinel preserved through the
collapse); -1 % dcp_size == dcp_size-1 (torch remainder follows divisor). Verified
with pure-Python replica. Non-sentinel negatives (<= -2) would alias onto sink/-1,
but no producer emits them (v2p freed rows are -1; req_to_token never holds <= -2).
- NOTE (seam, kernel-owned): `filter_dcp_local_kv_indices` keeps a -1 sentinel only
  on rank dcp_size-1 (value filter), whereas `maybe_dcp_kernel_indices` (positional
  stride over a page-aligned run) preserves every padded slot. Both are safe only
  because consumers treat -1 as sink and padded write lanes target sink slot 0.
  The pool write kernels (`set_mla_kv_buffer_dcp_sharded_triton`) own the
  owner-rule filter for writes; out of my files.

### 7. [CLEAN] write_req_to_token_pool_triton / assign_req_to_token_pool store bounds
No explicit `seq_len <= max_context_len` mask: a row write with seq_len > row width
would spill into the next row (contiguous [N, max_ctx] buffer). The bound is by
construction: rows are max(model context, ...) + 4 + speculative_num_draft_tokens
(kv_cache_configurator.py:839-842, extra_max_context_len), and admission caps
seq_len at model context_len; spec decode adds <= draft_tokens per step. Invariant
holds; flagged so configurator/scheduler auditors check the +4/+draft margin once
(overlaps their files).

### 8. [CLEAN] alloc_for_extend / spec / decode free-on-error paths
- alloc_for_extend except-path frees out_cache_loc (fresh virtual locs, no -1s).
- `_alloc_extend_loc_with_kv_reuse` dLLM path: validates
  prefix+extend <= kv_allocated_len and extend == retained before reusing; the
  seq_lens override for reuse reqs (seq=prefix -> extend 0 tokens) keeps
  extend_num_tokens consistent with out_indices sizing; fresh_ptr arithmetic exact.
- paged.py free()/free_segment(): free of a NEGATIVE index would release page -1
  (torch floor), but no caller can pass negatives: free sources are req_to_token
  row segments [protected, owned) and out_cache_loc tensors, all non-negative by
  construction; the debug-mode duplicate-page assert would catch bad frees in
  testing. Page 0 (sink) is never in free_pages (clear() starts at 1).

### 9. [CLEAN] swa free helpers in common.py (referenced)
- free_swa_out_of_window_slots: asserts cache_protected_len % page_size == 0,
  floors the eviction frontier to page, thresholds max(window, page) so the leaf is
  never all-tombstone; start_pos passed to free_segment is the pre-floor cursor
  (evicted_seqlen), which IS page-aligned because it starts 0/aligned and each
  advance floors (set_evicted_seqlen(new) with new floored).
- free_kv_row_segments asserts the swa eviction cursor is page-aligned whenever a
  page would straddle the full/swa free split; ceil_align in
  _release_overallocated_kv_indices aligns to the ALLOCATOR page (DCP-widened),
  with the comment explaining the live-page hazard.

### 10. [CLEAN] capture_token_capacity (kv_index_translator.py:238)
Only sizes batch-position-indexed capturer buffers (state_capturer/*, indexed by
out_cache_loc.shape[0], not by KV id). Both branches (translating: v2p.numel()*ps;
static: max_token_pool_size + page_size) are upper bounds on batch tokens. No OOB.

### 11. [CLEAN] alloc_paged_token_slots_decode single-token contract
`token_per_req` is only ever 1 at every call site (schedule_batch.py:3643);
multi-token spec growth goes through alloc_for_spec_decode (extend-style),
which over-evicts and uses the extend kernels with per-req seq deltas. The
paged decode kernels' single-slot-per-req assumption is therefore upheld.
Idle-token edge (all extends 0): alloc_extend_kernel returns after part 1 when
pre==seq for both aligned and unaligned cases; verified by hand + fuzz.

## Out-of-scope noted (other auditors' files, reported to them)
- dsv4/NPU c128 paths, unified_sub_pool/unified_memory_pool internals, NPU
  dcp-rank-local write localization (`localize_dcp_indices` sink-0 asymmetry:
  id 0 -> -1 on ranks != 0; NPU-only, not in our CUDA production surface).
- deepseek forward_mha.py:691 DSA-non-aiter path uses raw page_table_1_flattened
  (widened) directly in dequantize_k_cache_paged while the aiter path filters +
  translates — flagged to dsv4-pool-auditor if not already known.

## Methods
- Full read of allocation.py (841), kv_index_translator.py (609), kv_loc_plan.py
  (348), base_swa_memory_pool.py (36), common.py (474), layers/dcp/layout.py (202),
  paged.py allocator + kernels/ops/memory/{common,allocator}.py +
  kernels/ops/kvcache/kv_read_table.py (entry points used by my files).
- Pure-Python exact-semantics fuzz of extend+decode page math (20k trials, ps=64).
- Pure-Python replica of torch floor/remainder on negatives for DCP sentinel flows.
- Cross-file invariant tracing for seq<=max_ctx, prefix<=seq, last_loc chain
  alignment, non-negative free inputs.
