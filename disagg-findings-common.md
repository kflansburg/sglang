# Findings: disaggregation/common (staging_buffer, staging_handler, dcp_pack, bootstrap, conn shared)

Branch: agent/disagg-prefill-common. Owner: disagg-prefill-common.
Status codes: CANDIDATE (reasoning shown, needs repro/second review), VERIFIED, REFUTED.

---

## F1. CANDIDATE — race: late staging CHUNK_READY scatters into KV pages concurrently with abort/teardown freeing them
- File: staging_handler.py:297 (handle_chunk_arrived) -> :229 (submit_chunk_scatter) -> :393 (_scatter_region); teardown at :180 (unregister_decode_req/release_room).
- Threads: decode_thread (arrival path) vs scheduler main thread (unregister/release + page free in scheduler abort).
- Interleaving:
  1. decode_thread: `handle_chunk_arrived(room)` -> `_room_to_receiver.get(room)` != None; all writers arrived.
  2. main thread: `unregister_decode_req(room)` pops room, `release_room` syncs scatter_stream and frees staging allocs; scheduler abort path then frees the request's KV pages back to the pool for reuse by another request.
  3. decode_thread: `submit_chunk_scatter` ran BEFORE the pop or concurrently: `_room_to_decode_req.get(room)` still returned decode_req -> `_scatter_region` reads `req_to_token_pool.req_to_token[req_pool_idx, token_start:token_end]` and launches scatter kernel writing into those page indices.
  4. If pages/req_pool row got freed and REUSED between (2) and the kernel execution... Note: release_room does `stream.synchronize()` before freeing staging allocs, which drains the kernel; but the KV pages written are the decode request's own pages — writing to them is only harmful after they are freed/reassigned. `stream.synchronize()` in release_room ensures the kernel finished before unregister returns; the question is whether the scheduler frees the KV pages BEFORE calling unregister_decode_req.
- What breaks: cross-request KV corruption (scatter lands in reassigned pages) IF any teardown path frees KV pages before unregister_decode_req drains the scatter stream. Needs decode.py abort ordering to confirm exploitability (parent: disagg-prefill, decode.py owner).
- Also: `_scatter_region` reads req_to_token row from a snapshot AFTER teardown began: token_start from `cache_protected_len + page_start*page_size` — if retraction/abort shrank the alloc, kv_indices may be 0s -> scatters into page 0 (reserved?) or wrong pages.

## F2. CONFIRMED-BY-READING leak — _wm_subscribers grows forever
- staging_handler.py:121 (_wm_subscribers) never pruned on room teardown; every watermark free broadcasts ZMQ to every dead prefill session (exceptions swallowed). Slow leak + wasted sockets per request.

## F3. CANDIDATE — decode-side staging is shared ring across ALL rooms; `handle_staging_req` allocates with no per-room quota: a malicious/buggy room can pin the whole ring (watermark never advances), deadlocking other rooms' chunks until 30s timeout. Availability; matches known class "admission abort".

## F4. REFUTED — compute_head_slice_params src>dst head OOB: in src_attn_tp>dst_attn_tp branch, src_heads_per_rank <= dst_heads_per_rank always (src_tp>dst_tp), so head_offset+num_heads*head_dim stays within stride. No wrap overflow.

## F5. CANDIDATE — `_gather_all_layers_torch`: per call allocates `page_idx_tensor` + expanded `gather_idx` (num_tokens*num_heads*head_dim*8 bytes int64!) — for 32768-token chunk, 8 heads, 128 dim = 32K*8*128*8 = 268MB int64 gather index on EVERY chunk transfer. OOM risk under concurrency; also `torch.gather(out=)` with overlapping? Not memory-unsafe per se, but huge transient alloc in transfer worker.

## F6. CANDIDATE — PrefillStagingContext.prefetch_requested keys (room, chunk_idx, session_id) grow forever per room; also `prefetched_rooms` never pruned (only bounded by room reuse 2^63; slow leak).

## F7. CANDIDATE — `free()` empty-ring branch sets `head=0` and `round+=1`, and watermark=(new round, 0). A stale WATERMARK comparison in prefill `handle_watermark_msg` uses tuple compare `(round,tail) > prev` — fine. But `is_watermark_ready` returns True immediately for alloc_round<=0 — if allocator rounds ever wrap to negatives, no. Skip; kept for completeness.

(to be continued: conn.py bootstrap/metadata buffers, dcp_pack)
