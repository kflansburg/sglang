# NIXL disaggregation backend audit — findings (agent: disagg-nixl)

File: `python/sglang/srt/disaggregation/nixl/conn.py` (3638 lines, origin/main 3420bae).
Production surface: Kimi K3, `--disaggregation-transfer-backend nixl`, page 64, fp8_e4m3.

Note: this upstream file already carries many fixes for our known bug classes (peer reload
with `_peer_reload_lock`/rate-limit, `_await_handles` settle barrier, deferred abort ACK,
slice-dst dlists per head_group). Siblings / residual gaps below.

## Findings (candidates, to be verified)

### F1. RACE: `_reload_invalidated_peers` pops prep handles while another worker uses them
- `transfer_worker` runs in N threads (one per transfer queue, `ShardingQueue`, line ~560-570).
- On any exception a worker calls `self._reload_invalidated_peers(room_transfer_infos or {})`
  (line 1596). That iterates the *failed room's* peer names and, under `_peer_reload_lock`,
  does `self.prep_handles.pop(agent_name, None)`, `self.prep_handles_slice_dst.pop(agent_name, None)`,
  then `add_remote_agent` + `_prepare_payload_xfer` (lines 1671-1709).
- The lock does NOT protect readers: the hot path in `_send_kvcache_generic` (1745-1772),
  `send_kvcache_slice` (2107+), `send_kvcache_mixed` (2067), `send_kvcache_staged` read
  `self.prep_handles[peer_name]` / `self.prep_handles_slice_dst[peer_name]` with no lock.
- The in-code comment ("every room with this peer has the same peer set, so all shard to this
  worker") is FALSE: `add_transfer_request` (2955-2970) shards by `session_port_sum %
  num_queues`, so two DIFFERENT rooms with the same peer can hash to different workers.
- Consequences: another worker's in-flight room can KeyError at `prep_handles_slice_dst[peer_name]`
  (send_kvcache_slice does `self.prep_handles_slice_dst[peer_name]` directly), or worse grab the
  OLD prep handle while NIXL already invalidated the peer → make_prepped_xfer against a stale
  remote descriptor → **potential write to wrong remote region (cross-request leakage)** or ERR.
- The `_peer_reload_min_interval` throttle is only 1 s and per-agent; successive chunk failures
  for the same peer can pop/rewrite the handle repeatedly while other rooms are mid-transfer.

### F2. OOB: `send_aux` pairs prefill/decode aux buffers positionally, no length check
- Line 2362-2400: `for i,_ in enumerate(dst_aux_ptrs)` reads `prefill_aux_item_lens[i]` and
  `prefill_aux_ptrs[i]`. If decode registered MORE aux entries than prefill has, prefill threads
  crash (IndexError, benign-ish); if decode registered FEWER, extra prefill aux data is silently
  not sent (aux incomplete on decode → **silently incomplete aux, possible wrong metadata**).
- If aux item lens disagree, dst write length uses `prefill_aux_item_lens[i]` regardless of the
  decode-side buffer size → **possible OOB write past the remote aux buffer** into adjacent host
  memory (DRAM descriptors). No check like the KV path's item_len validation (which
  `_build_transfer_dst_indices` does for KV, cf. 1092-1105 raise).

### F3. Unbounded wait: `_await_handles` blocks a transfer worker forever on a hung peer
- Pre-failure path waits with no deadline while `any_failed` is False (line 739-757:
  `time.sleep(0)` busy loop). If NIXL never settles a handle (peer crash mid-RDMA where NIXL
  reports PROC forever), that transfer-queue worker is stuck forever; all rooms sharded to the
  same queue starve. No watchdog/handoff. (Only after a first ERR does the 5 s deadline arm.)
- Related: `room_transfer_infos`/`request_status` for those rooms never progresses; decode side
  relies on its own timeout.

### F4b. Cross-thread dict mutation without lock: `transfer_infos`, `request_status`,
  `decode_kv_args_table`, `_staging_outstanding` accessed from bootstrap thread, scheduler thread
  (`add_transfer_request`/`_prefetch_staging_reqs`), N worker threads, decode listener thread and
  heartbeat thread. CPython dict ops are atomic individually, but read-modify-check sequences are
  not, e.g. `add_transfer_request` checks `bootstrap_room not in self.transfer_infos` then
  dereferences; worker may pop concurrently at line 1549 (`self.transfer_infos.pop(room)`) after
  the check → `return None` race is benign here but `handle_staging_rsp(waiting_req_bytes,
  self.transfer_infos)` (line 3232) passes the live dict to a handler with no snapshot.

### F4 (confirmed at code level). Deferred-ACK predicate treats "popped" as "drained"
- transfer_worker top-of-loop (1250): chunk for a cleared room pops
  `self._staging_outstanding.pop(room, None)` then calls `_maybe_ack_drained_abort(room)`
  (_staging_outstanding is a defaultdict; the increment at 1261 only happens AFTER this check).
- If an ABORT already queued this room as Failed (so the counted, uncountered, or about-to-be-
  counted chunks are skipped at loop top), each skip pops the counter; the FIRST such pop
  drops any count contributed by a sibling counted chunk that is *actually still in flight*
  (e.g. re-enqueued via staging defer — `staging_deferred` re-queues the counted chunk onto
  the same queue, see 1484-1490/2988). `_maybe_ack_drained_abort` then sees outstanding==0 and
  fires the ACK, telling decode its KV slots are free while the deferred chunk's NIXL WRITE may
  still target them → later reuse hands those slots to another request, and the in-flight RDMA
  writes OLD KV over the new request's pages → cross-request corruption.
- Reachability requires: deferred release on + a counted chunk re-enqueued as staging_deferred
  + room cleared while it sits in the queue + another skipped chunk popping the count before
  the re-enqueued one drains. The `staging_counted` flag suggests the author intended dequeue-
  time idempotence but the pop at loop top(/bottom, 1549) breaks the invariant.

### F5. `_handle_abort_notification` status flip races the worker's settle barrier
- Bootstrap thread marks room Failed immediately on ABORT (3175-3187); a worker currently in
  `_await_handles` for that room is unaffected and later ALSO calls conclude_failure — benign
  double-notify path exists (`conclude_state` dedup in sender poll). OK.
- But `request_status` check uses `.get(..., KVPoll.WaitingForInput)` (723): an
  abort for a never-seen room registers a failure_record that is never cleaned
  (`failure_records.pop` only in failure_exception) → **slow memory leak of failure records**
  for garbage/attacker rooms on the ZMQ bootstrap socket. Minor.

### F6. Notification parsing — no unhandled-exception kill; update_transfer_status called only
from NixlKVReceiver.poll:3517, and malformed UTF-8/int raises propagate up through poll into
the decode scheduler's exception path (room fails). Batched notifications after the bad one in
the same `get_new_notifs()` map are lost → later chunks time out rather than complete cleanly.
Low severity (requires malformed peer).

### F7. `send_kvcache_staged` bounding: `dst_write_ptr = dst_staging_ptr + rank_offset` with
`rank_offset` and `per_rank_bytes` computed entirely from local params; only
`dst_staging_size < total_staging_needed` checked, but rank_offset+per_rank_bytes vs
total_staging_needed consistency depends on compute_staging_layout correctness
(common/staging_buffer.py — cross-check belongs to whoever owns that file).

## Known-bug verification status (upstream main)
- NIXL reload of invalidated peer: FIXED upstream (lines 1671-1709) — but see F1 (fix itself is racy).
- Deferred decode KV release: wired via `_maybe_ack_drained_abort`/`poison_deferred_ack_room` in
  worker loop — inspect common/conn.py (another agent's overlap) for ack target bookkeeping.
- Freeing KV after 30s timeout while RDMA in flight: NIXL side now has `_await_handles` settle
  BEFORE conclude_failure (good), but see F3 hang risk.
