# HiCache prefetch/controller audit findings (agent: prefetch-ctl-audit)

Scope covered: cache_controller.py (prefetch/backup/write paths, threads, reset/detach),
storage_prefetch.py, hicache_key_scheme.py, hicache_auto_size.py, hicache_storage.py
 callers in unified_radix_cache.py (check_prefetch_progress, drain_storage_control_queues_impl,
 release_aborted_request, _handle_prefetch_result, _execute_and_commit_kv_backup),
 buffer_mode/pipeline.py callgraph (entry points only), scheduler.py (event loops,
 get_new_batch_prefill prefetch gating), mooncake_store.py zero-copy paths.
Production config targeted: write_through, page_first_direct host layout, direct io
 backend, mooncake storage w/ group semantics, prefetch stop policy TIMEOUT, page 64,
 fp8_e4m3, disaggregated prefill/decode. Versions verified: latest upstream main
 (3420bae), v0.5.14 tag, v0.5.20 tag (cloned from github).

=====================================================================
CONFIRMED BUGS
=====================================================================

## F1. Mooncake short reads counted as successful page fetches (PRIVACY/CORRUPTION)
File: python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py
`_batch_postprocess` (upstream line ~1173):
    for gets: `all(res > 0 for res in group)` — Mooncake's batch_get_into contract
    returns *bytes read* per key. Any short read (1..page_bytes-1) is treated as a HIT.
Effect chain: `_page_get_zero_copy` counts the page fetched -> PrefetchAck completed
tokens include it -> host page (containing partially-written garbage: stale bytes from
the pool's previous tenant) is inserted into the host radix tree / consumed by the
request. Result: garbled model output + stale cross-request KV bytes served as fresh
context, undetected.
Fix: compare `res == expected_size` per key (buffer_sizes is available in the caller);
treat anything else as failure (fail-soft -> prefix truncation, which the pipeline
already handles).
Affected versions: PRESENT in v0.5.14 (line 986), v0.5.20 (line 1059), latest main
(line ~1201). Same `>0` check exists for both K and V sub-entries via key_multiplier.
This is the known class "Mooncake store short reads treated as success" — still NOT
fixed upstream anywhere.

## F2. ClearDiskCache / admin clear runs with in-flight prefetch+backup (minor)
`HybridCacheController.clear_storage_backend` (hybrid_cache_controller.py:428) calls
`storage_backend.clear()` unconditionally, no drain/terminate of in-flight storage ops.
Backup thread may be mid `batch_set_v1`, prefetch io mid `batch_get_v1` against keys
being deleted. Backend-dependent behavior (mooncake: per-key errors -> partial pages,
handled fail-soft; but a clear() that resets group state mid-group-set could error or
corrupt group semantics). Only reachable via admin API while traffic flows; low
severity, worth a guard (reject clear while backup_queue/prefetch in flight).

=====================================================================
VERIFIED GUARDED / CLEAN IN LATEST UPSTREAM (with assumptions)
=====================================================================

## V1. Prefetch TIMEOUT freeing pages under in-flight mooncake DMA — GUARDED upstream
Known class "abort leaking pins / H2D into freed buffers".
State machine for one allocated prefetch (cache mode):
 - scheduler drain: `allocate_storage_hit` allocs host_indices, op handed to io thread
   (operation.host_indices set before io thread can see the op).
 - io thread writes batch k pages, posts per-batch progress ack (MIN-reduced across
   ranks in prefetch_sync_thread), posts completion ack in `finally` AFTER
   _page_transfer returns.
 - terminate (timeout/abort): scheduler `_handle_prefetch_result` frees/inserts ONLY
   [0:completed_tokens] where completed_tokens comes from drained acks (monotonic,
   asserted `<=`). Tail [completed:] is NOT freed by the scheduler; it is released by
   the ack drain when the completion ack arrives
   (PrefetchOperation.ack_releases_incomplete_host_indices = True,
    hybrid_cache_controller.py:182). The completion ack is posted after the last
   batch_get_v1 returned, i.e. after DMA has stopped touching the tail.
 - abort (`release_aborted_request`) same partition: [0:completed] retired now, tail at
   completion ack.
Assumption: mooncake `batch_get_into` is synchronous (DMA complete when it returns) —
true per its contract (per-key return codes). Given that, no DMA-into-freed-page
window remains in latest upstream.
CAVEAT: `completed_tokens` is only assigned in the ack drain; progress acks are
rank-MIN reduced, so [0:completed] is a rank-agreed prefix. The one-batch lag
under-counts (never over-counts). OK.

## V2. v0.5.14 (production version) prefetch lifetime — partition-safe, different mechanism
Verified in /tmp/sgl-v0514: io thread itself releases the tail
(`prefetch_io_aux_func`: append_host_mem_release(host_indices[completed:])) AFTER
_page_transfer returns; scheduler timeout path frees only pages < completed_tokens
(tree-matched + [min:completed] cross-rank-excess). Partitions are disjoint both
per-rank and temporally because `increment()` stops once terminated, freezing the
boundary. Write-then-increment in _page_transfer fails closed (page written, not
counted, freed later — wasted but not corrupted).
Residual v0.5.14 nits:
 - controller `terminate_prefetch` reads completed_tokens/hash_value without the op
   lock (line 889); GIL makes the int read atomic; hash_value was already truncated by
   the query thread BEFORE the op was exposed to the drain/scheduler — no torn list.
 - abort releases [0:completed_tokens] with NO cross-rank MIN (hiradix_cache 1727);
   safe because pages are rank-local.

## V3. write_through host-page lifetime during backup — GUARDED
`_execute_and_commit_kv_backup`: `inc_lock_ref(node_id)` before tracking; the host
pages of a node pending storage backup cannot be evicted until `_drain_backup` sees
the ack and `dec_host_lock_ref`s. backup_thread reads only host pages of locked
nodes. Double-write of the same node: chain walks skip nodes whose transfer set is
empty; mooncake set is set-if-absent (`batch_exist` filter). Node/host-range aliasing
across two queued backups of the same span cannot happen while the first op holds
the lock... EXCEPT the lock covers eviction, not duplicate enqueue; duplicate enqueue
is prevented by commit_backup marking the node (build_backup_spec returns empty.
Verified at unified_radix_cache 1615-1645.

## V4. Thread-boundary audit of shared maps
- ongoing_prefetch / ongoing_backup / prefetch_tokens_occupied / pending_hit_allocs:
  scheduler thread only. io threads communicate via Queues. OK.
- PrefetchOperation: _terminated_flag locked; hash_value written by query thread
  before the op is put on prefetch_hit_queue (queue = publication barrier); read by
  scheduler after dequeue. completed_tokens written only in ack drain (scheduler).
  storage_hit_count likewise pre-publication.. No data race found beyond the
  benign ones listed.
- `CacheOperation.counter`, `StorageOperation.counter`: incremented without lock,
  called from scheduler + io threads? StorageOperation created in scheduler thread
  (write_storage) and prefetch() — scheduler only. CacheOperation.scheduler only. OK.
- reset()/detach_storage_backend(): documented "no in-flight requests" requirement,
  joins threads with 10s timeout, raises if still alive (won't corrupt; blocks
  re-attach). drain-before-stop ordering in StorageAttachment.detach is correct.

## V5. hicache_key_scheme / sizing — no memory-safety issue
- Unified key scheme: namespace digest (model,dtype,page_size,layout,head/layer grid)
  prefixes keys; mismatches miss rather than collide. page_size in namespace =>
  page-64 vs other never share keys. msgpack deterministic encoding versioned.
- hicache_auto_size: ratio reduced to fit budget with 5% slack, MIN-reduced across PP
  world group (single collective, avoids per-pool deadlock), ratio<=0 raises. Float
  rounding: ratio is float64; an undersized host pool vs assumptions would raise at
  pool build, not at runtime indexing. `_draft_bytes` scales draft sidecar by
  target.size//draft.size (integer floor; underestimates by <1 slot per draft — pool
  builders validate). No OOB found.

## V6. storage_prefetch.py (retry bookkeeping) — pure dict/state machine, no memory
 lifetime. `StoragePrefetchRetries.pop_ready` cancels the queue head's speculative
 retry so it can never starve admission. No issue.

=====================================================================
NOTES / HANDOFF
=====================================================================
- decode_kvcache_offload_manager.py builds a PLAIN HiCacheController (not Hybrid)
  for --disaggregation-decode-enable-offload-kvcache; disagg team should check its
  ack/free discipline separately (it predates ack_releases_incomplete machinery; in
  latest upstream the drain code that checks the flag lives in unified_radix_cache
  which the offload manager may not use).
- Buffer-mode paths (pipeline.py stage_completed_prefetch / allocate_and_queue_prefetch)
  NOT in our production surface (cache mode); park points and staged holds were not
  deeply audited — flagged for whoever owns buffer mode.
- Known class "prefetch policy timeout": current implementation is rank-consistent
  (PP0 wall clock, TP MAX, PP reduce inside @rank_consensus) and the free discipline
  above makes it memory-safe upstream. v0.5.14's timeout frees only <completed — also
  safe as analyzed in V2.
- Mooncake group semantics (enable_group_semantics=true in prod): set path expands
  group ids per key; failure of one sub-entry fails the group in _batch_postprocess
  for sets (res==0 strictly). Get path's `>0` bug (F1) applies per sub-entry too.

Repro status: F1 is a one-line semantic bug verified against the Mooncake client
contract (bytes-read returns counted as hits); a CPU repro would require a stub
backend returning short counts — trivially constructible; recommend the fix land with
a unit test asserting short counts truncate the prefix (fail-soft) instead of
counting a page.

=====================================================================
LEAD DIRECTIVE FOLLOWUP (2026-10-16): test pins + siblings
=====================================================================

## Test pins verified for the guards I audited

- "abort leaking pins / H2D into freed buffers": PINNED by
  test_unified_radix_cache_unittest.py::test_release_aborted_request_l3_prefetch_io_in_progress
  (blocks io thread inside _page_transfer, aborts, asserts NOTHING is queued for
  release while io in flight, then asserts full release + pool-size restoration after
  the io thread finishes + ack drain). Would fail under pre-guard immediate tail
  release. Also ..._io_done (post-completion variant). SWA extra-pool variant
  covered by the same test (SWA-only fixture + extra release queue).
  Siblings hunted: timeout-terminate (same _handle_prefetch_result path), hybrid
  all-or-nothing discard (_retire_ongoing_prefetch with completed prefix only),
  abort-before-alloc (revoke_pending_prefetch, no buffers), sidecar-terminated-early
  skip (is_terminated + kv_completed==len(hash) gate in _page_transfer_sidecar).
  All follow the same partition: scheduler only ever owns [0:completed]; tail and
  sidecar buffers are released post-completion-ack.
- "prefetch timeout races": pinned by consensus tests in
  test_unified_radix_cache_unittest (uses _can_terminate_prefetch mocks) +
  test_prefill_memory_budget / test_hicache_load_back_timing cover gating; the
  free-discipline half is pinned by the two abort tests above (same code path).
- "write_through ownership": covered by test_hicache_staged_write_back_dispatch.py
  (1374 lines; dispatch/staging) + ongoing write-through tracking in
  unified_radix_cache (see ~3350: ongoing_write_through consensus reduce).
- hicache_auto_size: pinned by test_hicache_auto_size.py (141 lines).

## NEW SIBLING CANDIDATE (needs verification; not yet confirmed)

## C4. PrefetchAck.pool_hits is NOT rank-reduced (TP-divergence hazard, hybrid sidecar pools)
`CacheController._reduce_prefetch_ack` MIN-reduces only completed_tokens. The
sidecar `pool_hits` dict (SWA/mamba/DSA per-pool fetched-page counts from this
rank's batch_get_v2) is forwarded rank-locally. `_check_hybrid_prefetch_result`'s
all-or-nothing decision consumes `pool_storage_result.extra_pool_hit_pages`
WITHOUT a cross-rank MIN. Path: TP>1 + SWA(sidecar) + transient per-rank sidecar
read failure where KV completes identically on all ranks -> rank A decides
"all_succeeded" and inserts into its host tree; rank B discards and retires
(releases) the same buffers. check_prefetch_progress is @rank_consensus
same_results=True, which agrees only on the returned bool, not on the differing
side effects inside _handle_prefetch_result. Outcome if it fires: divergent host
radix trees / free pools across TP ranks; in practice likely surfaced as a later
crash/violated invariant rather than silently served wrong bytes (consensus
checks + host-pool sanity checks), but the divergence itself is real.
Production relevance: SWA + hicache + TP>1 are all in our deployment matrix.
Status: CANDIDATE. Confidence the reduction is absent: high (read both functions).
Confidence it's exploitable in practice: medium (needs asymmetric sidecar failure).
Suggested fix: include per-pool hit counts in the PrefetchAck MIN-reduce in
_reduce_prefetch_ack (dict over fixed pool-name set), or clamp to rank-min inside
_check_hybrid_prefetch_result via _all_reduce_attn_groups.

## Mooncake short-read (F1) — OWNED by mooncake-audit
Independently confirmed by mooncake-audit with 4/4 repro incl. siblings in get(),
batch_get(), embedding store (their post #133, findings in
audit/hicache_mooncake_findings.md). I defer the fix+test to them.

## Cross-refs
- host reclaim order + free-list double free: hostpool-audit owns (anchors listed).
  My drain-path reading found release funnel (host_mem_release_queue -> scheduler
  drain -> mem_pool_host.free single-threaded) which their tests pin.

## C4 verification detail (for lead review)
- completed_tokens: MIN-reduced across ranks in CacheController._reduce_prefetch_ack
  (cache_controller.py:1414-1426) and folded into ack before the scheduler ever sees it.
- pool_hits (sidecar fetch outcome): produced rank-locally in
  HybridCacheController._page_transfer_sidecar (count_pool_hits of this rank's
  batch_get_v2), forwarded by both sync-thread variants without any all_reduce:
  base prefetch_sync_thread_func (cache_controller.py:1397-1414),
  hybrid override (hybrid_cache_controller.py:~1820). Consumed at
  unified_radix_cache.py:3079 (update_extra_pool_hit_pages) and :2433
  (_check_hybrid_prefetch_result all-or-nothing), all rank-local.
- Asymmetry window: hit QUERY boundaries are rank-folded (query_pool_hit_pages),
  so all ranks agree SWA is present; the FETCH must fail short on exactly one
  rank (transient per-rank IO error). Then: rank-fail discards + releases host
  buffers [0:completed] (and sidecar buffers), rank-ok inserts the same host
  buffers into its tree as resident KV. Divergent free lists -> subsequent
  divergent allocations/matches -> TP ranks serve different KV -> garbage/hang.
- Shipped TP-consistency test test_tp_swa_prefetch_dropped_when_peer_misses
  (~l.6383) simulates a UNIFORM miss via fake _reduce (all ranks drop) — it
  cannot generate the asymmetric case because it patches the reduce function
  itself rather than producing genuinely different per-rank fetch results. No
  shipped test pins asymmetric-fetch consistency.
- Suggested fix: in _reduce_prefetch_ack, when ack.pool_hits is not None,
  all_reduce MIN per pool name over prefetch_completion_sync_groups (pool-name
  set is stable for a given op), before forwarding. Guards all-or-nothing and
  any per-pool truncate decision simultaneously.

## C4 RETRACTED AT HEAD (2026-10-16, lead review verdict: false positive) + REAL VERSION WINDOW FOUND
Review correction (verified re-reading the code): HybridCacheController._reduce_prefetch_ack
(hybrid_cache_controller.py:1795-1812) OVERRIDES the base method and DOES rank-MIN
fold pool_hits over the fixed PoolName universe (.get(pool,0), `is not None`
guard for failure acks). Every production sidecar fetch ack flows through the
override's sync thread, so at HEAD (and v0.5.20) C4 is NOT reachable. My repro
hand-injected divergent pool hits into the decision function, bypassing the
fold — it proves the decision logic needs synced inputs, which the controller
provides. Credit: the override already had the guard (author's comment even
spells out the empty-dict failure case). My fix 8215dc6 was REVERTED (1a5e597):
it also double-reduced hybrid acks and could wedge on failure acks (empty dict
skips the collective while peers enter it).

VERSION WINDOW — the ACTUAL live bug (mamba sidecar path):
hi_mamba_radix_cache.py::check_prefetch_progress (mamba deployments) reduces
ONLY completed_tokens across TP ranks, then reads rank-local
operation.pool_storage_result.extra_pool_hit_pages[MAMBA] >= 1 as `mamba_loaded`
to decide mamba-slot attach/free with NO cross-rank sync:
- v0.5.14      hi_mamba: VULNERABLE (l.~1815-1866: rank B may free the mamba
               slot + skip attach while rank A attaches -> divergent trees).
- v0.5.15.post1 hi_mamba: VULNERABLE (identical shape, l.~1815-1861). NOTE:
               the UNIFIED cache path in the same tag IS guarded
               (unified_radix_cache.py:2021-2033 packs completed+sidecar hits
               into one MIN all_reduce at terminate).
- v0.5.17      hi_mamba: VULNERABLE (identical shape, l.~1869-1917).
- v0.5.20      hi_mamba_radix_cache.py file REMOVED -> class resolved by rewrite.
- HEAD         class resolved (unified hybrid controller folds pool_hits).
Trigger: TP>1 mamba model + hicache storage prefetch + asymmetric per-rank
mamba-sidecar batch_get_v2 failure (exactly one rank short) after all ranks'
hit queries agreed present. Consequence: divergent mamba host trees + freed
slot reissued on one rank while live on the other -> cross-request state mix.
No fix needed on HEAD; if prod patches are still taken for 0.5.15/17 mamba
branches, the equivalent fix is to extend the MIN reduce in
hi_mamba check_prefetch_progress to include the mamba hit (as unified did).
NOTE: the shipped fixture-based TP tests can NOT run in this pure-CPU sandbox
(the produce stage needs native D2H kernels gated on _is_cuda; sgl_kernel PyPI
wheel requires libnvrtc). Fixture spliced earlier was reverted. The mock-level
repro exercises the exact decision function whose input is the unsynced dict.

## ENV: lifecycle two-rank failure = sandbox hostname artifact (not a bug)
test_storage_prefetch_lifecycle.py::test_two_rank_completion_capacity_and_anchor_loss
fails on clean HEAD in CPU sandboxes because gloo ProcessGroupGloo constructor
fails with "File name too long" when the machine HOSTNAME is long (65-char hex
in swarm sandboxes -> gloo unix-socket path over the ~108-byte sockaddr limit).
Repro: 2-rank spawn init alone fails; `hostname scpu` -> all 10 lifecycle tests
PASS (verified). Verdict: environment artifact; recommend agents run
`hostname scpu` before CPU gloo tests. (audit/cpu_test_env_recipe.md updated.)

## C5 CANDIDATE (pre-existing on HEAD, repro = shipped test)
test_pp_prefetch_ticket.py::TestPPPrefetchTicket::test_allocation_error_preserves_ack_sequence_and_next_ticket
RETRACTED (2026-10-16): environment artifact, NOT an SGLang bug. The failure
chains to the missing native-hash JIT: without ninja on PATH,
get_storage_hash_str raises inside pp_prefetch_command_thread_func BEFORE the
alloc attempt; the op's (empty) hash_value makes the except-path early-return
without mark_terminate -> is_terminated() False. With PATH including ninja
(and libssl-dev, python3.13-dev installed; build cache warm):
- the single test passes, - the whole test_pp_prefetch_ticket.py passes 11/11.
Verified the intended behavior directly: alloc side_effect [kv, KeyError(SWA)]
-> KV rolled back via free(kv, pool=KV), op terminated w/ empty host_indices,
next ticket proceeds. Verdict: alloc-error discipline for PP tickets is
CORRECT at HEAD; the recipe (audit/cpu_test_env_recipe.md) now covers the
toolchain needs.

## Buffer-mode pipeline spot check (thin-area follow-up, NOT production surface)
Reviewed: BufferModePipeline.stage_completed_prefetch, release_staged_hold,
allocate_and_queue_prefetch/_allocate_prefetch_host_indices/_defer/_queue,
pop_prefix_ctx, occupancy accounting. Staged holds exist only POST-completion
(io threads done) -> abort frees synchronously via release_staged_hold
(anchor pin + KV/aux bounce + occupancy grant, one-shot pop prevents re-entry);
alloc failure paths release the bounce immediately or defer without ownership;
oversize declines and revokes. No async-DMA-after-free edge spotted in these
functions. NOT exhaustive: load-back H2D overlap with staged evictions and the
anchor-lock re-take paths were not deeply traced. Verdict: structurally sound
by the same post-completion ownership pattern as cache mode.

## FINAL COVERAGE SUMMARY (audit scope)
1. Mooncake short reads = success (F1) — CONFIRMED, owned by mooncake-audit
   (fix scope incl. get/batch_get/embedding siblings).
2. Timeout/abort host-page lifetime (H2D into freed buffers class) — GUARDED
   at HEAD; pinned by test_release_aborted_request_l3_prefetch_io_in_progress;
   v0.5.14-era code independently partition-safe. CLEAN.
3. write_through eviction-vs-backup-read — GUARDED by inc/dec_host_lock_ref
   pairing. CLEAN.
4. Thread boundaries/queues/maps — reviewed; benign items listed. CLEAN.
5. hicache_key_scheme / hicache_auto_size / storage_prefetch retries — CLEAN.
6. C4 TP sidecar pool_hits divergence — REFUTED at HEAD/v0.5.20 (hybrid fold);
   REAL live window on shipped prods via hi_mamba_radix_cache (v0.5.14-17,
   mamba models) — reported to version-triage; no HEAD fix needed.
7. C5 PP-ticket alloc error — RETRACTED (env artifact); alloc rollback
   discipline verified correct by direct simulation. CLEAN.
8. Buffer-mode staged-hold spot check — structurally sound (see above).
Env: lifecycle two-rank failure = hostname/gloo artifact (`hostname scpu`);
recipe in audit/cpu_test_env_recipe.md.

## FINAL: buffer-mode staged-hold lifetime verdict (lead's bounded task)
VERDICT: CLEAN (no host-slot/pin lifetime bugs found in staged-hold paths),
with the assumptions below. Pass restricted to lifetime edges:
- staged prefetch (bounce + aux staging + anchor device pin + occupancy grant):
  every exit path frees exactly-once on the scheduler thread:
  abort/defer-overflow -> release_staged_hold (one-shot pop + _free_staging_now);
  admission success -> ownership VERBATIM transfer to ongoing_buffer_load_back
  (del staged_prefetches + insert ongoing_buffer_load_back under same flow).
- in-flight load-back: bounce freed ONLY at H2D fill ack
  (try_finish_load_back: ack.finish_event.synchronize() BEFORE
  _free_completed_load_back_buffers) -> no H2D-into-freed-bounce.
  abort while H2D in-flight: bounce stays until ack, then freed; accounting
  early-returns (hit-remaining already discarded). Device slots belong to the
  tree by then (published at admission; ownership checked by the fail-stop
  RuntimeError in init_load_back comparing canonical indices).
- reset(): drops maps; same "no in-flight requests" precondition as cache mode
  (verified equal assumption).
- aux double-free guarded: _free_staging_now skips transfers with
  indices_from_pool set (borrowed KV rows) and frees each aux on its own pool.
ASSUMPTIONS: (1) single scheduler thread runs all these paths
(release_aborted_request @rank_consensus call site + check_hicache_events ack
drains; verified by call-site reading); (2) host-pool free() guards now cover
double-free defensively (hostpool-audit's fix); (3) reset precondition enforced
by callers; (4) buffer_only mode not in production surface (cache mode is) —
no fixture-level repro was attempted.
