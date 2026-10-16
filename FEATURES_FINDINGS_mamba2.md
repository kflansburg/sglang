# Mamba/hybrid audit — mamba2-auditor (backlog items 1-5, board #8)

Branch: agent/mamba2-auditor (branched off origin/dev @ 2198b3e).
Scope: Kimi K3 prod config — extra_buffer (non-lazy AND lazy), unified pool,
replayssm (no_buffer), int8 checkpoint pool, spec (EAGLE/DSPARK), page 64, disagg.

Conventions: file:line refs are against origin/dev 2198b3e (= HEAD state on this branch).

---

## Item 1: ping-pong/donate double-ownership (TRACING COMPLETE — see verdicts below)

### Structural map (verified by reading)
- Active decode state: `req.kv.mamba_pool_idx` — alloc'd once at req_pool alloc
  (memory_pool.py:1612), freed only in `free_mamba_cache` (memory_pool.py:1828-1831).
- Track/checkpoint slots: `req.kv.mamba_ping_pong_track_buffer` (len 2, or 1 in
  spec) — alloc'd lazily (`_alloc_ping_pong_buffer`, memory_pool.py:1762);
  non-lazy allocs both upfront; lazy allocs 1 (other = -1), second alloc'd at
  boundary (`mamba_lazy_prealloc_at_boundary`, schedule_batch.py:3503-3532).
- Donation (running req, extra_buffer, non-int8): mamba.py:624-629 —
  `new_slot = _alloc_mamba_slot()`; `donate_mamba_ping_pong_slot(req, new_slot)`
  (memory_pool.py:1802-1822) clones old `buf[keep_idx]` out as
  `mamba_value_donated` and writes `new_slot[0]` into `buf[keep_idx]`.
  Old slot goes to the tree via `insert_params.mamba_value`.
- Donation (finished req, extra_buffer): mamba.py:593-601 — `active_value =
  buf[keep_idx]` handed to tree, kept slot NOT replaced; cleanup
  `free_mamba_cache(req, mamba_ping_pong_track_buffer_to_keep=keep_idx)`
  frees mamba_pool_idx + the OTHER buf slot (1-keep_idx) at memory_pool.py:1850-1858.
- Failed insert (mamba_exist or no insert_result): running path frees
  `insert_params.mamba_value` via `_free_mamba_value` (mamba.py:681-684);
  finished path frees BOTH buf slots (keep_idx=None, memory_pool.py:1834).

### Verdict 1a: donate path — NO double-ownership found at HEAD (traced all interleavings)
Interleavings checked:
1. donate (running, keep slot S donated, replacement R) → boundary
   `mamba_lazy_post_decode_at_boundary(track_idx)` frees `buf[1-track_idx]`.
   After donation `last_track_idx` still points at the slot holding R; post-boundary
   frees the other slot only. S is tree-owned after insert, not in the buffer. OK.
2. donate + insert fails (mamba_exist) → `_free_mamba_value(S)`; buffer still
   holds R at keep_idx; rest of lifecycle consistent. OK.
3. finished + donation skipped in lazy spec (`mamba_lazy_checkpoint=False`,
   batch_result_processor.py:1570-1575) when a scatter wrote/may write the keep
   slot in flight → `release_kv_cache(req, checkpoint=False)` → `on_release`
   → `cleanup_after_caching_req(is_finished=True, insert_result=None)`.
   NOTE: this path passes insert_result=None so `mamba_value_inserted=False`,
   keep_idx=None → `free_mamba_cache(req)` frees pool_idx + both buf slots. OK.
4. Retract (decode, host backup): mamba slots are NOT freed; backup only copies
   `mamba_pool_idx` state (schedule_batch.py:2055-2067 `get_cpu_copy`), ping-pong
   buffer rides along untouched. Not a frees-twice; see Item 2 for accounting.
5. `set_mamba_ping_pong_slot` (memory_pool.py:1790-1799) keeps
   `req.kv.mamba_ping_pong_track_buffer` and the req-pool mirror
   `req_index_to_mamba_ping_pong_track_buffer_mapping` in sync; all mutators go
   through it (grep-verified: schedule_batch.py:3096 reads; :3529 sets via
   set_mamba_ping_pong_slot; batch_result_processor.py:1585/1655 via
   set_mamba_ping_pong_slot). No desync found.

### Verdict 1b: `_free_mamba_value` vs int8 pool split — potential double-free IF mixed
`_free_mamba_value` (mamba.py:536-540) frees into the int8 pool when
`int8_ckpt_pool is not None` else into `mamba_allocator` — this assumes
`insert_params.mamba_value` was allocated from the matching pool. Check:
- int8 pool non-None + finished: `mamba_value` comes from `_commit_int8_checkpoint`
  (int8 slot) — freed to int8 pool. OK.
- int8 pool non-None + running: `src_active` (active-pool slot) freed at
  mamba.py:619 DIRECTLY via `mamba_allocator.free(src_active)` — correct pool.
  `mamba_value` = int8 ckpt slot; on failure `_free_mamba_value` → int8 pool. OK.
No mismatch found: the two free sites match their allocation pools.

---

## Item 4 (C4 class): -1 / unchecked slot indices — PARTIALLY CHECKED

### 4a. REFUTED at HEAD: -1 tombstone destroyed by unified v2p translate (my early candidate)
Hypothesis: `set_mamba_track_indices_from_reqs` sets `mamba_track_indices[freed_rows] = -1`
(schedule_batch.py:2270-2275) as a scatter no-op marker; unified-pool
`UnifiedMambaSlotAllocator.translate` (unified_memory_pool.py:1150-1152) gathers
`virtual_to_physical[-1]` — PyTorch negative indexing returns the LAST row —
which would convert the -1 no-op sentinel into a LIVE physical slot id, and the
scatter kernel (`_fused_track_mamba_states_kernel`,
mamba_state_scatter_triton.py:78-79) would then write a live slot because its
`check_freed_slots` guard only skips `dst < 0`.

REFUTED: `MultiEndedAllocator.virtual_to_physical` is allocated with
`num_virtual_ids + 1` rows, all -1, and `virtual_to_physical[-1]` is the
deliberate trailing sentinel = -1 (unified_sub_pool.py:361-368; comment
"Page 0 is the padding anchor; the trailing row is the -1 sentinel",
:516-519 on grow). So translate(-1) = -1 and the kernel guard fires correctly.
The design is consistent (kernel comment at mamba_state_scatter_triton.py:77-78:
"Only the unified pool emits -1 tombstones (from the v2p translate)").

Residual LOW: this is an implicit cross-module invariant (table sized N+1 BECAUSE
translate is called with -1). Worth a one-line assert in
`UnifiedMambaSlotAllocator.translate` documenting `virtual_ids >= -1` contract.
Not a live bug.

### 4b. same class, unguarded sites (OPEN, tracing callers):
- `MambaPool.clear_slots/copy_from` (memory_pool.py:1145-1210): fused fast path
  `fused_clear_conv_slots(self._conv_slot_desc, indices)`; non-fused
  `t[:, indices] = 0`. NO bounds/-1 check on `indices` either way.
  Callers pass `mamba_clear_indices` built from `req.kv.mamba_pool_idx`
  only (schedule_batch.py:3154-3174) — `mamba_pool_idx` is always a valid slot
  when set (alloc asserts, freed to None). Window: can
  `_collect_deferred_mamba_cow_and_clear` run while `mamba_pool_idx is None`?
  That would AttributeError on `.unsqueeze`, not an OOB — fail-loud, benign.
- `MambaCheckpointPool.store_from_active/load_to_active`
  (mamba_checkpoint_pool.py:246-262): `c[:, ckpt_slots] = conv[i][:,
  active_slots]` — tensor indexing with int64; -1 would alias LAST slot
  (slot 0? no: last) — PyTorch negative index aliasing, silent.
  Callers: `_commit_int8_checkpoint(active_slots)` (mamba.py:527-533) and
  `load_to_active` from `_execute_mamba_deferred_ops` int8 path
  (model_runner.py:1926-1932). `active_slots`/`ckpt_slots` provenance:
  - donate int8: `req.kv.mamba_pool_idx.view(-1)` (finished) or
    ping-pong `src_active` (running extra_buffer). mamba_pool_idx valid; the
    ping-pong `src_active` is `buf[keep_idx]` — CAN be -1?
    `donate_mamba_ping_pong_slot` has a `_MAMBA_DEBUG_ASSERTS`-only check that
    the donated slot != -1 (memory_pool.py:1815-1821) — i.e., in production
    (asserts off) a -1 in buf[keep_idx] at donation time silently yields
    `mamba_value_donated == -1` → int8 path:
    `_commit_int8_checkpoint([-1])` → `store_from_active(..., active_slots=-1,
    ckpt_slot=K)` → `temporal.store_from_pool(temporal, -1, K)` copies the LAST
    active slot's state into checkpoint K → CACHED POISONED CHECKPOINT,
    later served to prefix matches (cross-request garbling, persistent in radix).
    OPEN QUESTION: can `buf[keep_idx]` be -1 at `prepare_for_caching_req`
    (running, extra_buffer)? `_select_finished_checkpoint` handles -1 via
    `cache_len <= token_ids_len`; running req: keep_idx =
    `mamba_last_track_idx`; buf[keep] == -1 happens in lazy mode when
    second slot never allocated (`mamba_last_track_idx` points at the only
    real slot though — init sets last_track_idx=0 via get_mamba_ping_pong_other_idx... LAZY sets it to 0 with buf=[slotA,-1], so buf[0] is real).
    Lazy sets `mamba_last_track_idx=0` initially (memory_pool.py:1783-1788
    lazy branch: `0 if lazy else other_idx(0)`) — buf[0]=slotA real.
    After boundary post: track_idx=1 → last=1, buf=[S? , slotB]; buf[1] real.
    After donation: replacement written at keep_idx. So `buf[last_track_idx]`
    SHOULD be invariantly non-(-1) for a running req... PROVING the invariant
    needs the harness (below) over the full transition set. If ANY edge lets
    last_track_idx point at a -1 (e.g., prealloc failure sets
    `mamba_last_track_seqlen=None` but does NOT reset last_track_idx — and the
    OTHER slot stays -1 — then donation at a short key falls back via
    `_select_finished_checkpoint`... that's the finished path, which handles
    overshoot via prev/other slot that may be -1: `_select_finished_checkpoint`
    picks `get_mamba_ping_pong_other_idx(keep_idx)` = the OTHER slot which in
    lazy mode is -1 whenever the transient second slot was already freed by
    post_decode_at_boundary!!). *** See Finding M2-1 below. ***

### Finding M2-1 (CANDIDATE): finished lazy req with overshoot donates -1 slot
- Site: `MambaComponent._select_finished_checkpoint`
  (components/mamba.py:543-562) + `prepare_for_caching_req`
  (components/mamba.py:590-601, extra_buffer finished branch).
- Interleaving:
  1. Lazy extra_buffer req runs decode. At each boundary,
     `mamba_lazy_post_decode_at_boundary` FREES the non-keep slot and sets
     buf[other] = -1 (batch_result_processor.py:1646-1658).  So steady-state
     lazy buffer is [slotX, -1] or [-1, slotX] with
     `mamba_last_track_idx = mamba_next_track_idx = <the real one>`.
  2. Req finishes at token_ids_len T with a tracked checkpoint whose
     `cache_len = req.kv.mamba_last_track_seqlen` OVERSHOOTS T
     (`cache_len > token_ids_len`), e.g., finish lands between the last decode
     boundary and its checkpoint seqlen (the checkpoint grid is page-granular
     so cache_len is a multiple of page_size; abort/finish mid-page makes
     T not page-aligned; extra_buffer doesn't floor cache_len to the key when
     is_finished).
  3. `_select_finished_checkpoint`: pool.mamba_ping_pong_track_buffer_size == 2
     (non-spec extra_buffer lazy), `previous_cache_len = req.kv.mamba_prev_track_seqlen`.
     In LAZY mode, post-boundary sets "mamba_prev_track_seqlen stays None"
     (comment at batch_result_processor.py:1521-1523: "Lazy frees the slot it
     stops tracking, so nothing names the previous checkpoint there;
     mamba_prev_track_seqlen stays None"). So `previous_cache_len is None`
     → returns None → "donate nothing", safe?
     → BUT: is `mamba_prev_track_seqlen` REALLY always None in lazy?
     It is set at batch_result_processor.py:1523 for the NON-lazy branch only,
     and at schedule_batch.py:3147 (extend tracking sets
     `req.kv.mamba_last_track_seqlen`; prev only in the non-lazy line above).
     Also free_mamba_cache resets both to None. Looks None-invariant for lazy.
  4. So for PURE lazy the -1 donate can't fire (prev is None → bail).
     BUT the mask in `prepare_for_caching_req` uses
     `self.cache.enable_mamba_extra_buffer` only — not lazy — and spec decode
     extra_buffer non-lazy sets buffer_size 1? (memory_pool comment at
     free_mamba_cache: "ping-pong buffer size is either 2 (normal) or 1 (spec
     decode)"). With buffer_size==1, condition
     `pool.mamba_ping_pong_track_buffer_size != 2` → return None. Safe.
  5. Remaining sub-case: NON-lazy extra_buffer finished overshoot: keep_idx =
     last_track_idx (real slot), other_idx slot real too (both alloc'd
     upfront and never -1 in non-lazy). `previous_cache_len` set on boundary
     transitions. donate picks the OTHER slot if prev <= key < last.
     The OTHER slot was freed? NO — non-lazy keeps both slots for the req's
     whole life. Slots are freed only by `free_mamba_cache`.
     → REFUTED for non-lazy; REFUTED for lazy/spec via prev=None leaf-set.
- VERDICT: M2-1 REFUTED on close trace (the -1 leaf requires prev<=key while the
  other slot is -1, which lazy's prev=None prevents and non-lazy always has real slots).
  The `_MAMBA_DEBUG_ASSERTS`-gated check at memory_pool.py:1815 stays the only
  tripwire; adding an always-on cheap guard is defense-in-depth worth MR'ing later.

---

## Item 2: extra_buffer slot accounting — boundary prealloc vs retract/abort (CHECKED)

- `mamba_lazy_prealloc_at_boundary` (schedule_batch.py:3503-3532): allocates the
  transient second slot ONLY if `seq_lens_cpu[i] % interval == 0` and
  `buf[other_idx] == -1`. On alloc failure: sets `mamba_last_track_seqlen=None`
  (degrades tracking), no slot.
- Overlap note in code (:3516-3520): if `buf[other_idx] != -1` the in-flight
  previous forward's post step hasn't freed yet → skip. This re-checks CURRENT
  buffer state, so no double-alloc.
- Interleave with abort: if req is aborted mid-step, `free_mamba_cache` frees all
  non-(-1) buf slots including the transient one. No leak.
- Interleave with retract: mamba slots are kept (backup path, Item 1#4). A
  retracted-then-resumed req resumes with its transient slot held; boundaries
  recompute correctly. Accounting: every retracted req holds 1-2 EXTRA slots
  beyond its active one while retracted — capacity pressure only, bounded by
  max retracted reqs × 2 slots; alloc asserts loudly (memory_pool.py:1609) if
  exhausted. Not corruption. VERDICT: clean (with capacity note).
- Spec variant `mamba_lazy_spec_prepare` (schedule_batch.py:3534-3577): same
  pattern; failure leaves scatter in place into keep slot
  (`track_positions.append(req.kv.mamba_next_track_idx)`) — and the in-flight
  verify then OVERWRITES the keep slot; the code marks
  `mamba_last_track_seqlen=None` so no future radix key names a stale slot. OK.
  HOWEVER — see Item 3 (replayssm + spec ring) for the remaining open chain.

## Item 3: replayssm ring accounting vs spec draft paths (IN PROGRESS)
- `prepare_for_caching_req` ReplaySSM (no_buffer) finished path subtracts
  `write_pos[mamba_pool_idx]` from cache_len and zeroes write_pos
  (components/mamba.py:574-586). Running path unrelated.
- TO CHECK NEXT: ring cursor across retract/restore (backup does NOT capture
  replayssm_write_pos — backup_kv_cache copies state via get_cpu_copy; the ring
  rawv/d buffers live in the pool keyed by slot; slot kept on retract so ring
  survives? verify ring buffers keyed by pool slot, not req), vs
  test_replayssm_ring_accounting.py coverage of extra_buffer-ON + linear-spec-ON.

## Item 5: int8 checkpoint pool alloc/free symmetry (CHECKED at read depth)
- `MambaCheckpointPool` embeds `MambaSlotAllocator` (mamba_checkpoint_pool.py:222).
- `_alloc_int8_ckpt_slot` (components/mamba.py:518-524): evict loop via
  `self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))` then assert non-None.
  NOTE: uses `self.cache.evict` not `evict_for_alloc` like `_alloc_mamba_slot` —
  unverified whether evict(mamba_num=1) guarantees an int8-slot eviction vs an
  active-pool eviction. If `cache.evict` treats mamba_num as ACTIVE-pool slots,
  int8 pool could stay full → assert crash (availability only). TO VERIFY against
  unified_radix_cache.evict signature.
- Tree eviction of mamba nodes: `_free_mamba_value(indices)` at
  components/mamba.py:994 routes to int8 pool free or active free — matches the
  store site (int8 nodes store ckpt slots; non-int8 store active/pp slots). OK.
- `MambaSlotAllocator.free` (allocator/mamba.py:82-86): NO double-free guard,
  NO -1 validation, NO bounds check, NO dup filter (pools-lead's hardening
  1a92c7b is NOT on dev yet — adds overflow assert + debug dup sweep).
  If any caller passes -1 (e.g., from a ping-pong buf freed while scan filters
  miss it), free(-1) inserts -1 into free list; alloc later returns -1 as a
  slot; downstream `t[:, -1]` aliases the LAST slot → silent corruption.
  All current free sites mask out -1 (`free_mamba_cache` lazy filter at
  memory_pool.py:1859-1863, and other free sites use owned values) — traced.

---

## NEXT (queue)
1. Harness: simulate the exact slot lifecycle (extra_buffer lazy, spec non-lazy,
   donate, boundary, finish/abort, retract) as pure-python index math against a
   slot-ownership oracle; assert single-ownership. (deliverable for item 1-3)
2. Verify `cache.evict` vs `evict_for_alloc` semantics for `_alloc_int8_ckpt_slot` (item 5).
3. replayssm spec ring chain (item 3).
4. multi_layer_eagle_worker_v2.py:594-598 `_apply_deferred_mamba_init_to_draft_pools`
   calls `pool.clear_slots(clear)` / `pool.copy_from(cow_src, cow_dst)` on DRAFT
   pools with the TARGET batch's indices — if the draft pools are sized
   differently (draft models have smaller pools!), indices valid for the target
   pool could exceed the draft pool size → OOB. ALSO no translate call. TRACING.
