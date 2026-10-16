# radix-emb findings: embedding / multimodal cache audit (HEAD as of branch)

Scope files: mem_cache/embedding_cache_controller.py, embedding_store.py,
multimodal_cache.py, index_key_cache.py + callers (managers/mm_schedule.py,
disaggregation/encoder/server.py, storage/mooncake_store/mooncake_embedding_store.py).

## FIXED (branch agent/radix-emb, commit 9743d29)

All of EMB-1..EMB-4 plus the escalated asymmetric-unpin bug (EMB-5 below).
Regression tests: test/registered/unit/mem_cache/test_embedding_cache_controller.py
(32 tests pass on CPU torch 2.14.1+cpu; the 6 store/controller-failing tests were
verified to FAIL against the pre-fix code).

### EMB-5 (escalated by lead; cross-request UAF) release_pool_views asymmetric unpin
get_pool_views pins only when entry exists AND READY; release_pool_views unpins any
present entry. FILLING->READY between get and release + another request pinning the
entry => release drops *the other request's* pin -> entry evictable under live read.
Fix: server releases only hashes with non-None views (same multiplicity);
controller docstring enforces contract. Fixes in _assemble_global_cache_cpu.

## CONFIRMED BUGS

### EMB-1 (pin leak -> permanent pool-page leak): `_assemble_global_cache_gpu` error path
File: python/sglang/srt/disaggregation/encoder/server.py ~L1600-1625
`load_to_device_async` pins a READY entry per cache-hit item inside the assembly
loop. If a later item raises `InternalError` (handle is None) or the copy setup
raises, the already-taken handles/pins in `copy_handles` are never released
(`wait_load_to_device` is after the loop, no try/finally).
Effect: entry.ref_count stuck > 0 -> entry never evictable -> its pages leak
permanently -> pool shrinks over time until OOM-of-pool (allocation_failures).
Trigger: entry evicted between prefetch and assembly (timing race) makes
load_to_device_async return None -> raise. Reachable in production.

### EMB-2 (FILLING entry leak -> permanent pool-page leak): store handles dropped
Files: server.py `_compute_global_cache_embedding` (~L1664-1700, keep_on_gpu path
and CPU path), controller store_to_pool_async.
Sequence: `_stage_global_cache_slices` allocates pool pages + creates FILLING
entries and returns d2h_handles. Then `_assemble_global_cache_gpu/_cpu` can raise
(InternalError, OOM, validation). `_launch_global_cache_insert` is skipped ->
`wait_store_to_pool` never runs -> entries stay FILLING forever.
FILLING entries are NOT in the evictable LRU, so `_evict_for_pool` can never
reclaim their pages: permanent pool shrink. Same if `_copy_tensor_to_pool` raises
mid-loop inside store_to_pool_async (entry already in self.entries, exception
propagates, caller `_stage_global_cache_slices` swallows and returns [],[] ->
handles for earlier entries dropped -> they leak FILLING too).

### EMB-3 (short Mooncake GET treated as success -> corrupted embedding cached+served)
File: storage/mooncake_store/mooncake_embedding_store.py L57-63, L100-110
`batch_get`/`batch_get_into_multi_buffers` treat any `res > 0` as success, but the
mooncake pybind contract (see mooncake_store.py:L1178) is "bytes read, negative on
error". A short read (0 < bytes < requested) passes as success; controller
`_finish_get` marks the entry READY; partial garbage is served for that image to
the current and all future requests hitting this hash (cross-request corruption).
Sibling of the known hicache "short reads treated as success" class.
Fix: compare against expected bytes (sizes[i] / sum(sizes[i])).

### EMB-4 (metadata leak): ongoing_prefetch never reaped on timeout/abandon
controller.prefetch inserts self.ongoing_prefetch[req_id]; popped only by a
successful check_prefetch_progress. On the 60s timeout path in
_wait_global_cache_prefetch or request abandon, the op entry stays forever
(slow unbounded growth; also op holds ptr/size lists). Also a repeated prefetch
for the same req_id silently overwrites the older op.

## CLEAN / DEFENDED CLASSES (need challenger)
- EMB-C1 asymmetric unpin in `_assemble_global_cache_cpu` raise path:
  get_pool_views pins only READY entries; release_pool_views is called for all
  hashes; unpin guards ref_count<=0 with a warning. FILLING entries always have
  ref_count==0 (all pin sites check READY), so no underflow corruption. Wart only.
- EMB-C2 mm hash collision (class 4): both scheduler cache (mm_schedule.py) and
  global cache key on sha256(content) truncated to 8B; identical images shared
  intentionally; mm_schedule.py additionally validates token counts against each
  item's expected counts and discards mismatched entries
  (_discard_mismatched_cached_embedding). combine_hashes uses tuple-hash of the
  content hashes. No per-request salt, but content-addressing makes wrong-content
  collision infeasible except adversarial 64-bit collision.
- EMB-C3 double-free of pool pages: freed ranges merge defensively; each entry is
  created/finished by exactly one op type (GET for prefetch-created, D2H wait for
  store-created), single-free holds. _finish_get can only free FILLING entries it
  created, FILLING never evictable, never pinned.
- EMB-C4 pin/unpin pairing in get/load/insert paths on the happy path; duplicates
  of the same hash in one call pin/unpin balanced.
- EMB-C5 MultiModalStaticCache size accounting: admission clones non-owning
  tensors (replace+clone when storage_size>data_size) so free() storage-accounting
  matches; LRU eviction subtracts same metric.
- EMB-C6 insert_batch PUT never races D2H: mark READY is gated on the entry's own
  copy event (wait_store_to_pool), and PUT only pins READY entries.

## index_key_cache.py (DSA) - notes
- cpu_copy/load_cpu_copy: `indices[::page_size] // page_size` -> page indices for
  arbitrary token location arrays; assumes locations are page-aligned-ordered
  (stride sampling of EVERY page_size-th loc). If a caller passes locations not
  starting at a page boundary, wrong pages offloaded (data loss on retract
  resume, not corruption). dsa_cache_layer_split owns locs; flagged for
  version-triage/dsa owner to confirm alignment invariant.
- get_local_buffer: `wait_until(layer_id - self.pool.start_layer)` can go
  negative if called with layer_id < start_layer (caller contract; not
  memory-unsafe by itself).
