# Mooncake store backend audit — findings (agent: mooncake-audit)
Scope: python/sglang/srt/mem_cache/storage/mooncake_store/*, storage/__init__.py, storage/backend_factory.py, mem_cache/l2_transfer.py. Production: --hicache-storage-backend mooncake, prefetch policy timeout, enable_group_semantics=true, store SHARED ACROSS REPLICAS with backend tag, page 64, fp8_e4m3, page_first_direct, direct io.

Ranking: P1 = memory-safety / correctness in prod config; P2 = correctness edge / leak; P3 = robustness.

## P1-1. Short reads treated as success on ALL get paths (known class "Mooncake store short reads treated as success" — STILL PRESENT upstream HEAD)
Contract (documented in mooncake_store.py `_batch_postprocess` docstring, citing mooncake pybind_client.h): batch_get_into result per key = "number of bytes read on success, or a negative value on error". So 0 < r < requested_size is possible and means PARTIAL data in the destination buffer; remainder is whatever was in the host page before (stale KV from an evicted request, or another request's data in a shared host pool).
Callers treat any positive r as full success:
- `_batch_postprocess(results, is_set_operate=False)`: `all(res > 0 for res in group)` — mooncake_store.py:1177. Used by batch_get_v1 (prefetch path into host pages -> later H2D copied to device and served). A partial-fetch host page is marked hit and served => stale/garbled KV, cross-request leakage.
- `get()`: `return get_result[0] >= 0` (mooncake_store.py:~1352) — r==0 (zero bytes!) counts as success. `>= 0` also accepts negatives? No: negative fails, but r==0 success is wrong.
- `batch_get()`: `if get_result[i] < 0: return i // key_multiplier` — partial positive counts as a full page, prefetch boundary includes it.
- Embedding fetch: mooncake_embedding_store.py batch_get/batch_get_into_multi_buffers map `res > 0` -> True; embedding_cache_controller._finish_get treats True as fully fetched embedding => partial/garbled embedding used for the image tokens (mm paths on Gemma/Kimi are in the production surface).
Correct check: success iff r == requested_size for every component (or r >= size for plain get); anything else (including r==0) must be failure.

## P1-2. batch_set_v2/_batch_io_v2(is_set=True): exists-before-put is silently dropped when the exists RPC fails/glances over
In `_batch_io_v2` is_set branch: `io_results = [0 if state == 1 else -1 for state in exist_result]`; only `state == 1` treated as pre-existing. mooncake batch_is_exist returns 1 (exists), 0 (not exists), and negative/error values per key; any non-1 (including errors and ambiguous replies) triggers a put. Not a memory bug per se (put overwrite of same-content key is benign IF key=>content is a pure function of the prefix hash) — but with group semantics across replicas a duplicate put of the same group while another replica's GET is in flight is the known "reload of invalidated peer / in-flight read vs delete/overwrite" class. NEEDS: confirm mooncake put of an existing key is upsert-with-atomic-swap or delete+create (latter breaks in-flight RDMA reads of same key at another replica). Also batch_set() (raw) calls _put_batch_zero_copy_impl unconditionally even when set_keys is empty (no `if len(set_keys) > 0` guard, unlike batch_set_v1) — empty-list put into mooncake; behavior depends on client (possible error/exception on empty batch).

## P2-1. batch_set() (raw Storage API) returns "consecutive-prefix success" but put failures are not rolled back and result mapping can mark not-put keys as existing
mooncake_store.py batch_set: exist_result reused as status array; `for i in range(len(set_indices)): if put_result[i]==0: exist_result[set_indices[i]]=1`. If put of key i fails, exist_result[i] stays !=1 and success_count stops there — but keys AFTER i that were put successfully are not reported. Callers that treat False as "nothing was stored" then retry puts. Benign for content-keyed data; flag for awareness.

## P3-1. _batch_exist pp_rank rewrite is string-surgery on keys
`f"_{pp_rank}_".join(key.rsplit(f"_{self.pp_rank}_", 1))` — only correct if the LAST occurrence of `_{pp_rank}_` is the rank field. With pp disabled suffix is `{local_rank}` for MHA; for MLA pp>1 suffix `{pp_rank}`. A key whose hex/token-hash tail ends with `_{pp_rank}` digits before suffix still works (rsplit count=1 hits the true suffix because suffix is later). BUT if two rank fields can have equal numeric value (tp_rank==pp_rank for MHA-pp), the wrong occurrence may be replaced. Prefetch extra_info pp_rank comes from the request. Low likelihood in prod (MLA everywhere disagg?); note as fragile.

## Areas verified CLEAN (with assumptions)
- direct linker load path (mooncake_direct_linker.load_layer_wise) DOES validate exact byte counts: `expected=[sum(item) for item in sizes]; list(result) != expected -> raise`. And session start checks all-zero. Short reads on THIS path are caught (any short read raises, fail counter). Good.
- direct linker registers each distinct GPU allocation once (`seen` set keyed on (ptr,nbytes)) => no double-register; but close() calls storage.close() which is a no-op and MooncakeDistributedStore dtor unregisters — host pool tensors freed by python GC after store dtor? If pool freed first, mooncake dtor may unregister an already-freed region (only at teardown; assumption: store  destructor handles missing gracefully). Verify at version-check step.
- l2_transfer.py (device<->host DMA engine): index tensors kept alive via record_stream + resolved list; layout domain acquire/release with finish event; host slot relocation gate documented. No free-before-completion inside this file; slot-lifetime guarantee depends on hicache controller honoring the move gate (out of my scope — hicache host-pool owner should check).
- MooncakeHostTensorAllocator: buffer allocated by native allocator; tensor wraps raw ptr via frombuffer; `self.allocator` keeps ownership, `self.ptr` recorded. No free in this class — lifetime bound to pool object. OK assuming pool outlives store usage (true: pools live for process lifetime). NOTE: frombuffer tensor does NOT own memory; if allocator.alloc'd buffer is freed by native lib teardown while torch tensors still reference it -> UAF at teardown only.

## Still to check
- [ ] production-version exposure (fetch v0.5.14/0.5.15.post1/K3 branch 87dd4596, v0.5.17, v0.5.20) for P1-1 and P1-2.
- [ ] mooncake client short-read semantics from upstream mooncake repo (does batch_get_into ever return partial byte counts? sessions?)
- [ ] prefetch timeout/cancel: is a pending batch_get_into ever cancelled while RDMA targets host page? (prefetch policy timeout lives in hicache controller — interface-wise, mooncake store get calls are synchronous; no async handle => no cancellation while RDMA in flight at THIS layer. Confirm there's no thread doing get per request.)
- [ ] backend_factory.py rest + storage/__init__.py (API contract).
- [ ] group semantics: set-when-incomplete / concurrent replicas put-same-group overlap window.
- [ ] test_mooncake_store.py coverage gaps.

## CONFIRMED via CPU repro (mock mooncake client, REAL upstream functions extracted by AST and exec'd)
4/4 confirmation cases fail-safe (i.e., bug present):
- _batch_postprocess([2.5MB partial of 5MB]) -> [True]   (prefetch page counted as hit)
- get() with 0 bytes read -> True
- batch_get([k], sizes=[5MB]) with 1.25MB partial -> reports 1 page fetched
- embedding batch_get & batch_get_into_multi_buffers with 100 of 4096 bytes -> [True]
Mooncake's own convention proves the API returns byte counts that can differ from the request:
mooncake-transfer-engine wheel 0.3.13.post1, mooncake/structured_object_store.py
`_read_chunks_with_batch_get_into`: `if actual_size != expected_size: raise RuntimeError(...)`
i.e. the only correct success test is equality with the requested size; SGLang's `> 0`
accepts any partial transfer. Impact with shared store + prefetch: short RDMA read
(e.g. torn object after eviction mid-read, SSD-offload truncation, replica mismatch)
-> host page holds stale bytes of an evicted page -> served as valid KV.
Cross-request leakage / garbled output in the exact production config.

## VERSION EXPOSURE (short-read checks present; line numbers shift but identical code)
- v0.5.14 (49e384ce): mooncake_store.py `res > 0` (l.1059 area), `get_result[0] >= 0`, `get_result[i] < 0`; embedding store `res > 0`. AFFECTED.
- v0.5.15.post1 (0b3bb0cb): same. AFFECTED.
- v0.5.17 (2948168546): same. AFFECTED.
- v0.5.20 (94602c9c): same. AFFECTED.
- Kimi K3 branch (87dd4596/bd4828e2): commits not public (couldn't verify). Same file lineage spans 0.5.15.post1..0.5.20 unchanged in these functions => treat as AFFECTED unless internal patch.
- upstream HEAD (this repo): AFFECTED (lines 1177/1199, 1376, 1406; embedding 57/94).
This is the production's locally-patched class; the local patch is NOT upstream.

## Sibling of P1-1 found while tracing: warmup() uses store.get(warmup_key) == bytes (strict equality) -> SAFE. Direct-linker load path validates exact sizes -> SAFE.

## Fix contract (for whoever lands the fix; I do NOT submit to merge queue per instructions)
For every batch_get_into/batch_get_into_multi_buffers result r with requested size s:
  success iff r == s (r < 0 or 0 <= r < s -> failure; r > s never happens).
Apply in: _batch_postprocess (needs buffer_sizes passed in), get(), batch_get(), batch_get_v1,
MooncakeEmbeddingStore.batch_get / batch_get_into_multi_buffers. Note: for MHA split-heads
key_multiplier>1, sizes per component differ -> check each element against its own size.

## ROUND 2 RESULTS (lifecycle, group semantics, embedding controller, tests)

### P2-2 (confirmed by repro): raw batch_set() calls mooncake batch_put_from with EMPTY lists when all keys exist
mooncake_store.py batch_set: exists loop removes nothing; set_keys=[] but
`_put_batch_zero_copy_impl(set_keys, ...)` is called unconditionally (batch_set_v1 has the
`if len(set_keys) > 0` guard; batch_set does not). Repro: all-exist batch_set returns True
but issues `batch_put_from([], [], [])` to the native client. Behavior with the native lib
is unverified (may raise/no-op); either way it's wasted RPC + a crash vector at teardown/failover.
Who calls raw batch_set()? None inside sglang hicache (v1/v2 paths used) — dead-ish in prod, low risk.

### P2-3: embedding prefetch marks partially-FILLING entry READY (short-read sibling, impact path)
embedding_cache_controller._finish_get: success -> _mark_ready(FILLING) -> entry served.
With MooncakeEmbeddingStore's `res > 0` short-read acceptance, an embedding entry whose last
bytes are stale pool memory becomes READY and is scattered into input_embeds (mm_utils).
Buffer ownership during the async GET is SAFE: FILLING entries are not in the evictable LRU
so their pages cannot be reallocated while the RDMA is in flight; failure frees after completion
(sequential io_loop). PUT path pins (_pin_read) until _finish_put. No UAF in the controller.

### Group semantics / shared-store cross-replica (enable_group_semantics=true)
- Key isolation: config_prefix = extra_backend_tag + model_name -> `_tag_keys` prefix on ALL
  keys incl. component keys & warmup? (warmup key NOT tagged but randomized via uuid -> fine).
  Two deployments sharing a master but different backend tag cannot collide. Two replicas same
  tag intentionally share. Assumed: ops never strip the prefix (verified in _tag_keys call sites:
  batch_exists(_v1) tags; _batch_io_v2 tags both page keys and component keys; batch_exists_v2
  tags via component helper; set()/get()/batch_set()/batch_get() (raw API) do NOT tag keys —
  but the raw API is not used by hicache controllers (they use v1/v2) — CHECK: any external
  caller of raw set/get would bypass the tenant tag. grep shows nothing in sglang calls raw
  set/get/batch_set/batch_get on MooncakeStore except tests. Safe by disuse; flag for hardening.
- Duplicate-put race (two replicas pass exists before either puts): mooncake master rejects
  duplicates — store.so strings: "Key already exists in storage backend, skipping",
  "Reserved key became duplicated before commit". Loser gets non-zero result -> SGLang marks
  that component failed -> benign (self-heals on next backup). Content identical anyway
  (keys are prefix-hash derived).
- put vs concurrent GET of same key at another replica (the NIXL "reload of invalidated peer"
  analog): mooncake objects are immutable once committed; refs exist during get sessions
  (batch_get_session_start/end used by direct linker). For plain batch_get_into between
  commit and eviction: removal during in-flight RDMA read is handled by transfer-engine
  refcounting per mooncake docs (ASSUMPTION — not verified in native code). The SGLang-side
  ambiguity is exactly what the short-read check should catch; with P1-1 fixed, a torn read
  surfaces as failure instead of silent garbage. P1-1 fix is therefore the load-bearing one.
- clear() -> store.remove_all(): called from hybrid_cache_controller.clear(). On a SHARED
  store this deletes other replicas' objects as well (namespace-wide remove_all, prefix-less).
  If deploy tooling ever calls clear() on a shared deployment this is cross-replica data
  destruction. Flagged as operational hazard; verify call sites of controller.clear().

### OOB / index math review (batching priority 4)
- _batch_io_v2 expansion, _build_pool_buffer_meta missing-page filtering,
  _pack_multi_buffer_meta: index arithmetic consistent; sizes carried alongside;
  assert len(key_list)==len(ptr_list) at every builder; no OOB found.
- MLA page_first_direct multi-buffer get: partial layers? get_page_buffer_meta returns
  num_layers ptrs per key consistently; mooncake multi-buffer API requires len(ptrs[i]) ==
  len(sizes[i]) per key — zipped, safe.
- Embedding _build_storage_transfer_buffers: ptr/size per page run — not reviewed deeply
  (owner: whoever handles embedding store in prod; low traffic).

### test_mooncake_store.py coverage verdict
Integration-only (needs live master). NO mocked-client unit tests; NO failure-path tests
(short read, put failure, exists-error, empty batch); NO group-semantics tests; NO
concurrent-replica tests. The P1-1 class would be caught by a trivial mocked short-read test
=> regression test should be exactly that (exists in my repro; can be ported as pytest with
monkeypatched mooncake module, no native lib needed).

## FINAL COVERAGE STATEMENT (files read end to end)
mooncake_store.py (1516), mooncake_direct_linker.py (440), mooncake_embedding_store.py (127),
test_mooncake_store.py (196), backend_factory.py (276), l2_transfer.py (197),
storage/__init__.py, embedding_cache_controller.py lifecycle parts (440-740),
hybrid_cache_controller.py call sites (1440-1660).
Confirmed bugs: P1-1 (short reads, 4 call sites, all prod versions, repro), P2-2 (empty put call).
Flagged risks: clear()->remove_all on shared store; raw-API keys bypass tenant tag (unused);
mooncake-native put-vs-inflight-read refcounting (assumption).
Clean-with-assumptions: direct-linker load path (exact-size validated), embedding controller
buffer lifetime (FILLING not evictable; PUT pinned), l2_transfer (record_stream + layout lease;
host-slot move gate owned by hicache controller — outside my files).

## ROUND 3 (2026-10-16, post lead directives) — siblings, contract evidence, caller impact
Lead owns the fix for sites (a) batch_get, (b) _batch_postprocess, (c) get(). I hunt siblings/support.

### Mooncake native contract — short reads are REAL production events (hunt item 3, resolved)
Evidence from mooncake-transfer-engine 0.3.13.post1 wheel, mooncake/store.so strings:
- "batch_read short read for key:" and "DFS short read for key" / "DFS short write for key"
  are distinct log sites (batch RDMA read path + SSD/DFS offload path) — a short read is a
  known, names logged event in the client, returned per-key as byte counts (pybind docstring)
  rather than a negative error. Under prod load (SSD offload on, eviction pressure,
  8x mlx5 RDMA), these fire and SGLang's `res > 0` accepts them.
- First-party convention: structured_object_store._read_chunks_with_batch_get_into requires
  actual == expected, else RuntimeError. SGLang is the outlier.
- In-flight read vs eviction: native strings "OBJECT_HAS_LEASE", "lease_id=",
  "Clear replicas for the given keys. Requires lease to be expired." => reads take leases and
  eviction waits for lease expiry. The read-vs-delete race is lease-guarded NATIVELY; the
  residual failure surface is a lease that expires mid-read (client-side timeout config) —
  which surfaces exactly as a short read => again caught only by an r==size check.
Upgrades my earlier assumption: put-vs-inflight-read is MOSTLY guarded natively; the
lease-expiry edge funnels into the short-read check. P1-1 remains the load-bearing fix.

### Sibling sweep of the lenient-success class (hunt items 1,4)
- SET path: mooncake put results are 0 / negative only (no byte-count success channel) per
  pybind docstring quoted in _batch_postprocess. `all(res == 0)` is correct. CLEAN.
  Except: P2-2 empty-list batch_put_from call in raw batch_set() (round 2) — lead aware? (it's
  outside the three get sites; including in fix scope suggestion).
- batch_set_v1 / _batch_io_v2(is_set=True): exist-result negative/error -> treated as
  "missing" -> put attempt -> duplicate reject -> reports failure. Conservative, CLEAN.
- exists()/batch_exists()/batch_exists_v2: any exist error/negative truncates the hit
  boundary -> conservative miss -> recompute fallback. CLEAN.
- Embedding store batch_put / batch_put_from_multi_buffers: `res == 0` required. CLEAN.
- set() single-key: exist error -> proceeds to overwrite-put attempt -> duplicate-reject ->
  returns False. Conservative. CLEAN.
- v2 group ids ARE built from TAGGED page keys in both batch_set_v1 and _batch_io_v2
  (`_expand_group_ids(tagged_keys, ...)`) => group ids carry backend-tag+model namespace;
  no cross-tenant group collision (checked after initially suspecting untagged ids).
  Same page across pools shares one group id in one batch -> atomic group lifecycle intended.
- Duplicate group put racing replicas: native rejects ("Key already exists in storage backend,
  skipping:"), loser reports failure, content identical (prefix-hash keyed). CLEAN.
- _batch_exist pp_rank string rewrite: re-verified all suffix layouts (MLA pp, MHA pp,
  split-heads) — rsplit(count=1) always lands on the true rank field. Downgraded: NOT a bug.
- get() `>= 0` zero-byte read: inside lead's site (c). The four get-side call sites I found are
  lead's (a),(b),(c) + embedding store `res > 0` x2 (mooncake_embedding_store.py:57,94) —
  CONFIRM embedding sites are in lead's fix scope? They serve production mm paths (Gemma/Kimi)
  and are the same class but in a DIFFERENT file than the three listed sites. Flagging to lead.

### Caller impact of False mid-batch AFTER the fix (hunt item 2) — fix propagates cleanly
cache_controller._page_get_zero_copy counts a strict SUCCESS PREFIX of results[], breaks at
first False; _page_transfer only reports completed_pages up to the first failure and stops
further storage IO (still emits PrefetchAcks for TP sync). Pages after the failure are never
read; pages before it are fully read. Host-index tail release for the un-fullfilled span is
handled by the deferred-release path (verified by prefetch-ctl-audit: guarded in main).
_page_transfer_kv_batch clamps kv_hits vs sidecar_hits with min() -> conservative. No double
use, no stale-page serving, no cleanup gap introduced by turning short reads into failures.
The only behavior change: fewer phantom hits, occasional extra recompute — desired.

### Regression-test pinning status (directive 1)
- test_mooncake_group_semantics.py: pins group_ids construction (tagged logical page key),
  filtering alignment with missing keys, MHA/MLA/split-heads/indexer variants, fallback when
  old mooncake lacks group_ids. It does NOT pin any short-read or duplicate-put behavior
  (fake store returns ideal 0/1 exists, all-0 puts, full-size gets — fake_get_into returns
  [len]? checked: fake returns success vectors, never partial).
- test_mooncake_store_config.py / test_mooncake_tenant_config.py / test_mooncake_ssd_offload_path.py:
  config-loading pins only (checked file list; no IO-semantics tests).
- unit/disaggregation/test_mooncake_custom_mem_pool_batch.py: batching/address math pin.
- storage/mooncake_store/test_mooncake_store.py: live-master integration, happy path only.
==> NO shipped test pins the short-read guard. Lead's new regression test will be the first.

## ROUND 4 — hostpool-audit suspect resolution: "L3 RDMA raw physical ptrs w/o layout lease"
Layering check (traced from storage call sites up):
1. Production HiCache+Mooncake pairing = UnifiedRadixCache -> HybridCacheController
   (typed at unified_radix_cache.py:285). hybrid_cache_controller._page_transfer (l.1481)
   and _page_backup (l.1567) wrap the ENTIRE mooncake IO (batch_get_v1/v2, batch_set_v1/v2 —
   all SYNCHRONOUS, RDMA completes before return) in mem_pool_host.layout_lease().
2. UnifiedPageEnvelopeHostPool compaction (_begin_compaction) blocks while any
   acquire_layout() user is active and synchronizes _pending_transfer_events before moving
   pages; l2_transfer._submission passes its finish event into release_layout so async
   device<->host DMA is also fenced. So: pages cannot move mid-RDMA and not mid-DMA.
3. UnifiedPageEnvelopeHostPool is referenced ONLY from hybrid_cache_controller,
   hybrid_pool_assembler, pool_configurator. The BASE HiCacheController (cache_controller.py,
   no layout_lease anywhere) is paired with plain MHA/MLA buffer host pools that do NOT
   compact/move pages. ==> The pairing that would make the suspect live (base controller +
   unified compacting pool) does not exist in the tree.
4. MooncakeDirectLinker load path writes GPU KV (not host pool) via synchronous
   batch_get_into_multi_buffer_ranges; GPU slots pinned by the live request's lifecycle —
   different domain, no unified-host lease required.
5. Residual edge (kept for completeness): if anyone EVER pairs base HiCacheController with
   UnifiedPageEnvelopeHostPool, mooncake L3 gets run unleashed and _compact_side can move a
   page while RDMA targets its old physical range. Not reachable today; an assert in the base
   controller ("host pool must not require layout") would harden it.
Verdict for hostpool-audit's suspect: NOT EXPLOITABLE in the shipped pairings; mooncake side
of the question is fully answered (synchronous IO inside layout lease + event-fenced DMA +
compaction blocking on lease users).

## ROUND 5 — P2-2 severity set to LOW + fix landed on agent/mooncake-audit
- File re-labeled LOW (unused in prod paths; native-client crash vector).
- Fix: batch_set() skips _put_batch_zero_copy_impl when set_keys empty (mirrors
  batch_set_v1's guard). Empty-put RPC eliminated.
- Regression: test/registered/unit/mem_cache/test_mooncake_batch_set_empty.py
  (2 cases: all-exist -> zero put RPCs and True result; partial-miss -> put RPC
  contains exactly the missing keys). Fake-client style mirrors the lead's
  test_mooncake_short_read.py so both run in base-a-test-cpu.

## FINAL STATUS (2026-10-16 ~04:08 UTC)
- P1-1 short-read family: FIXED by hicache-lead (branch agent/hicache-lead e766366a + test_mooncake_short_read.py, 6 tests). Independently verified by me (board item #33 APPROVED: ran tests 6/6, over-strictness review, set-path check, pre/post AST behavior check). In merge batch MR #15 (running).
- P2-2 empty batch_put_from: FIXED by me on agent/mooncake-audit (commit 82660af + test_mooncake_batch_set_empty.py, 2/2 green); handed to lead for the merge batch.
- Embedding-store short-read sites (mooncake_embedding_store.py:57,94): same class, assigned to spec-auditor (per hicache-lead #353 routing), NOT in MR #15.
- hostpool-audit's unified-pool/L3-RDMA-vs-compaction suspect: resolved non-exploitable (round 4); acknowledged.
- npu_memcache/simm own-_batch_postprocess short-read siblings: flagged to leads, out of prod scope (charter).
- Regression-test pinning: before this swarm, NO shipped test pinned any failure-branch of the mooncake store (all fakes returned ideal vectors). Now pinned: short-read (6 tests, lead), empty-put (2 tests, mine).
This concludes my assigned scope for board item #6 (mooncake store backend).

## ROUND 6 — MooncakeDirectLinker failure-semantics note (fed into C4 decisive-line discussion)
Within my scope, the direct-linker path (production mooncake + unified cache, mamba-capable)
loads store→GPU per TP rank; completion gating is LayerWiseLoadCounter
(unified_cache_linker.py:66-133). Documented behavior at :66-73: a failed layer does NOT stop
the forward that waits on it — forward runs on whatever the GPU slots hold; failure is only
reported at finish()/finish_loads for a post-hoc scheduler abort of the loading request.
Consequences: (a) TP-asymmetric load failure => rank-divergent forward on stale GPU slots
until abort lands; (b) safety depends entirely on "abort before their output is used"
(call sites cache_controller.py:360, UnifiedCacheLinkerWrapper.finish_loads :532) which is
outside my files — handed to hicache-lead/prefetch-ctl for verification, NOT claimed as a bug.
Good news from the short-read fix: its exact-size errors raise into layer_done_counter.fail
(linker path already validated exact sizes), so torn loads become detectable failures there —
the residual question is purely the abort-before-output airtightness.

## C6 LINKER-SIDE HALF (assigned to me by hicache-lead; scheduler half: decode-offload-audit)
Question: can output escape a request whose direct-linker GPU load failed (stale slots)?
State machine (HEAD origin/dev):
1. load() queues per-rid in pending_loads (per TP rank thread).
2. start_layer_wise_loading: plan-digest all-gather across TP (linker :262-276) — divergent
   PLANS fall back to all-rank reads; counter_index = producer index; MLA dedup: non-src peers
   short-circuit complete() (:297-300) and rely on broadcast; ready_event gates stream order.
3. Load thread (load_layer_wise): session_start → per-layer batch_get_into_multi_buffer_ranges
   with EXACT-size validation (raises on short read — corroborated again post short-read fix)
   → per-layer layer_done_counter.complete(); ANY exception → fail(counter_index): sets
   exception on ALL not-yet-done layer futures for the batch.
4. Forward-side waits: pool layer-wait hook → LayerWiseLoadCounter.wait_until (unified_cache_linker
   :103-112): future.exception() returns (does NOT raise) on failure → forward PROCEEDS on
   whatever the slots hold (private slots — see 6). On failure it STILL invokes
   on_layer_ready → _broadcast_loaded_layers broadcasts layers even when the src's load failed
   (comment in wait_until is explicit): peers thus consume broadcast STALE data for failed
   layers as well — TP-DIVERGENT garbage window, rank-local for plain path; broadcast-wide for
   MLA dedup path. Neither raises into forward.
5. Post-forward: scheduler calls linker path finish → UnifiedCacheLinkerWrapper.finish_loads
   (unified_cache_linker.py:532): waits ALL layers of the batch on this rank, MIN-reduces the
   landed verdict across attn groups (TP-consistent verdict), returns failed rids.
6. unified_radix_cache.finish_external_linker_loads (:3490): failed rid →
   discard_output_reason=FINISH_ABORT + skip_radix_cache_insert=True. Output marked for drop;
   KV never enters tree. Slots are request-private per load_back_is_cache_owned=False
   (:3484) "so nothing else read them".
7. Airtightness boundary passed to decode-offload-audit: prove finish_external_linker_loads
   runs before ANY output for those rids is consumed (streaming/logprobs/cache-report).

Findings within the linker half itself:
- The stale-forward window IS real and by design (docstring), on BOTH plain and MLA-dedup
  paths; its safety rests entirely on step 6 + scheduler ordering. NOT a bug in my files.
- complete()-after-fail() InvalidStateError: fail() sets exception on all pending futures;
  subsequent complete() for a later layer raises InvalidStateError inside the load loop →
  swallowed by the blanket except → noisy masking of the true stack; behavior still fail-safe
  (counter already failed). Cosmetic/observability, recommend guarding complete() with done()
  check. (Micro, worth folding into whatever fix C6 lands.)
- broadcast_loads[index] cleanup only happens when on_layer_ready reaches the LAST layer
  (mooncake_direct_linker._broadcast_loaded_layers deletion at threshold==num_layers-1);
  if the batch is aborted before forward waits on tail layers (possible under abort storms),
  the entry + prepared broadcast tensors linger until reset(). Memory-lean, not a leak of
  correctness; flag for lifecycle review with C6 fix.
- finish_loads pops private_load_indices for ALL rids even on success; failed rids' slots
  freed via abort flow (request-owned). No double-free seen in linker itself.

## C6 LINKER ROOT — FINAL (board #64, my half complete)
Reset hygiene: linker.reset() is reached only via scheduler-driven tree_cache.reset()
(scheduler.py:5411) which is TP-lockstep; load/offload queues are joined, pending/counter/
broadcast state cleared; producer/consumer index rewind is coherent per-rank at a
TP-consistent boundary. CLEAN w/ assumption (scheduler never resets mid-forward — true in
SGLang design; no in-linker cross-rank fence needed).
Producer/consumer race on LayerWiseLoadCounter.futures: plain-dict ops under GIL; complete()
after finish()'s delete cannot happen because finish() first waits all futures then deletes
(:122-123 comment verified in behavior); fail-after-finish also ordered by the same wait.
InvalidStateError masking stands as observability-only (complete-after-fail within the same
batch: fail() sets exception on remaining futures; load loop's next complete() raises; blanket
except masks original stack — recommend done() guard, micro).
Stale-broadcast-across-peers (MLA dedup, src failure): CONFIRMED BY CONSTRUCTION — wait_until
invokes on_layer_ready even on failure (:110-112 with comment); peers consume broadcast stale
layers; contained by post-forward MIN-verdict abort. Window is real; containment verified up
to the scheduler boundary (decode-offload's half determines final verdict).
My half of the state-machine proof is complete; remaining gap is the scheduler's output-commit
cadence (owned by decode-offload-audit).

## C6 JOINT VERDICT (with decode-offload-audit)
decode-offload scheduler half: TP/dp-attn AIRTIGHT (discard_output_reason gates
batch_result_processor.py:316/:473; MIN-verdict attn-consistent). PP>1: CONFIRMED BUG —
all_reduce_attn_groups folds only attn groups; middle-PP-stage load failure never marks the
request on the last stage => last stage streams garbage sampled from propagated hidden states
(>=1 token can escape before cross-PP reconciliation). Fix (decode-offload): pp_group in the
MIN-reduce. My linker-half answers their cross-check: batch_get/session exact-size errors
raise in the load thread -> fail(counter_index) for every REAL producer index (the only
indices ever enqueued); index=-1 never associates with started loads (return -1 happens only
when pending_loads is empty, pre-update_producer); double finish impossible via wrapper's
inflight_load_index swap. So get-side failures RELIABLY mark the counter; the PP escape is
purely the verdict-reduce width (their fix), not a missed fail signal. C6 = CONFIRMED BUG
(PP>1 path), fix owned by decode-offload-audit; design-change ACK needed from leads per
protocol (pp_group widens a collective).

## C6 PRODUCTION EXPOSURE NOTE (hicache-lead, #1188)
External cache linker is opt-in and MUTUALLY EXCLUSIVE with HiCache => outside today's
production surface (all prod deployments run HiCache+Mooncake store, not the direct linker).
C6 fix still warranted (garbage-token class, small diff) but no prod-version patch is needed;
if a deployment ever enables the linker with PP>1, it must carry the fix. Regression test
shape prescribed by lead: pure-logic per-stage verdict reduce with stubbed distributed —
my AST-extraction technique (verification section) is directly applicable for CPU-only runs.

## C6 HYGIENE MICRO REFINEMENTS (after decode-offload-audit's trace)
- complete()-after-fail: DOWNGRADED to latent-only. In the current call graph, fail() aborts
  the same load loop iteration (exception from the layer loop bodies -> except -> fail ->
  loop exited), so complete() never runs after fail() within a batch; the finally-fail path
  is done()-greedy. Keep the done() guard as cheap hardening against future call-graph changes;
  NOT a live bug today. (My earlier "masks the true stack" phrasing overstates: the blanket
  except covers get-side exceptions whose stack IS the true stack.)
- broadcast_loads[index] device-tensor linger: CONFIRMED independently by decode-offload
  (abort-before-last-layer threshold => entry+popped only at reset; device-side, bounded).
  Recommendation stands: pop broadcast_loads[index] in any failure/abort path of
  finish_layer_wise_loading/reset-independent cleanup.
Incidentally, their airtightness line adds my missing piece: linker loads are admission-only
(schedule_policy.py:1456) => the failing forward is the req's FIRST forward => no earlier
stream exists; discard gate precedes stream_output of committed ids (:316/:473/:502/:1089).
C6 = CONFIRMED BUG PP>1 only; fix decode-offload's (pp_group in verdict MIN).
