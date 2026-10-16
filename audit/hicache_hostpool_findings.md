# HiCache host pool + free-list audit (hostpool-audit, board item #6 sub-area)

Scope: python/sglang/srt/mem_cache/memory_pool_host.py, host_memory.py,
pool_host/{base,common,group,mha,mla,page_unified,unified}.py; skim
{dsa,hisparse,qsa,mamba,mha_mxfp8}.py. Ignore npu_memfabric.py.
Production: page_first_direct, io_backend=direct, page_size 64, fp8_e4m3,
write_through, decode offload-kvcache. Versions to check: v0.5.14, v0.5.15.post1,
K3 branch (~v0.5.16), v0.5.17, v0.5.20, HEAD.

Note: this repo has squashed history (1 commit). Upstream tags fetched from
github.com/sgl-project/sglang for archaeology.

## Status legend
- CONFIRMED = repro or precise two-path trace
- SUSPECT = strong evidence, needs repro
- CLEAN = checked, assumptions listed

## Upstream regression-test pinning (per lead directive) — test_mem_pool_host.py
- Base-pool double-alloc/double-free: pinned by
  `test_double_alloc`, `test_double_free`, `test_free_unallocated`,
  `test_free_after_clear` (same file) — BUT only against
  `MHATokenToKVPoolHost` (which uses the base, guarded, alloc/free).
- `TestLazyHostPoolRelease` (`test_mamba_pool_lazy_release`,
  `test_deepseek_v4_pool_lazy_release`, `test_logical_pool_lazy_release`)
  covers exactly the three UNGUARDED overrides and pins the CURRENT unguarded
  behavior — it never frees the same index twice. So F1's sibling gap is
  exactly pinned-as-absent here: mamba/dsv4/logical double-free = 0 tests.
- Lazy release merge per se: also test_paged_allocator_lazy_release.py
  (device paged allocator sibling, out of my files).
- Reclaim order / unified transfer-compaction: test_unified_hicache_regressions.py
  `TestTransferStreamOrdering`, `TestSwaBackupAfterCompaction`,
  `TestDirectBackendTranslation`, `TestSwaBackupAfterCompaction` pin the L2
  lease + event ordering (l2_transfer.py layer). F5 (L3 raw-ptr mooncake IO
  w/o lease) has NO test — siblings hunted:
  mooncake_store.py:1075/1139/1152 all use the same unleased meta path.
- F2 sentinel: no test frees or page-indexes a negative index anywhere.
  hicache_storage.py `_read_page`/`_write_page`
  (set_from_flat_data_page/get_data_page) would also alias the last page on
  a -1 page_offset — sibling of F2 via the storage-key path, not only free().

### K3 branch check
git fetch of bd4828e2 from public upstream fails (branch not public).
version-triage then FETCHED v0.5.16 directly (2026-10-16 ~04:00) and
verified line-by-line: v0.5.16 DeepSeekV4PagedHostPool.free +
LogicalHostPool.free = `release_slots.append` with no guard; MambaPoolHost
unguarded likewise — so the K3-production base (~v0.5.16) is CONFIRMED
affected by F1 (and by F2, which has no bounds check on any tag).
(No longer "assume affected".)

---

## Findings (ranked)

### F1. [CONFIRMED] Double-free is silent corruption in subclass host pools (guards missing), incl. all pools on v0.5.14

**Impact chain (repro below):** one double-free puts the same slot ids twice in
`release_slots`; after `_merge_release_slots`, two `alloc()` calls return the
same page → two live owners write/read one host KV buffer → cross-request KV
corruption / context leakage. Completely silent (no assert, no log).

The known-prod bug class "host pool free-list double free" was an upstream
caller bug (e.g. evict + abort freeing the same host_indices). Response:
- v0.5.14: base `HostKVCache.free` is `free_slots = torch.cat(...)` — NO lock,
  NO guard. Double-free is silent for EVERY pool (MHA/MLA/mamba). v0.5.14 is
  in production for Kimi K2.6/K2.7 and Gemma.
- v0.5.15..HEAD: base `HostKVCache.free` (pool_host/base.py:555) got a
  `threading.RLock` + `slot_used[]` assert detector. Base-pool double-frees
  now crash loud instead of corrupting.
- BUT the following subclass free()/alloc() OVERRODE the base and never got
  the guard on ANY version incl. HEAD:
  - `MambaPoolHost` mamba.py:389 (also no slot_used anywhere; own __init__
    never creates it). Used by Kimi K3 hybrid mamba HiCache.
  - `DeepSeekV4PagedHostPool` memory_pool_host.py:449. DSv4 production.
    (`DeepSeekV4StateHostPool.free` raises NotImplementedError — no free
    list — by design.)
  - `LogicalHostPool` memory_pool_host.py:140. DSv4 anchor pool; its slots
    are FULL-pool anchors: two owners of one anchor = two writers to one
    unified full-KV region.
  - `_SharedPageEnvelopeHostBacking.free` (unified.py:603) HAS its own
    detector. OK.

**Repro (verbatim class source exec'd; HEAD):**
```
PAGE=64; pool=LogicalHostPool(size=128,page_size=64)
a1=pool.alloc(64); a2=pool.alloc(64)
pool.free(a1.clone()); pool.free(a1.clone())   # double free
b1=pool.alloc(64)                               # merge pulls a1's slots twice
b2=pool.alloc(64)                               # b2 == b1's slots
# b1 ∩ b2 == 64 slots  -> SILENT ALIASING CONFIRMED
```
Same repro textually applies to MambaPoolHost.free (mamba.py:389) and
DeepSeekV4PagedHostPool.free (mph:449) — identical `release_slots.append`
bodies with no guard.
A parallel exec of base.py HostKVCache.free on HEAD raises
`AssertionError: Double-free detected` — i.e., upstream considered the class
real enough to add detection on the base but missed the overrides.

**Versions affected:** v0.5.14 (all pools, silent); v0.5.15.post1, ~v0.5.16/K3,
v0.5.17, v0.5.20, HEAD (mamba/DSv4-paged/logical pools silent; MHA/MLA/unified
pools crash-loud). Kimi K3 branch uses MambaPoolHost → exposed.
(version-triage independently verified all of the above by fetching tags incl.
v0.5.16: base guard only exists .15p1+; all three override frees unguarded on
EVERY tag. CHERRY-PICK NUANCE: at v0.5.14/.15p1 MambaPoolHost and
DeepSeekV4PagedHostPool live in mem_cache/memory_pool_host.py; at v0.5.20+
mamba moved to pool_host/mamba.py while DSv4/Logical stayed in
memory_pool_host.py — prod backports of the fix need both file paths.)

**Suggested fix:** move the slot_used bookkeeping into a shared helper
(HostKVCache mixin) used by every override: set bit in alloc, assert+clear in
free; logical_size-sized bool tensor per pool (~1 byte/slot). Or add the same
asserts to the three overrides + LogicalHostPool.

(legacy SUSPECT section kept for reference below)

### F1+F2 FIX STATUS (board #25, branch agent/hostpool-audit, commits 1520122 + 1ad4f3d)
Fix applied on this branch:
- `pool_host/base.py`: new module helper `assert_valid_host_slot_indices`
  (asserts indices >= 0 and < logical_size, named-caller message);
  `HostKVCache.free` calls it before the slot_used check.
- `MambaPoolHost` (mamba.py), `DeepSeekV4PagedHostPool` and `LogicalHostPool`
  (memory_pool_host.py): gained slot_used init in clear() + double-alloc
  assert in alloc() + sentinel/double-free asserts in free(), mirroring the
  base class. Import added for the helper.
- Regression tests in test_mem_pool_host.py: base-pool sentinel/out-of-range
  frees (stub-based so they run in small sandboxes), and
  double-free+sentinel guard tests for mamba/dsv4/logical pools via the
  existing __new__ stub pattern. Also re-verified the pinned lazy-release
  behavior is preserved exactly (DSv4/mamba/logical lazy_release tests pass).
- Local pytest: failure set IDENTICAL to origin/main on the same sandbox
  (10 pre-existing env failures: host-memory budget, shm, mamba empty-pp);
  my branch goes 14->19 passed (+5 new guard tests green).
Re-exec of the verbatim-source repros (audit/repros) against the patched
code now raises AssertionError ('Double-free detected', 'Host-pool free:
negative (sentinel)...') for all three pool families + base.
NOTE for integrator: `slot_used` bool tensor is sized `size` (or logical_size)
— 1 byte/slot; for a 2M-token host pool ~2 MB CPU, negligible.
Behavior change surface: free() now raises on sentinel/out-of-range input —
callers that (incorrectly) passed them will crash loud instead of corrupting
(which is the desired outcome per the audit), but any TEST that exercised
that path will now fail; none found in the file's tests.

### F1-notes. Guard inventory used above
Base HostKVCache.free (pool_host/base.py:555-566) has a `slot_used[]`
double-free/double-alloc detector (added upstream between v0.5.14 (absent)
and v0.5.15 (present); present in v0.5.20 and HEAD).

Subclasses whose free() OVERRIDE the base and LACK the guard on HEAD:
- pool_host/mamba.py MambaPoolHost.free (mamba.py:389) — Kimi K3 production
  (mamba pools on host). Also its alloc (mamba.py:374) lacks the double-alloc
  assert and clear() (mamba.py:362) never creates slot_used, and __init__
  calls HostKVCache.__init__? [VERIFY: mamba __init__]
- memory_pool_host.py DeepSeekV4PagedHostPool.free (mph:449) — DSv4
  production. Its __init__ does NOT call HostKVCache.__init__ (own init,
  mph:215-330), so slot_used never exists. DSv4StateHostPool.free
  (mph:954) [VERIFY same].
- memory_pool_host.py LogicalHostPool.free (mph:140) — V4 anchor pool.
  Logical-only (no KV buffer); a double-free there hands the same logical
  anchor to two writers → cross-request corruption of whatever the anchor
  indexes (full-pool indices in DSv4 unified). Guard absent on HEAD.
- pool_host/unified.py _SharedPageEnvelopeHostBacking.free (unified.py:603)
  HAS its own detection ("Host pages cannot be freed more than once"). OK.

Impact: a double-free at any of these pools is SILENT (no assert) → same
host slot index appears twice in release_slots → two allocs return the same
slot → two live requests write/read the same host KV buffer → cross-request
KV corruption / context leakage. The base-class guard upstream shows the
class is real (a double-free reached production; they patched locally and
upstream added detection).
Versions: guard absent in v0.5.14 (all pools); present for base HostKVCache
in v0.5.15+. Subclass overrides lack the guard on all versions including
HEAD (verify per tag).
TODO: check whether each subclass is reachable in prod configs and whether
any caller can double-free (ownership trace); DSv4StateHostPool check;
check K3 branch mamba free.

### F2. [CONFIRMED] free() accepts -1 sentinel indices: ownership theft + -1 issued as a live slot
`HostKVCache.free` (base.py:555) asserts `slot_used[indices_cpu].all()` but
NEVER checks `indices >= 0`. Negative indices wrap: `slot_used[-1]` is
`slot_used[logical_size-1]`.

Repro (verbatim base.py methods, HEAD):
```
f.size=128; f.alloc(128)            # live owner of 0..127
bad = full(64,-1); bad[0]=5
f.free(bad)                         # NO assert raised!
# slot_used[127] is now False (a LIVE slot marked free),
# release_slots holds 63 copies of -1 + slot 5
f.alloc(64)  ->  returns tensor containing 63x -1 and slot 5
```
Consequences downstream:
- slot 127 silently freed while live -> a later alloc hands it to a second
  owner (double-alloc without any double-free!).
- alloc() issues `-1` as a slot id -> every indexing site
  (`kv_buffer[-1]`, `get_data_page(-1)`, kernel index loads) aliases row
  `logical_size-1` / `page_num-1` -> cross-request KV corruption.
- The -1 sentinels live FOREVER in the free list (never allocatable
  legitimately; when re-issued they corrupt repeatedly).

Vectors: per-request host slot tables in the HiSparse / DSv4 C4 paths are
padded with -1 (`req_to_host_pool`; hisparse.py:59 filters `>= 0` in
`allocated_host_indices`, proving -1 padding exists in these tables). Any
free path that forgets the filter (evict, abort, retract, prefetch-cancel)
hits this. DeepSeekV4StateHostPool "reuses SWA transfer indices" -> its
host_indices flow through DSv4 paged pools' free.
Subclass overrides (mamba/DSv4/logical) do NO checks at all.
Versions: ALL (v0.5.14 -> HEAD), all pools.
Suggested fix: in every free(), assert `(indices >= 0).all()` and
`(indices < logical_size).all()` (or filter -1s explicitly if a caller
contract legitimately passes them). Cheap: one reduction on CPU tensor.

### F3. [CLEAN under assumptions] page_first_direct layout index math at page_size 64
Checked MLA (mla.py) & MHA (mha.py):
- Buffer dims page_first_direct MLA: (page_num, layer_num, page_size, 1,
  kv_cache_dim); MHA: (2, page_num, layer_num, page_size, head_num,
  head_dim). get_page_buffer_meta computes k_ptr = base +
  indices[page_start]*layer_stride... Verified: pages are page_size-aligned
  contiguous (host allocator carves page_size multiples out of arange), so
  page_start_token*L*D == page_id*page_bytes/elem. v_ptr = k_ptr + v_offset
  with v_offset == kv_buffer[0].nbytes (MHA) == correct v slice base.
  get_data_page/set_from_flat_data_page use index//page_size consistently.
- QSA/mha_mxfp8 `_page_ids` use indices[::page_size]//page_size -- same
  alignment assumption.
ASSUMPTIONS (challenge points): (1) every indices tensor handed to these
functions consists of page-contiguous, page-aligned blocks of page_size --
the only guard is `len % page_size == 0`; a token-granular or misaligned
caller would silently read the wrong page (DSv4 C4 token-granular host
indices DO exist but go through transfer_cache_dsv4_mla, not
get_page_buffer_meta). (2) Direct-kernel per-token addressing (sgl_kernel
transfer_kv_*_direct_*) is kernel-side, not auditable here -- page stride
spacing matches layout_dim defs passed in.

### F4. [CLEAN] DeepSeekV4PagedHostPool.alloc rounds UP to slot_page_size
(mph:434-447): alloc returns a rounded-up set; free() accepts arbitrary
lengths. Callers must free the RETURNED tensor, not the requested count.
ASSUMPTION: all callers free transfer.host_indices / allocated tensors as
returned (release_transfers in group.py does exactly this). A caller freeing
only need_size elements would leak the rounded tail permanently (capacity
leak, not corruption). Not found in-scope.

### F5. [CLEAN — resolved by two independent traces] Unified page-envelope pool:
mooncake store L3 IO uses raw physical ptrs WITHOUT layout lease.
UnifiedPageEnvelopeHostPool.get_page_buffer_meta (unified.py:1065) resolves
physical addrs under lock but returns raw ptrs; the mooncake store
(mooncake_store.py:1075,1139,1152) then issues RDMA set/get on those ptrs
with NO acquire_layout(). If _compact() runs while that IO is in flight,
physical pages move -> RDMA reads/writes the wrong host page (cross-request
leak / corruption). RESOLVED CLEAN by independent traces from prefetch-ctl-audit and
mooncake-audit (2026-10-16): (1) mooncake batch_get_into/batch_put_from are
SYNCHRONOUS per the per-key byte-count return contract — no async RDMA
outlives the call; (2) unified pools pair only with HybridCacheController
(unified_radix_cache.py:285), whose _page_transfer/_page_backup wrap ALL
mooncake L3 calls in mem_pool_host.layout_lease()
(hybrid_cache_controller.py:1481/1567); (3) _begin_compaction blocks on
_layout_users and syncs _pending_transfer_events (which l2_transfer posts);
(4) plain HiCacheController (no layout_lease) pairs only with
non-compacting plain MHA/MLA pools. Residual hardening idea (mooncake-audit):
assert base HiCacheController host pools don't require layout leases so a
future base+unified pairing can't silently recreate the hole. So: NOT
exploitable in shipped pairings.
The L2 device-side transfer path DOES lease correctly
(l2_transfer.py:103-128 + per-release finish events; event overwrite for
same transfer_key is SAVED by same-stream ordering: the last recorded
finish event on a stream subsumes all earlier DMA on that stream. CLEAN
under the one-stream-per-direction-per-engine invariant).

### F6. [CLEAN] HostPoolGroup.resolve/release_transfers (group.py:175-260)
derived transfers alias source indices and are skipped in release -> single
owner; rollback frees only fresh allocs. ASSUMPTION: callers never double
invoke release_transfers on one list nor free host_indices independently.

### F7. [CLEAN] host_memory.py, pool_host/common.py alloc/register/unregister:
registration chunking records per-range bases for unregister; destroy()
idempotent; HIP frombuffer finalize ties free to tensor lifetime; rollback
of partial registration ok. ASSUMPTION: destroy() only runs at shutdown (no
in-flight IO).

### F8. [CLEAN] DeepSeekV4StateHostPool: no allocator/no free list by design
(free raises). Its host_indices alias SWA pool rows; protection depends on
group release_transfers skipping derived transfers (F6 assumption).

### F9. [CLEAN] LogicalHostPool page-aligned alloc/free arithmetic itself
(rounding, merge) matches base semantics.

### Remaining minor notes
- hisparse.alloc_paged_token_slots: writes page-aligned slots into request
  table; table overflow raises (loud via shape mismatch), not silent. OK.
- v0.5.14 base free() also was NOT synchronized (race between concurrent
  free/free and alloc -> lost updates/dup indices). v0.5.15+ has RLock.
- mem_state tensor (base.py clear()) is written but never read anywhere in
  pool_host; dead state (vestigial), not a bug.

### F3. Reclaim order: free of host slots referenced by in-flight direct IO —
TODO (boundary with cache controller; check how eviction vs io_uring/direct
read completion is ordered).

### F4. alloc()>need rounding in DeepSeekV4PagedHostPool.alloc (mph:434-447):
rounds need_size up to slot_page_size and returns the rounded set; callers
using device_indices numel to drive transfers must use returned tensor, not
requested count. Round-trip: free() does not re-round/validate — a free of a
non-page-multiple count leaks the tail slots. Check callers.
Also: free() of an indices tensor containing duplicates → same slot twice in
release_slots (silent, no guard). Ties to F1.

---
## Clean areas (with assumptions)
- host_memory.py: cgroup headroom calc; sizing-only, no allocator state. CLEAN
  for memory safety (assumption: misuse only mis-sizes pools).
- HostKVCache.destroy(): idempotent via _destroyed flag; only touches
  kv_buffer. CLEAN.
- HostPoolGroup.resolve_host_transfers/release_transfers (group.py:175-260):
  derived transfers (indices_from_pool set) alias source pool indices and are
  skipped in release → single-free ownership invariant holds there.
  rollback() frees only freshly-allocated entries. ASSUMPTION: callers never
  call release_transfers on the same transfer list twice, and never free
  transfer.host_indices themselves after release_transfers. To verify via
  caller audit (controller).
