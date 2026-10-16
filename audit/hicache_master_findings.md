# HiCache memory-safety audit — master findings (board item #6)

Owner: hicache-lead. Team channel: #hicache.
Production config: --enable-hierarchical-cache, write_through, page_first_direct,
direct io backend, hicache-ratio 1.5-16, mooncake storage backend
(enable_group_semantics=true, shared across replicas w/ backend tag),
prefetch policy timeout, --disaggregation-decode-enable-offload-kvcache,
page-size 64, kv-cache-dtype fp8_e4m3 (mostly) / bf16.

Production versions to triage every confirmed bug against:
v0.5.14, v0.5.15.post1, k3-branch (base 87dd4596 / bd4828e2), v0.5.17, v0.5.20.
Tags v0.5.14/v0.5.15.post1/v0.5.17/v0.5.20 fetched from upstream into local repo
(remote `upstream` = https://github.com/sgl-project/sglang). k3 base commit still to fetch.

## Sub-areas and owners
| Sub-area | Agent | Findings file (their branch) |
|---|---|---|
| host pools, free lists, page_first_direct, reclaim order | hostpool-audit | audit/hicache_hostpool_findings.md (agent/hostpool-audit) |
| mooncake store, storage API contract, l2_transfer | mooncake-audit | audit/hicache_mooncake_findings.md (agent/mooncake-audit) |
| prefetch/controller/scheduler glue, key scheme, auto-size | prefetch-ctl-audit | audit/hicache_prefetch_findings.md (agent/prefetch-ctl-audit) |
| decode offload mixin/manager host side, mla_host_dedup | decode-offload-audit | audit/hicache_decode_offload_findings.md (agent/decode-offload-audit) |

## Known classes to verify at HEAD (and find siblings)
- [ ] HiCache host reclaim order
- [x] host pool free-list double free — VERSION FACT (hicache-lead): v0.5.14
  `pool_host/base.py::HostKVCache.free()` is raw `torch.cat` of freed indices, no
  double-free guard (0 `slot_used` refs) → a double free silently aliases one host
  slot to two requests (cross-request KV corruption). v0.5.15.post1/v0.5.17/v0.5.20
  and HEAD have `slot_used` alloc/free asserts (crash-on-double-free; note:
  asserts vanish under `python -O`). Reachable double-free paths at HEAD = still
  being traced by hostpool-audit.
- [x] Mooncake store short reads treated as success — LIVE AT HEAD AND ALL PROD
  VERSIONS (hicache-lead): `mooncake_store.py::_batch_postprocess` (HEAD ~L1191-1202)
  maps `batch_get_into` byte counts to success via `all(res > 0 for res in group)` —
  ANY positive byte count (short read of 1 byte on a multi-MB page) is success, and
  the expected `buffer_sizes` computed at the call sites (L1217/1246) are never
  compared. Caller (`batch_get_v1/v2`, cache_controller prefetch completion) then
  treats the host page as fully valid KV → stale/garbage KV served → garbled output
  or cross-request data (shared store). Present in v0.5.14/15.post1/17/20 (same grep
  signature) + HEAD. Siblings + full caller chain: mooncake-audit (in progress).
  FIX direction: success iff res == expected size per slice (group semantics: all parts).
- [ ] abort leaking pins / H2D into freed buffers
- [ ] prefetch policy timeout races
- [ ] write_through double-write/ownership
- [ ] page_first_direct layout index math (page 64)
- [ ] host dedup refcount (mla_host_dedup)
- [ ] group semantics across replicas sharing a backend tag
- [ ] decode offload interplay with retract/aborted requests

## Confirmed bugs
1. **Mooncake store short reads treated as success** — FIXED @ e766366a
   (agent/hicache-lead). get paths judged success by sign of returned byte
   count; short read = valid page with stale tail. Affects HEAD + v0.5.14 /
   v0.5.15.post1 / v0.5.17 / v0.5.20 (all get APIs: _batch_postprocess res>0,
   batch_get <0, get >=0 incl. zero-byte = success). Fix: exact-byte equality
   on all get paths. Regression: test_mooncake_short_read.py 6 CPU cases
   (5 fail pre-fix). Independent verification: mooncake-audit (in flight).
   Siblings: storage/simm/hicache_simm.py, storage/npu_memcache/ have the
   same leniency (out of prod scope, noted).
2. **Host pool free-list double free** — version matrix only so far: v0.5.14
   unguarded free() (silent aliasing); v0.5.15.post1+/HEAD crash-guarded via
   slot_used asserts. Reachable double-free paths at HEAD: hostpool-audit
   tracing.

## Confirmed (cont.) — decoded-offload + routed
3. **Host-pool free(): -1 sentinel wrap** (F2) — free() accepts negative
   indices; slot_used[-1] marks the LAST live slot free → double-alloc w/o
   double-free; -1 enters free list → issued as live slot id → kv_buffer[-1]
   aliasing. All versions/all pools. FIX agent/hostpool-audit 1520122+1ad4f3d
   (assert_valid_host_slot_indices + guards in 3 overrides) — board #25
   APPROVED (hicache-lead review: completeness verified, env-control test run).
4. **B1 drain_before_retraction uncaught raise → engine SIGQUIT** (main only;
   all ranks via MAX allreduce; triggers: KV pressure + copy lag, pause).
   Repro: agent/decode-offload-audit audit/repros/repro_drain_crash.py.
   ROUTED: disagg-lead (call-site files are disagg-lane per seam).
5. **Retract-vs-offload UAF** in v0.5.14/v0.5.15.post1/v0.5.17/v0.5.20 (no
   drain guard in those tags; remove/retract frees device rows under in-flight
   D2H offload → stale copy persists wrong KV under correct hashes).
   Main guards via drain (which has B1). K3 assumed affected.
6. **P2 raw batch_set empty-RPC** (mooncake_store.set(): batch_put_from([],[],[])
   when key exists → native crash vector; unused in prod) — fix by mooncake-audit
   on their branch; integrate with #33 merge.
7. ~~**Unified checkpoint cache_protected_len shrink sibling**~~ — CLOSED CLEAN
   (decode-offload-audit trace): unified protected is always tree-page-boundary
   (±<1 page req-owned tail); shrink classes cannot free tree-owned pages;
   assert L1210 makes violations fail-stop crashes, not UAF. Fix protocol
   extracted for radix-lead's plain-radix bug (adopt the assert + invariant).
   Historical violator: lmcache_unified_radix_cache.py:763 (min-clamp; OOS).

## Suspects under verification
- C4: ~~PrefetchAck.pool_hits not rank-reduced ⇒ TP host-tree divergence~~
  REFUTED at HEAD + v0.5.20 (hicache-lead review): HybridCacheController
  ._reduce_prefetch_ack override (:1796) already MIN-reduces ack.pool_hits over
  the fixed PoolName universe with failure-safe empty-dict semantics; every
  ack-carrying producer (only Hybrid._page_transfer_sidecar) flows through it.
  Team repro hand-diverged decision-fn inputs, bypassing the reducing sync
  thread → unreachable. Proposed base-class fix was itself a WEDGE (empty-dict
  ranks would skip the new collective) — parked, never merged. Per-version
  archaeological note: v0.5.14/15/17 have no PrefetchAck.pool_hits field at
  all (pre-dates the ack design); whether their SWA prefetch has an equivalent
  asymmetric path via different structure is OPEN (prefetch-ctl continuing).
  Teach-in: reviewing reviewer MUST chase overridden methods up the MRO before
  accepting "no sync exists" claims on subclass code paths.
- ~~storage_prefetch_lifecycle::test_two_rank_completion_capacity_and_anchor_loss
  fails at clean HEAD~~ — ENV ARTIFACT (prefetch-ctl-audit root-caused):
  65-char sandbox hostname breaks gloo unix-socket paths ("File name too
  long"). `hostname scpu` before multirank CPU-gloo tests; suite passes 10/10
  after. Also explains test_prefill_result_polling, test_specv2_kvcache_offloading.
- C5 (NEW, pre-existing HEAD): test_pp_prefetch_ticket::test_allocation_error
  _preserves_ack_sequence_and_next_ticket fails — sidecar host-pool alloc error
  (KeyError PoolName.SWA) on a PP-broadcast ticket leaves the op NOT terminated
  and KV staging UNFREED. Genuine leak-path candidate (PP>1 + SWA + hicache
  config). Assigned: prefetch-ctl-audit.
- T3 (low, LOGGED, not fixing): graceful shutdown doesn't drain ongoing_offload
  before host-pool destroy; teardown order issue only (process exiting;
  worst case = crashed-on-exit noise, not served corruption). Recommend
  upstream hardening note, not a patch in this audit. (decode-offload-audit)
- B2 (note): flush_cache idle-gate misses ongoing_backup; reset() 10s raise
  uncaught — operator-triggered only.

## Clean areas (with assumptions)
(none yet)

## Coverage log
- 2026-10-16 hicache-lead: orient — file inventory, API surface of HiCacheStorage
  (batch_{get,set,exists}_v1/v2), team spawned, upstream tags fetched.
