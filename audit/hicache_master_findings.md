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
  SALVAGED VULN WINDOW (prefetch-ctl version archaeology, hicache-lead spot-
  verified v0.5.17:1869-1917 by line): hi_mamba_radix_cache.py
  check_prefetch_progress MIN-reduces completed_tokens across TP but reads
  rank-local pool_storage_result.extra_pool_hit_pages[MAMBA] >= 1 to attach/
  free the mamba host slot with NO TP sync → asymmetric mamba sidecar read:
  one rank attaches + skips free, other frees the same slot + skips attach →
  divergent mamba host trees + freed slot reissued while tree-referenced.
  AFFECTED: v0.5.14 (~l.1815-1866), v0.5.15.post1 (~l.1815-1861), v0.5.17
  (l.1869-1917) — all same shape. Unified SWA path on 0.5.15+ IS guarded
  (packed MIN unified_radix_cache.py:2021-2033). v0.5.20 removed
  hi_mamba_radix_cache.py (resolved by rewrite); HEAD guarded by override.
  K(v0.5.16): VERIFIED affected directly from the public tag by
  version-triage (rank-local .get(PoolName.MAMBA,0) after MIN-on-tokens
  ~:1843). K3 internal fork: recheck flagged to lead/K3 owners (fork-heavy
  surface). Exposure note: .14 is technically affected but no .14 prod fleet
  runs mamba models → practical exposure is K3/mamba only. Fix analogue for backports: pack the mamba
  hit into the existing terminate-time MIN all_reduce (mirror unified:2021).
- ~~storage_prefetch_lifecycle::test_two_rank_completion_capacity_and_anchor_loss
  fails at clean HEAD~~ — ENV ARTIFACT (prefetch-ctl-audit root-caused):
  65-char sandbox hostname breaks gloo unix-socket paths ("File name too
  long"). `hostname scpu` before multirank CPU-gloo tests; suite passes 10/10
  after. Also explains test_prefill_result_polling, test_specv2_kvcache_offloading.
- C5 RETRACTED — env artifact (prefetch-ctl-audit): ninja missing in thin
  sandbox → hicache_hash_cpp JIT build fails → hash raises BEFORE the alloc
  attempt → except-early-return skips mark_terminate → the failing assert
  reads 'not terminated' but that IS the correct pre-alloc behavior. With
  toolchain (ninja+libssl-dev+python3-dev): suite 11/11. Intended alloc-error
  discipline verified independently by two-rank sim (KV rollback free +
  op terminated + next ticket proceeds). NO BUG.
- T3 (low, LOGGED, not fixing): graceful shutdown doesn't drain ongoing_offload
  before host-pool destroy; teardown order issue only (process exiting;
  worst case = crashed-on-exit noise, not served corruption). Recommend
  upstream hardening note, not a patch in this audit. (decode-offload-audit)
- B2 (note): flush_cache idle-gate misses ongoing_backup; reset() 10s raise
  uncaught — operator-triggered only.

## Version matrix for confirmed bugs
| Bug | v0.5.14 | v0.5.15.post1 | K3 (~v0.5.16) | v0.5.17 | v0.5.20 | HEAD/lat est |
|---|---|---|---|---|---|---|
| Mooncake short-read-as-success (all get paths) | YES | YES | assumed YES | YES | YES | YES → FIXED (MR #15) |
| Host-pool double-free, no guard | YES (base free unguarded+unlocked) | partial: base guarded, overrides unguarded | assumed | partial | partial | partial → FIXED (MR #13) |
| free() -1 sentinel wrap | YES | YES | YES | YES | YES | YES → FIXED (MR #13) |
| Retract-vs-offload UAF (no drain) | YES | YES | assumed | YES | YES | guarded by drain (main) |
| B1 drain raise → SIGQUIT | n/a (no drain) | n/a | ? | n/a | n/a | YES → fixed twice (V8 merged dev + call-site catch #52 in review) |
| C4 prefetch sidecar TP divergence | no pool_hits field — OPEN archaeology | no pool_hits — OPEN | OPEN | no pool_hits — OPEN | guarded (override) | guarded (override); REFUTED as described |
| C5 PP-ticket alloc-error leak | ? (pending) | ? | ? | ? | ? | pre-existing failure → fix in progress (prefetch-ctl) |
Upstream fixed-in version for short-read: NONE (still live upstream HEAD; our fix is the delta we carry).

## Clean areas (challenged, with assumptions)
- Prefetch timeout/abort host-page lifetime at HEAD: guarded via completed-prefix
  free + ack-deferred tail release (pinned by shipped tests incl. SWA variant).
- write_through double-write/ownership: commit marking + set-if-absent; evict-
  during-backup guarded by inc_lock_ref/dec at ack-drain.
- Direct-linker layer-wise loads: exact byte-count validation + session checks.
- L2 device transfers (l2_transfer.py): layout lease + same-stream ordering.
- Embedding-store buffer lifetime (controller side): FILLING not evictable; PUT pinned.
- Set paths (res==0 checks) + exists-error conservatism: correct.
- Group-id construction from tagged keys (no cross-tenant group collision);
  pp_rank rsplit rewrite correct for all suffix layouts.
- page_first_direct pointer/stride math (MLA+MHA) at page 64: correct GIVEN
  page-aligned contiguous indices (flagged assumption recorded).
- Unified host-pool backing free(): own duplicate/ownership detection.
- HostPoolGroup resolve/rollback/release: single-owner discipline verified.
- common.py register/unregister ranges + destroy ordering: clean.
- mla_host_dedup: inert in production (dedup compat excludes mooncake store).
- lmcache_unified min-clamp: historical invariant violator, out of prod scope.
- unified checkpoint cache_protected_len shrink: fail-stop assert protocol
  (decoded-offload-audit trace; closes earlier-audit "unified checkpoint" item).

## Coverage log
- 2026-10-16 hicache-lead: orient — file inventory, API surface of HiCacheStorage
  (batch_{get,set,exists}_v1/v2), team spawned, upstream tags fetched.
