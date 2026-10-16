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

## Suspects under verification
(none yet)

## Clean areas (with assumptions)
(none yet)

## Coverage log
- 2026-10-16 hicache-lead: orient — file inventory, API surface of HiCacheStorage
  (batch_{get,set,exists}_v1/v2), team spawned, upstream tags fetched.
