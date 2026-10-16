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
- [ ] Mooncake store short reads treated as success
- [ ] abort leaking pins / H2D into freed buffers
- [ ] prefetch policy timeout races
- [ ] write_through double-write/ownership
- [ ] page_first_direct layout index math (page 64)
- [ ] host dedup refcount (mla_host_dedup)
- [ ] group semantics across replicas sharing a backend tag
- [ ] decode offload interplay with retract/aborted requests

## Confirmed bugs
(none yet)

## Suspects under verification
(none yet)

## Clean areas (with assumptions)
(none yet)

## Coverage log
- 2026-10-16 hicache-lead: orient — file inventory, API surface of HiCacheStorage
  (batch_{get,set,exists}_v1/v2), team spawned, upstream tags fetched.
