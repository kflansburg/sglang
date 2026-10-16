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

---

## Findings (ranked)

### F1. [SUSPECT] Missing double-free guard in subclass free() implementations
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

### F2. page_first_direct layout index math at page_size 64 — TODO
Plan: read mla.py / mha.py get_flat_data_page / set_from_flat_data_page /
transfer index math; verify against sgl_kernel direct kernels' strides.

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
