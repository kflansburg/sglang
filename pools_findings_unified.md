# unified-auditor findings

Scope: unified_memory_pool.py, allocator/unified_sub_pool.py, unified_mamba.py, unified_hybrid_swa.py,
unified_draft_pool.py, kv_cache_configurator.py, allocation.py, allocation_sizing.py, kv_index_translator.py,
kv_loc_plan.py, base_swa_memory_pool.py, swa_memory_pool.py, deepseek_v4_memory_pool.py (pool mechanics),
allocator/mamba.py, mamba_slot_fused.py, mamba_checkpoint_pool.py.

Production config lens: --page-size 64, fp8_e4m3/bf16, chunked prefill 12288-32768, DP attention 1-8,
SWA + hybrid mamba (Kimi K3: --mamba-radix-cache-strategy extra_buffer), disaggregated prefill/decode.

## Findings (incremental)

### F0. Known-class verification status (upstream HEAD, 2026-10-16)

- "unified sub-pool double-free and -1 sink OOB write": PARTIALLY MITIGATED by design.
  - Base EAGER free (`unified_sub_pool.py:1290-1310`) has stale-slot assert on `(freed_p_pages < 0).any()`.
  - `_free_lazy` + `free_unbind_inplace_kernel` (`python/sglang/kernels/ops/memory/virtual_slot.py:101-127`):
    on a double-freed virtual page, v2p reads -1, then `tl.store(p2v_ptr + p, -1)` = **raw write at
    p2v_ptr - 8 bytes (BEFORE the table)** — kernel has NO bounds guard. Safety rests entirely on the
    caller contract "must not double-free". A GPU OOB write of 8 bytes here corrupts whatever tensor
    the allocator placed below p2v (undefined).
  - SWA composite (`unified_hybrid_swa.py:566-712`) filters swa frees with tombstone mask
    `swa_v2p_pages > 0` (strictly: -1 tombstoned and page-0 padding-sink both skipped) — a double-free
    on the swa side is filtered at the composite. Full side relies on "full owns the ids".
  - `FloatMultiEndedAllocator.free` (`unified_sub_pool.py:2352+`): NO stale-slot check even in a debug
    mode; on double-free: `freed_p_pages` contains -1 -> `physical_to_virtual.index_fill_(0, p2v_with_-1)`
    (torch index_fill_ with negative index => WRAPS: writes p2v[-1] == p2v[num_pages-1]: a LIVE page
    belonging to the high END pool gets tombstoned) AND -1 joins `_free_phys_pages`, from which
    `take_physical_pages` later hands out physical page **-1**, and `bind` then writes
    `tl.store(p2v_ptr - 1)` (OOB write) + v2p[v] = -1 (translate -> negative kv loc -> OOB in
    set_kv_buffer). Severity gated on reachability of a double-free into a float pool: need to find
    a caller. IN PROGRESS.

### F1. `free_group_end` duplicate pages (documented contract, unenforced)
`MultiEndedAllocator.free_group_end` (unified_sub_pool.py:1980): `reps = torch.cat(pending)` then
`free(reps, _pages=reps // page_size)` — if two segments in one group touch the same page (caller bug),
the SAME virtual page enters `free_virtual_ids` twice -> same virtual page allocated to two requests
-> cross-request KV aliasing (silent corruption, no OOB). Contract comment at free_segment says
"a page must be freed by only one call per group" — not asserted. Callers: radix/scheduler paths; flag
to radix-lead to confirm the invariant is maintained at cache_finished/unfinished with page 64.

