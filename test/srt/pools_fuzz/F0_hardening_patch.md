# F0 hardening patch skeleton (unified sub-pool free paths)

Status: SKELETON for pools-lead; applies cleanly conceptually, needs a GPU run for the kernel path.
Branch target: agent/unified-auditor -> dev.

## Confirmed behavior (CPU empirical, 2026-10-16)

```
alloc = build_swa_pool(page_size=4); full = alloc.full_attn_allocator
tok = alloc.alloc(16); alloc.free(tok.clone()); alloc.free(tok)   # double free
# -> full._free_phys_pages = [196,197,198,199,-1,-1,-1,-1]
# -> full.live_page_count = -4
# -> full._byte_accounting_violations() == []   (IDLE VERIFIER IS BLIND)
```

Consequences if it happens on GPU:
1. `free_unbind_inplace_kernel` stores `p2v[p]=-1` with p=-1 -> raw write at p2v_ptr-8 (OOB).
2. Later `take_physical`/`_flush` drain hands out physical page -1 -> `bind_inplace` stores at
   p2v_ptr-8 again (OOB), and v2p[v]=-1 => translate yields -1 => negative kv loc downline.
3. live_page_count underflow corrupts leak-invariant + capacity views (available_size can go UP).
4. Count-based END-pool verifier cannot catch it (span == live+holes+pending still balances);
   the FLOAT verifier only catches it incidentally via p2v-cross-check when the index_fill_ wrap
   also corrupted a live page (CPU semantics; GPU is OOB instead).

## Patch elements

### 1. Kernel guard (the root OOB write) — virtual_slot.py

In `free_unbind_inplace_kernel`, mask the p2v store on p >= 0:

```python
p = tl.load(v2p_ptr + v, mask=mask, other=0).to(tl.int64)
tl.store(out_phys_ptr + offs, p, mask=mask)
tl.store(v2p_ptr + v, -1, mask=mask)
tl.store(p2v_ptr + p, -1, mask=mask & (p >= 0))   # <- guard: stale free has p == -1
```

Same guard in the CPU reference branch (`p2v.index_fill_` currently WRAPS on -1):
```python
bound = phys_pages >= 0
p2v.index_fill_(0, phys_pages[bound], -1)
```
Note: the returned `phys_pages` still carries -1 for stale entries; callers filter below.

### 2. `_free_lazy` stale filter (unified_sub_pool.py)

After `freed_p_pages = free_unbind_inplace(...)`:

```python
stale = freed_p_pages < 0
if bool(stale.any().item()):
    n_stale = int(stale.sum().item())
    self._stats_n_stale_frees += n_stale          # new counter, surfaced in _maybe_emit_stats
    if envs.SGLANG_DEBUG_MEMORY_POOL.get():
        self._raise_stale_slot_assertion(free_v=free_v_pages, freed_p=freed_p_pages)
    logger.error(
        "MultiEndedAllocator(%r)._free_lazy: %d stale page free(s) (double-free "
        "or never-allocated); dropping them to protect the free list",
        self.sub_pool_name, n_stale,
    )
    live_mask = freed_p_pages >= 0
    free_v_pages = free_v_pages[live_mask]
    freed_p_pages = freed_p_pages[live_mask]
    if freed_p_pages.numel() == 0:
        return
```
(recycle virtual ids + hole-cat + live_page_count then proceed on the filtered set.)

### 3. `FloatMultiEndedAllocator.free` / `free_physical` — same filter before

`self._free_phys_pages = cat(...)`, mirrored on `free_v_pages`.

### 4. Idle verifier upgrades (turns the class from silent to self-reporting)

- `MultiEndedAllocator._byte_accounting_violations` (lazy): value-check holes:
  `(self._free_phys_pages >= 0).all()` and `(self._free_phys_pages < self.num_pages).all()`
  and duplicates check `len(set) == numel` (host-side; idle path only).
- `live_page_count < 0` -> violation string.
- Float `_byte_accounting_violations`: same hole-value checks.

### 5. `free_group_end` duplicate-page tripwire (F1)

In `MultiEndedAllocator.free_group_end` and `UnifiedSWA*._release_page_reps`, under
`SGLANG_DEBUG_MEMORY_POOL`: assert `len(set(reps_pages)) == len(reps_pages)`; on violation raise
with the radix-derived reps (this config asserts instead of silently double-inserting into
`free_virtual_ids`). Cheap common path: data-dependent only in debug mode.

## Regression tests (in test/srt/pools_fuzz/)

- test_f0_double_free_filtered: after the patch, double-free leaves free list clean,
  live_page_count non-negative, virtual ids not double-recycled, and the stale counter increments.
- test_f1_group_segment_dup_assert: debug mode raises on same page in two segments of one group.
- file already added: test_unified_alloc_fuzz.py (model checker, ps=4 & 64, pair + tri-pool) — PASS.

## Reachability (still open, radix team)

Who can double-free? Candidates: cache_finished+abort overlap, retract paths,
deferred decode KV release, grammar double release (known patch). radix-sched is hunting.
The patch makes the allocator robust regardless of the answer.
