# Mamba / hybrid linear-attention KV cache audit — findings (mamba-auditor)

Branch: agent/mamba-auditor. Scope: Kimi K3 mamba configs (extra_buffer, lazy,
max-mamba-cache-size, replayssm spec), unified radix tree, disaggregated serving.

## Candidate 1: Deferred mamba COW source is NOT protected between prefix match and the forward that performs the copy
- Status: CANDIDATE (needs independent verification)
- Class: COW source use-after-evict (known bug class #2)
- Code:
  - `python/sglang/srt/mem_cache/unified_cache/components/mamba.py:192-221`
    `finalize_match_result_in_cache`: sets `req.kv.mamba_cow_src_index =
    tree_core.get_component_device_value(result.best_match_node, MAMBA)` and
    (if needed) `req.kv.mamba_pool_idx = mamba_allocator.alloc(1)` as dst.
  - `python/sglang/srt/managers/schedule_batch.py:3154-3175`
    `_collect_deferred_mamba_cow_and_clear`: gathers `mamba_cow_src_index` per
    req into batch-level `mamba_cow_src_indices` consumed at forward start.
  - `python/sglang/srt/model_executor/model_runner.py:1920-1937`
    `_execute_mamba_deferred_ops` (extend forward): actually performs the copy.
  - `python/sglang/srt/managers/schedule_policy.py:1607`
    `req.lock = self.tree_cache.lock(req.last_node)` — locks ONLY
    `last_device_node`.
  - `python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:713-730`
    `inc_lock_ref`: locks only the given node, NOT its ancestors.
- Mechanism:
  1. best_match_node (the node *holding the mamba state*) can be a strict
     ancestor of last_device_node: mamba states live only at checkpoint-grid
     depths (and are capped per path by mamba_max_states_per_path / evicted
     shallow-first), while Full-KV match often runs deeper
     (`mamba_branching_seqlen` exists precisely for "Full KV may extend beyond
     the latest reusable Mamba state" — components/mamba.py:143-157).
  2. Between `finalize_match_result_in_cache` (in `init_next_round_input`,
     scheduler.py:4177) and the extend forward that copies, the source node X
     still sits in the MAMBA device LRU with lock_ref==0, because the request
     locks only `req.last_node` (typically `last_device_node != X`).
  3. Any mamba allocation in that window evicts X:
     `_alloc_mamba_slot` / `_alloc_request_state_slot` in components/mamba.py
     (alloc → evict_for_alloc(mamba_num=1) → alloc) — called from the NEXT
     waiting request's `finalize_match_result_in_cache`/`prepare_load_back`
     inside the same `get_new_batch_prefill` adder loop, before the batch's
     forward. Also any `evict_for_alloc` for KV pressure in the same window.
  4. X's slot is freed via `evict_component` (mamba.py:~~~320) into
     `device_frees[MAMBA]` → `mamba_allocator.free`, then handed to another
     request or donation and **overwritten**.
  5. Req A's pending COW then copies a rewritten slot into its fresh active
     slot → cross-request mamba-state leakage / garbled output. Silent.
- Why existing comments don't cover it: the opt-in comment at
  unified_radix_cache.py:1221-1227 ("Safe only because any future COW source is
  the COWing request's own admission-locked last_node ... locked before the
  next alloc") describes the post-checkpoint re-lock path and COWs anchored on
  the request's *own* last_node. The admission-time initial match is anchored
  on best_match_node, which that lock does not cover when it is an ancestor.
- PoC sketch (cpu harness): unified cache with MAMBA+FULL components; insert
  chain so that node L (depth 128) has FULL KV, ancestor X (depth 64) holds
  the only mamba state; new req R1 matches → cow_src = X.value, R1 locks L
  only; admit R2 whose mamba dst alloc triggers evict_for_alloc(mamba_num=1)
  → X evicted, slot reused; R1's deferred COW then reads the reused slot.
  Check: R1's copied state != state it should have had; slot owned twice.
- Fix sketch: in `finalize_match_result_in_cache`, inc_lock_ref the
  best_match_node for the MAMBA component when a COW is staged, and pair it
  with dec after `_collect_deferred_mamba_cow_and_clear` (or record the lock
  receipt on the req and release at prepare/import); alternatively anchor the
  COW source copy eagerly (small: one slot) into the dst at match time instead
  of deferring.

## In-progress checks (placeholders)
- extra_buffer lazy/spec ping-pong lifecycle across retract/abort/disagg-abort
- -1 sink writes: set_mamba_track_indices_from_reqs freed_rows coverage vs
  TARGET_VERIFY + retract interplay
- replayssm (no_buffer) write_pos vs donation consistency / abort paths
