# HOTFIX H-FREEGUARDS — host-pool free() double-free + -1 sentinel guards (F1+F2)

**Bugs (board #25, fixed on dev by hostpool-audit):**
- F1: free() overrides in DeepSeekV4PagedHostPool / LogicalHostPool /
  MambaPoolHost accepted duplicate frees silently (a doubly-freed slot is
  later handed to TWO live owners → cross-request host KV corruption); at
  v0.5.14 the base HostKVCache.free is also unguarded (and unlocked usage
  note).
- F2: no free() path rejected -1 sentinel / out-of-range indices; on torch,
  index -1 wraps to the LAST slot (slot_used[-1] marks an in-use slot free;
  -1 then enters the free list and is later issued as a live slot id →
  kv_buffer[-1] aliasing the final host row).

**Patch (per tag):** adds module-level in pool_host/base.py:
`assert_valid_host_slot_indices(indices, logical_size)` (sentinel + range
asserts), per-slot `slot_used` live-tracking (set at alloc, cleared at
free, double-free/double-alloc asserts), threading through the class
clear/alloc/free paths of:
- HostKVCache (pool_host/base.py): .14 full treatment; .15p1+ only the
  sentinel assert (slot_used double-free check already exists there).
- MambaPoolHost: in memory_pool_host.py at .14/.15p1 (raw-cat shape) and
  .16/.17 (release_slots shape); in pool_host/mamba.py at .20.
- LogicalHostPool + DeepSeekV4PagedHostPool (memory_pool_host.py):
  .14/.15p1 raw-cat+flatten shape (indices_cpu extracted then guarded);
  .16+ release_slots shape.

**Semantics notes:**
- Assert-based loud-fail (matches upstream-dev convention; note asserts
  disable under python -O — pos-conditions acceptable: the world has
  already crashed for callers if guards disabled).
- Mamba at .14/.15p1 didn't run its free() under torch.cat with cpu
  conversion; the patch preserves that native form.
- The .20 mamba patch also imports the helper into pool_host/mamba.py.

**Apply:** `git checkout <tag>; git apply H-FREEGUARDS.<tag>.patch`
Verified `git apply --check` against clean tags:
v0.5.14 (49e384ce9d30), v0.5.15.post1 (0b3bb0cbe318), v0.5.16 (fdebc938f7f4),
v0.5.17 (294816854627), v0.5.20 (94602c9c2b7c).

**K3 (~v0.5.16) note:** K3 fork carries internal edits to these pools
(mamba-unified host-pool, extra sub-pools). Verify the 5 anchor classes;
re-map by the anchor predicates (class names + free() body shapes listed
above). Regression reference (dev):
test/registered/unit/mem_cache/test_mem_pool_host_free_guards.py (+ dual-mode
loader) and test_mem_pool_host.py guard tests; 4/4 fail pre-fix.
