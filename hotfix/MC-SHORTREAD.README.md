# HOTFIX MC-SHORTREAD — mooncake_store short reads treated as success (all get paths)

**Bug (board #33, fixed on dev):** mooncake_store get APIs judged success by the
sign of the returned byte count (`for res > 0` / `>= 0` / `< 0`) instead of
exact-bytes. A short read (mooncake engine RDMA/linker returning fewer bytes
than the KV page) silently produces a valid page with a stale tail —
cross-request KV corruption under any transient engine hiccup.

**Fixes (same as upstream-dev commit series, ported per tag):**
1. `_batch_postprocess`: group success iff every slice `res == group_size`
   (exact-byte equality) instead of `res > 0`.
2. `get()`: `get_result[0] == target_sizes[0]` instead of `>= 0`.
3. `batch_get()`: short-key detect `get_result[i] != target_sizes[i]` instead
   of `< 0`.
4. `_get_batch_zero_copy_impl` gains optional `expected_sizes` and forwards it
   to `batch_get_into_multi_buffers` (client-side treated convention compat —
   older engine versions ignore extra kwargs upstream clients accept
   `expected_sizes` natively; mock/test flag if your production
   mooncake_store client predates the kwarg — see NOTE below).
5. `set()`: guard the raw-engine empty-put call (`key is None` fast-path) to
   avoid the mooncake client crash on `batch_put_from([], [], [])` (board P2-2
   sibling).

**Semantics note for the multi-buffer form:** in these tags the public per-key
`target_sizes` are per-query-key scalars in one-to-one order with query keys
(after mha k/v expansion one buffer each); expected-equality therefore uses
`target_sizes[i]`. Equivalent to the dev-side `_expected_io_bytes` collapse at
single-slice layout; verification by per-tag compile + apply.

**Apply (per tag):**
```
git fetch <upstream-remote> <tag>
git checkout -b hotfix-mc-shortread <tag>      # or cherry into your prod branch
git apply MC-SHORTREAD.<tag>.patch             # verified `git apply --check`
```
Verified apply against clean tags (this swarm repo, hotfix/ dir):
- v0.5.14  (49e384ce9d30): MC-SHORTREAD.v0.5.14.patch
- v0.5.15.post1 (0b3bb0cbe318): MC-SHORTREAD.v0.5.15.post1.patch
- v0.5.16 (fdebc938f7f4): MC-SHORTREAD.v0.5.16.patch
- v0.5.17 (294816854627): MC-SHORTREAD.v0.5.17.patch
- v0.5.20 (94602c9c2b7c): MC-SHORTREAD.v0.5.20.patch

**K3 branch NOTE:** prod K3 (~v0.5.16 base, internal SHAs) — verify its
mooncake_store.py matches v0.5.16's shape in the 5 edited regions; if drifted,
re-map by searching for the same broken predicates (`res > 0 for res in
group`, `get_result[0] >= 0`, `get_result[i] < 0`).

**Regression reference (dev):** test/registered/unit/mem_cache/test_mooncake_short_read.py
(6 CPU cases, 5 fail pre-fix), test_mooncake_batch_set_empty.py.
