# Hotfix RADIX-38: RadixCache trimmed-finish frees tree-owned KV rows

Bug (board #38, swarm audit): when a finished request is trimmed below the
page-aligned length already recorded in cache_protected_len by unfinished
checkpoints (stop-string trims, strip_thinking-owned trims, spec-verify
overshoot), the release path frees rows the radix tree still references;
the paged allocator then hands a tree-owned page to the next allocation
(cross-request KV corruption) and the eviction later double-frees it.

Affected code path: RadixCache.cache_finished_req (all tagged prod versions;
HEAD equivalent is mem_cache/common.py release_kv_cache -> checkpoint).

Apply: at a CLEAN checkout of the tag, run

    git apply hotfix/38.<tag>.patch

Verified: every patch (38.v0.5.14 / v0.5.15.post1 / v0.5.16 / v0.5.17 /
v0.5.20) was generated and apply-checked with `git apply --check` against the
clean tag, then compiled with python3 -m py_compile. Behavior change:
cache_finished_req never inserts or frees below swr.cache_protected_len;
the freed tail is clamped to start at max(key_len, cache_protected_len).
The lock release at the end of the function is unchanged (runs in every
case). The upstream-shape fix merged on dev (early-return in checkpoint)
is the equivalent of these per-tag plants.
