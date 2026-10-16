# Hotfixes: EMB / multimodal embedding global-cache memory-safety (swarm audit, radix-emb)

Board items: EMB #13/#14/#15/#18/#36. Full analysis: findings-emb.md (dev).

## N4 — Mooncake embedding GET short read treated as success (board #14)
Affected: ALL five production tags (file exists since v0.5.14).
Sites: mooncake_embedding_store.py batch_get (all tags),
batch_get_into_multi_buffers (v0.5.15.post1+).
`res > 0` accepted partial reads: a short read marked the entry READY with
partial garbage, then served that embedding for the hash to the current and
all later requests (cross-request corruption, silent).
Fix: success iff bytes read == requested bytes (per key; sum of slice sizes
for multi-buffer). Affects only the GET read paths; PUT semantics unchanged.

Apply per tag:
    git apply hotfix/N4-EMBEDDING-SHORTREAD.<tag>.patch

Verified: git apply --check + py_compile on clean
v0.5.14 / v0.5.15.post1 / v0.5.16 / v0.5.17 / v0.5.20.

## Encoder global-cache family (encoder/server.py + embedding_cache_controller.py)
Present only in v0.5.20 per version-triage ledger (#4); earlier tags have only a stub.

Kimi-K3 branch (internal fork, base ~v0.5.16; SHAs 87dd4596 / bd4828e2,
not public): patches target the PUBLIC v0.5.16 tag per version-triage
convention; line numbers were derived from public tags, so a vendor fork
carrying local changes to these files may need the hunks re-derived
against its own checkout (mooncake_embedding_store.py is small and
unlikely to diverge; mamba/nixl files are the known fork-heavy ones).

### EMB13 — pool-page leaks on encoder assembly error paths (board #13)
GPU assembly raised mid-loop after pins were taken (pins never released ->
pages un-evictable forever); exceptions after staging dropped D2H handles ->
entries stuck FILLING (never evictable -> permanent pool shrink); copy-launch
failure mid-loop leaked pending FILLING entries. Fix: release pins/handles on
all error paths; new controller cancel_store_to_pool (waits issued copies
BEFORE freeing pages — no free-while-in-flight).

### EMB15 — ongoing_prefetch bookkeeping leak (board #15)
Prefetch ops keyed by req_id were never reaped on the encoder 60s timeout /
abort path. Fix: release_prefetch on the timeout path + stale-op replacement
on repeated prefetch for the same req_id. Bookkeeping only; the io thread
still resolves the entries' pages (see EMB36 for the wedged-backend case).

### EMB18 — release_pool_views unpins entries the caller never pinned (board #18)
get_pool_views pins only READY entries; release_pool_views unpinned any
present entry. A FILLING->READY transition between get and release let one
request drop a CONCURRENT request's pin -> entry evictable under a live read
(use-after-free, wrong-image embeddings across requests). Fix: release exactly
the actually-pinned hashes with multiplicity (server-side), contract
documented in the controller.

### EMB36 — wedged Mooncake store call wedges the controller io thread (board #36)
A hung batch_get blocked the single io loop forever: later GETs/PUTs queued,
FILLING entries accumulated un-evictably, pool died, hash became a permanent
miss. Fix: worker-thread per store call with SGLANG_MM_STORE_OP_DEADLINE_S
(default 120s); late ops are NOT freed at timeout (the native transfer may
still be in flight — freeing would corrupt reallocated pages); a late-op
registry resolves via _finish_get/_finish_put when the worker actually
returns. Observability: stats late_store_ops + get_stats late_ops_outstanding.

Apply (v0.5.20 only):
    git apply hotfix/EMB13.v0.5.20.patch hotfix/EMB15.v0.5.20.patch \
        hotfix/EMB18.v0.5.20.patch hotfix/EMB36.v0.5.20.patch \
        hotfix/N4-EMBEDDING-SHORTREAD.v0.5.20.patch

Verified: each patch git apply --check + py_compile against clean 94602c9c2b7c
(v0.5.20); cumulative application of all five F patches applies and compiles.
Regression tests for all fixes are in
test/registered/unit/mem_cache/test_embedding_cache_controller.py (dev branch;
backport of the test file is optional — it imports schedule_batch).
