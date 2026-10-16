# Triage at HEAD (main, 3420bae) — status per known bug class
Legend: SP=STILL PRESENT, FU=FIXED UPSTREAM, RW=REWRITTEN

## Known locally-patched classes (13)
1. deferred decode KV release — SP: machinery exists at HEAD in python/sglang/srt/disaggregation/decode.py:2499+ (enable_deferred_kv_release, _deferred_releases, deferred_kv_release_timeout). Need to verify correctness vs our local patch.
2. disaggregation admission abort — TBD
3. disaggregation prefill abort cache leak — TBD
4. disaggregation grammar double release — TBD
5. disaggregation retract with SWA offload guard — TBD
6. paged allocator capacity check — TBD
7. Mooncake transfer batch indices — TBD
8. Mooncake transfer worker keep-alive — TBD (heartbeat checker thread exists, conn.py:3022 _start_heartbeat_checker_thread)
9. HiCache host reclaim order — TBD (host_memory.py:115-137 has reclaim-order logic)
10. NIXL reload of an invalidated peer — TBD
11. DSA tail req_pool index — TBD
12. KV integrity tagging — checksum.py exists in disaggregation/; TBD whether it matches our "tagging" patch
13. decode weight tracing — TBD

## Earlier-audit findings (14)
A. RadixCache.checkpoint shrink of cache_protected_len — LIKELY SP: checkpoint() at radix_cache.py:508 assigns req.kv.cache_protected_len = len(new_indices) after match_prefix on the re-inserted key; need to verify whether new_indices can be shorter than the previous cache_protected_len (e.g. checkpoint with smaller up_to). _insert_cache free_segment(kv_indices[cache_protected_len:prefix_len]) clips empty if prefix_len<protected, so no double-free in that call, but the tree-ownership invariant (protected monotonic) can be violated at the write step.
B. unified cache checkpoint same class — TBD (mem_cache/unified_cache/)
C. SWA reserved page 0 to live requests — TBD (swa_memory_pool.py)
D. SWA alloc_decode one-sided leak — TBD
E. unified sub-pool/mamba double-free + -1 sink OOB — TBD
F. mamba COW source use-after-evict — TBD
G. stale prefix-match at admission — TBD (scheduler admission)
H. dec_lock_ref underflow — code exists radix_cache.py:597; need to check guard. inc/dec via TreeLock; TBD.
I. host pool free-list double free — TBD (pool_host/)
J. unified pool move_kv_cache OOB — TBD (unified_memory_pool.py)
K. disagg 30s-timeout free while RDMA in flight — conn.py:2416 batch_transfer_sync timeout noted; TBD whether free happens while write may be in flight.
L. embedding cache unpin of entries not pinned by caller — TBD (embedding_cache_controller.py)
M. abort leaking pins / H2D into freed buffers — TBD
N. Mooncake store short reads treated as success — SP: mooncake_store.py:1406 batch_get checks only get_result[i] < 0 → any short read 0<=r<size counted as success; returns len(keys)//key_multiplier. batch_exists also accepts any ==1. Affected: prefetch from mooncake store can silently load partial/garbage KV → cross-request leakage/garble.
