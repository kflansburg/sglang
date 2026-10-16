# Radix-lane findings (radix-lead, radix-emb, radix-sched) — board item #3

Scope: radix_cache.py, base_prefix_cache.py, evict_policy.py, flush_cache.py, events.py,
embedding_cache_controller.py/embedding_store.py/multimodal_cache.py/index_key_cache.py (radix-emb),
scheduler.py abort/retract/finish + schedule_batch.py ownership + schedule_policy.py (radix-sched).
Production lens: disaggregated prefill/decode everywhere, page-size 64, fp8 e4m3, chunked prefill
12-32k, HiCache on, DP attention 1-8.

## CONFIRMED + FIXED (branch, board item, tests)

1. **RadixCache.checkpoint shrinks cache_protected_len → frees tree-owned pages** (#38, agent/radix-lead,
   REVIEWED+APPROVED by radix-sched). Trigger: disagg-decode mid-generation checkpoint_kv_cache advances
   protection into output; trimmed finish (stop-string / spec-verify overshoot / strip_thinking_cache)
   → release_kv_cache checkpoints at smaller owned_kv_len → cache_protected_len shrunk →
   free_kv_row([protected, owned)) spans rows rebound to tree pages → paged free_segment releases whole
   tree-referenced page → cross-request KV corruption + double-free at eviction.
   Fix: checkpoint early-returns when page-aligned publish len ≤ cache_protected_len (monotone).
   Repro at HEAD (page 64 semantics at page 4, PagedTokenToKVPoolAllocator, CPU): protected 128→64,
   free_kv_row([64,100)) freed tree page, match still served len=128, next alloc handed out the page.
   Test: test_checkpoint_never_shrinks_cache_protected_len (fails pre-fix).

2. **lock_ref underflow guards** (#45, agent/radix-lead, in review w/ radix-emb).
   TreeLock.released assert in BasePrefixCache.unlock + UnifiedRadixCache.unlock; dec_lock_ref asserts
   >0 per node. Proven before: double-unlock → lock_ref −1 → node evictable-while-held; dec walk over
   detached stale node decrements live ancestors. test_decode_radix_lock_ref 13/13 still pass.

3. **EMB pin/FILLING leaks on encoder assembly error paths** (#13, agent/radix-emb, APPROVED).
4. **EMB mooncake short-read as success** (#14, exact-byte predicate, APPROVED; matches hicache canonical).
5. **EMB ongoing_prefetch metadata leak** (#15, APPROVED).
6. **EMB release_pool_views unpins what caller never pinned (TOCTOU pin-theft UAF)** (#18, APPROVED; v0.5.20+ only).
7. **EMB/spec wedged io-loop (mooncake call never returns)** (#36, fixed by radix-emb, verified by spec-auditor).
8. **SCHED retract/capacity under-estimate by 1 page for committed≡63 (mod 64) under overlap** (#16,
   agent/radix-sched, APPROVED by radix-lead; crash-class under memory pressure).

## CLEAN VERDICTS (need challengers where marked)

- **Unified cache checkpoint_into_tree NOT vulnerable to the shrink class** (fail-stop: assert bans
  ≥1-page shrink; only request-owned <1-page tail can shrink). Independently traced twice
  (hicache-lead msg #637, decode-offload-audit #586). Challenger: radix-sched assigned, pending.
- **Page-alignment contract** (radix-sched): match_prefix/insert produce page-aligned lens →
  cache_protected_len page-aligned → non-disagg free_kv_row start alignment safe at HEAD.
- **DP-attention req_pool_indices**: rank-local pools; double-free crashes loudly. schedule_batch side clean.
- **Scheduler exception paths (#10)**: event loop fails loud; no silent-KV-corruption handlers.
- **EMB clean claims** (mm hash-collision defended via content-sha256+token-count; insert-vs-D2H race
  READY-gated; MultiModalStaticCache accounting consistent) — challenger: radix-sched assigned.

## OPEN / FLAGGED

- **Stale prefix-match at admission**: generic path defended by init_next_round_input per-req rematch
  + _lock_node temp pin in add_one_req (radix-sched verdicts 3/4). Tree-side: inc_lock_ref takes a
  detached stale node silently (probe-verified); dec side now asserts (#45). Caller skip-paths
  (retracted reqs, disagg decode rematch decode.py:844/decode_hicache_mixin.py:214, buffer_pipeline
  deferred admission pipeline.py defer_staged_admission) — radix-sched enumerating evidence.
- **disagg decode cache_protected_len=total_prefix_len incl L2/L3 host-gap rows** (decode.py:1604):
  early free of host-target rows, bounded (unique-based free) — flagged to disagg-lead/pools-lead.
- **buffer_mode pipeline.py:1514 protected=matched_len** from staged prefetch ctx — page-alignment
  confirm flagged to hicache.
- **TLru-on-legacy-cache startup guard** (robustness, agent/radix-lead commit 10744f3): eviction policy
  tlru crashed AttributeError mid-eviction on legacy TreeNode; now ValueError at cache construction.

## FUZZ NOTE

400-op invariant fuzz (random insert/match/lock/unlock/evict/checkpoint/retract over RadixCache +
PagedTokenToKVPoolAllocator, checking accounting + "tree refs never free" + "locked path never free"):
with honest ownership modelling (unlock ⇒ retract reset), early-failures traced back to fuzz row-reuse
modelling artifacts (ReqToTokenPool free_row/reuse semantics differ from production), not code bugs;
 parked — harness in skill catalog for hardening later. Unit tests + traced reasoning are the evidence.

## Coverage remaining

radix_cache.py: RadixKey bigram edges (features Lens pending), evict priority starvation (lower-prio),
pretty_print/dbg only. events.py: hash-value consistency for KV event consumers (read, low-risk).
flush_cache.py: CLI client only; flush semantics in scheduler (reset under active requests → io_struct
'inplace' guard) — read, no issue. embedding store/controller: full sweep done by radix-emb.
Scheduler abort/retract/finish: radix-sched sweep complete modulo challenge round.

Queue: #38 approved — submit pending env GO (plain python3 pytest format). #45 in review. #16 approved —
radix-sched submits. EMB #13/#14/#15/#18: radix-emb resubmits at GO.
