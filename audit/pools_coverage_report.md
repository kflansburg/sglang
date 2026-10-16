# POOLS COVERAGE REPORT — core device memory pools + paged allocators

Owner: pools-lead. Scope: python/sglang/srt/mem_cache/ excluding items owned by
other leads (radix tree policy/abort-retract, hybrid_cache policy, HiCache host
pools/storage, spec paths, disaggregation wire logic — cross-references noted).

Production surface covered: page_size 64; kv-cache dtype fp8_e4m3/bf16;
chunked prefill 12288-32768; TP 1-8; DP attention; SWA; unified pools (v0.5.20+);
hybrid mamba + spec (K3); DSA/sparse DeepSeek V4 pools; KV sharding /
DCP-widened pools; QSA/PLE state pools; disaggregated PD runtimes.

## Delivered fixes (agent/pools-lead, reviewed, regression-tested, CPU)

| # | Fix | Class / consequence | Status | Versions affected |
|---|-----|---------------------|--------|-------------------|
| 1 | Paged alloc_decode `assert page_size > 1` (+ kernel-naive equivalence sweep incl. part-3 reachability proofs) | latent host/kernel free-list desync at ps==1 -> double-alloc | MERGED dev (MR#27 @57f7f0c9) | all; trigger inactive in prod |
| 2 | MambaSlotAllocator.free always-on accounting assert + debug unique sweep | caller double-free -> mamba slot to two live requests -> cross-request state corruption | MERGED dev (MR#27) | all versions (class latent; caller gate open) |
| 3 | int32 bound guards on DCP-widened virtual id spaces (configurator logical capacity + DSA index_buf_size) | latent id-space wrap -> silent cross-request KV aliasing | MERGED dev (MR#27) | v0.5.17+ (DCP absent pre-.17 in this shape) |
| 4 | dsv4 transfer-index helpers raise on negative scheduler-local ids | defense-in-depth crash robustness (disagg-traced: wire-free) | MERGED dev (MR#27) | HEAD-only code line |
| 5 | non-V4 hisparse pure-surplus guard (alloc-auditor #56; shared V4/non-V4 helper `_free_surplus_device_pages`) | broken device-index density -> frees page still referenced by kept tokens | in review (dsa-auditor) | vulnerable pattern in all tags w/ the non-V4 path (v0.5.14+; alloc-auditor verified vs raw tags) |
| 6 | F-5 hisparse prefix-clobber (CONFIRMED cross-request compressed KV corruption): prefix region read-only + fresh-tail zero/harvest + surplus guard protected_indices + coordinator compressed prefix_len | alloc-auditor #62, 5-case prefix-guard suite (object.__new__ real DeepSeekV4 pool, fp8, CPU); dsa approved | MERGED dev (MR#44 @4608144) + family close (MR#57 @2cd88ba) + hotfix-f5 per-tag patches verified | v0.5.14 (BOTH V4 non-V4) + Kimi K3 bases (87dd4596, bd4828e2); PD=LOW (per-worker schedulers; decode tree C4 zero→host fallback); non-disagg single-worker = exposure carrier |

### Team-side fix merged before mine
| F0 | unified sub-pool stale-free hardening: Triton kernel store guards (GPU+CPU refs), _filter_stale_frees (debug raise / async tripwire), idle-verifier value checks (dup/OOB holes, live<0), no-host-sync contract preserved | caller-free-path stale (double/never-allocated) free -> raw OOB write below p2v table (GPU) or torch wrap tombstone of live page (CPU); silent pre-patch (proven: holes -1, live=-4, verifier []) | board #40, approved by pools-lead; MERGED dev (MR#34 @dad8953f) | v0.5.15.post1, v0.5.17 (multi_ended_allocator precursor w/ identical plain-scatter mechanism), v0.5.20, HEAD; absent v0.5.14 |

## F-5 family completeness note
- THE F-5 FAMILY IS FULLY MERGED: #56+62→#44 @4608144 (Q2 surplus guard + F-5 alloc-side prefix-harvest), #68+#71→#57 @2cd88ba (Q3 pad guard + F-5b finish-side exclusion). the F-5 prefix-read-only buffer invariant intersects the pre-existing coordinator wholesale free (hisparse_coordinator.py:1418-1430 = free of read-only tree prefix ids) — same clobber class mirrored at free time. alloc caught the finish-side sibling during Q3; pools-lead independently verified; dsa formalized as board #71. Land order #35 -> #68(k) -> #71, no deploy between (lead told).

## Verified clean workstreams (with evidence)

- **Paged/token allocator mechanics** (alloc-auditor): free-list conservation, capacity exactness of available_size (incl. need_sort staged pages — my independent refutation of their BUG-ALLOC-1/2 hypothesis), decode/extend page-math upper bound + exact path equivalence (245-case kernel-vs-naive), 60x20k-step fuzz across 20+ configs; boundary convention (extra page, buffers size+page_size) verified intentional. CONSEQUENCE-WATCH: paged/two-path debug-only dedup trusts caller contract (debug catches; no prod caller in seam set — latent item documented).
- **memory_pool core/layout/kv_vmm** (kvpool-auditor, retired): shape/index math clean; fp8 scale-sizing class ABSENT in prod (fp8_e4m3 → Unquantized, no scale buffers); G1 latent noted: store sinks validate via probes only when SGLANG_ENABLE_ASYNC_ASSERT on.
- **Unified pools + configurator + alloc_plan** (unified-auditor + 3 subs): unified sub-pool/page mechanics fuzzed (model checker at ps=4/64, byte-accounting per step) + 120-test unified gate; sizing math read clean (2 LOWS: +1-page row pad unbudgeted ~0.006%; DSA tail-buffers unbudgeted ~GBs on tight configs = OOM-direction, board #47).
- **deepseek_v4 pool mechanics** (dsv4-pool-auditor, retired): F1 window_layout contiguity validation landed/merged (board #24); F2 (wire-trust question) refuted by disagg-lead trace (local ids); F3/F4 routed.
- **RADIX/sharded/DCP pools** (pools-lead): page_interleave placement math proved bijective; page_interleave_pool scratch epoch protocol + trash-page + capacity asserts sound; hisparse V4 pure_surplus logic verified; mamba checkpoint pool read clean; PLE/QSA pools read clean (constructor invest asserts verified).
- **Allocation/unified-SWA sub-pool hybrid paths** covered under unified-auditor's #12.

## Refutations w/ proof (search budget spent, negated with structure evidence)

- BUG-ALLOC-1/2 (hisparse/SWA capacity guard) — need_sort staged pages counted exactly (+repro with real classes).
- BUG-ALLOC-4 (alloc_extend_kernel part-3 unmasked load) — per-program early returns make num_part3>0 entrance; no dead-lane OOB on outputs.
- hisparse Q2 page-split accounting (non-V4) — exact UNDER the density invariant; guard ported to remove the assumption anyway.
- dsv4 F2 (wire-exposed transfer rows) — disagg-lead traced both call sites to local metadata; not a wire-trust hole.
- BUG-ALLOC-6 "garbage slots" — retracted by alloc-auditor (own caller-side misuse).

## Open gates / follow-ups (post-completion)

CLOSED during the audit:
1. F0/F1 caller reachability — radix-lead verdict: no by-construction at HEAD (single release gate + holds_kv assert + retract atomics, 400-op fuzz clean); pre-#38 checkpoint-shrink is the DOCUMENTED historical trigger. Shipped as defense-in-depth w/ historical trigger (radix #38 is the release-math family discovery).
2. F-5 PD severity — dsa verdict: PD=LOW (per-worker schedulers; decode tree C4 zero→host fallback); non-disagg single-worker = exposure carrier.
3. #47 DSA tail-buffer budget bias — SHIPPED (MR#37 @a523d1f).
3b. Q3 — SHIPPED (MR#57 @2cd88ba, dsa policy exactly as framed).
5. Env queue rebatch — all green under the final command.

OPEN follow-ups (documented for the team, none block coverage completeness):
- GPU-side test for the Triton kernel store guards (CPU reference paths pinned; queue env is CPU-only; full-CI would cover).
- copy_pages/zero_pages async-assert tripwires (unified's recommendation; unify the async-assert convention).
- F1 debug tripwire at free_group_end (only matters if callers ever relax the no-double-free-in-group contract).
- Radix-lead's #38 deep-history version matrix across all tags (their lane; feeds rest of the report).
- decode.py:1604 protected_len=total_prefix_len host-gap row inclusion — routed to disagg-lead (their #19 lane; bounded by unique()).
- features' R-1 (#66) verify-padding inflation: consumer-bound check in flight; pools takes allocator-side if violation confirmed.

## Process notes for wave-2+ merges
- Queue hygiene rule (lead-adopted #1335): merge origin/dev + re-run your suite before every submit_merge. Originated from my #35 diagnosis: alloc's tranche-2 mamba probe asserted the pre-#32 legacy semantics; clash surfaced only on the merged tree (their branch green / merged 123-1 fail). Wave-2+ branches rebase first.
- Probe tests must pin fail-CLOSED semantics once a hardening is in flight (never assert corruption success): the alloc probe was rewritten post-#32 to pin AssertionError on the second free + the debug sweep (message regex exact).

## Deployment-config confirmations requested

- Whether any prod fleet currently enables hisparse V4 with radix cache (F-5 exposure); DeepSeek-V4 DSA trtllm deployments assert against hisparse (features-lead compiling).
- Confirm non-NPU DSA V4 pools in .17/.20 use page 64 (DSV4 compressed pool requires swa page == 256, loud-fails otherwise — verified).
