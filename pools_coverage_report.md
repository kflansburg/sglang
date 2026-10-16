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
| 1 | Paged alloc_decode `assert page_size > 1` (+ kernel-naive equivalence sweep incl. part-3 reachability proofs) | latent host/kernel free-list desync at ps==1 -> double-alloc | MR#20 queued, approved | all; trigger inactive in prod |
| 2 | MambaSlotAllocator.free always-on accounting assert + debug unique sweep | caller double-free -> mamba slot to two live requests -> cross-request state corruption | MR#20 queued, approved | all versions (class latent; caller gate open) |
| 3 | int32 bound guards on DCP-widened virtual id spaces (configurator logical capacity + DSA index_buf_size) | latent id-space wrap -> silent cross-request KV aliasing | MR#20 queued, approved | v0.5.17+ (DCP absent pre-.17 in this shape) |
| 4 | dsv4 transfer-index helpers raise on negative scheduler-local ids | defense-in-depth crash robustness (disagg-traced: wire-free) | MR#20 queued, approved | HEAD-only code line |
| 5 | non-V4 hisparse pure-surplus guard (alloc-auditor #56; shared V4/non-V4 helper `_free_surplus_device_pages`) | broken device-index density -> frees page still referenced by kept tokens | in review (dsa-auditor) | vulnerable pattern in all tags w/ the non-V4 path (v0.5.14+; alloc-auditor verified vs raw tags) |
| 6 | F-5 hisparse prefix-clobber (CONFIRMED cross-request compressed KV corruption): zero/free only fresh post-prefix slots | in-flight: alloc-auditor implements, dsa-auditor reviews | --- | hisparse+V4 radix-hit deployments |

### Team-side fix merged before mine
| F0 | unified sub-pool stale-free hardening: Triton kernel store guards (GPU+CPU refs), _filter_stale_frees (debug raise / async tripwire), idle-verifier value checks (dup/OOB holes, live<0), no-host-sync contract preserved | caller-free-path stale (double/never-allocated) free -> raw OOB write below p2v table (GPU) or torch wrap tombstone of live page (CPU); silent pre-patch (proven: holes -1, live=-4, verifier []) | board #40, approved by pools-lead; MR#17 queued by unified-auditor | v0.5.15.post1, v0.5.17 (multi_ended_allocator precursor w/ identical plain-scatter mechanism), v0.5.20, HEAD; absent v0.5.14 |

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

## Open gates / follow-ups

1. **radix-sched: double-free/duplicate-page caller reachability at HEAD** (abort, retract, disagg admission abort, grammar) — gates grading F0/F1 as prod-bug vs pure contract-defense. Shipped hardening is zero-risk either way (no hot-path change in default mode).
2. **hisparse Q1/F-5 PD severity** (dsa + disagg) — decides cross-request severity in PD deployments; fix lands regardless.
3. **#47** DSA tail-buffer budget bias (GBs, OOM-direction) — unified board item.
4. GPU-side test for the Triton kernel store guards (CPU reference paths pinned; defer to full-CI).
5. Env queue rebatch after GO: MR#20 (my 5-fix bundle), MR#17 (unified F0), plus F-5 when landed.

## Deployment-config confirmations requested

- Whether any prod fleet currently enables hisparse V4 with radix cache (F-5 exposure); DeepSeek-V4 DSA trtllm deployments assert against hisparse (features-lead compiling).
- Confirm non-NPU DSA V4 pools in .17/.20 use page 64 (DSV4 compressed pool requires swa page == 256, loud-fails otherwise — verified).
