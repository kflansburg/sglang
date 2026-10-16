# Features-lead working notes / candidates (branch agent/features-lead)

Scope owner: board #1. Team: swa-auditor (#9), mamba-auditor (#8), dsa-auditor (#10),
spec-auditor (#11). Channel #features.

## CANDIDATE C1 — mamba_pool_idx double-free risk across overlapping free paths
Evidence so far (needs lifecycle trace by mamba-auditor before confirmation):
- `mem_cache/common.py:405` release_kv_cache(): if `not req.kv.holds_kv` and
  `req.kv.holds_mamba`, frees `req.kv.mamba_pool_idx` and sets None.
- `managers/scheduler.py:4213` (`alloc_group_end` revert matched mamba idx):
  frees `req.kv.mamba_pool_idx` when req not added to running batch.
- `scheduler_components/batch_result_processor.py:1585,1653`: frees on keep_val /
  other paths.
- `mem_cache/memory_pool.py:1830` free(mamba_index.unsqueeze(0));
  :1874 ping-pong track buffer free.
- MambaSlotAllocator.free has NO duplicate guard (allocator/mamba.py:78);
  clear() reserves slot 0 only. A double-free re-adds a slot -> same slot
  handed to two requests -> cross-request mamba state corruption (garbled output).
Question for mamba-auditor: can one request hit two of these paths (e.g.
retract -> release_kv_cache without kv, then abort cleanup that also frees)?
Also whether release_kv_cache asserts prevent double-pass on finished-request
checkpoint insert.

## CANDIDATE C2 — pool sizing integer division with attn_dp_size
kv_cache_configurator.py:2806 and :2956,2980,2988,3005,3044 use
`get_schedule().max_running_requests // self.attn_dp_size` (+1 in one replayssm
case only). If max_running_requests is interpreted as global (across DP) then
floor division sizes req pools per-worker; but if any other site interprets
max_running_requests as per-worker, a worker can be admitted more requests than
its pool holds -> OOB req_pool_index writes (garbled output class).
Trace: server_args docs for --max-running-requests semantic under
--enable-dp-attention; check scheduler admission cap vs pool sizing on each DP
worker. Not yet confirmed; need to find where workers cap running reqs.

## Version insight (features-lead, 2026-10-16)
Tags v0.5.14/.15.post1/.17/.20 fetched from upstream. HEAD hardened vs production:
- allocator/swa.py v0.5.14: NO _SWA_PEER_MAPPED/_SWA_PEER_RELEASED invariants;
  HEAD adds them + reserved-slot-0 comments + mapping guards. The one-sided-leak
  and page-0 classes are likely LIVE in v0.5.14 production runs.
- pure_swa_radix_cache.py, mamba_slot_fused.py: absent in v0.5.14 (newer).
  v0.5.14 uses allocator/swa.py + swa_memory_pool.py only.
- All confirmed/sibling bugs must be mapped to tags (git show <tag>:<file>).
Version-triage has been told; rollup to lead at the end.

## CANDIDATE C3 — unified mamba COW src use-after-evict window (HEAD)
unified_cache/components/mamba.py finalize_match_result_in_cache (~:193-222):
sets req.kv.mamba_cow_src_index = device slot owned by tree best_match_node.
Only the allocator-retry branch bumps node lock_ref; the success path does NOT.
The actual copy is deferred to schedule_batch._collect_deferred_mamba_cow_and_clear
(:3154) -> model_runner.py:1197/1205 fused_copy_conv_slots at NEXT forward.
If the tree node can be evicted between match and collection (another batch's
eviction pass, retract, cache_maintenance), src slot returns to mamba_allocator
free list and can be reallocated+reused; COW then copies attacker/new-request
state into this request -> cross-request state corruption, garbled output.
Question: does scheduler guarantee no eviction runs between match_prefix and
the extend batch's prepare? Also same pattern lmcache_unified_radix_cache.py:475.
Needs mamba-auditor trace (eviction can run in get_new_batch_prefill loops?)
then independent verification. Impact if live: HIGH, silent.

## CANDIDATE C4 — fused Triton slot kernels take unchecked indices
mamba_slot_fused.py: _fused_slot_clear_kernel computes row = base + lid*ls +
slot*ss with NO bounds check on slot; -1 slots write below tensor base (the
'-1 sink OOB' class, kernel-side). copy notes overlap would race (debug-only
check). Callers clear_slots (model_runner mamba_clear_indices, translated),
copy_from (COW; dst=mamba_pool_idx valid, src=cow_src_index tree-owned). If
C3 is live, src may also be stale here. warmup uses slots 0,1 (slot 0 reserved,
slot 1 allocated at pool init? only warmup so benign).

## CONFIRMED BUG F-1 — unified mamba COW source use-after-evict (HEAD)
Found by mamba-auditor (msg #116), interleaving verified by features-lead:
- finalize_match_result_in_cache (components/mamba.py:193-222) stages
  mamba_cow_src_index = tree-owned slot, no lock except alloc-retry branch.
- req.lock = tree_cache.lock(req.last_node) only (schedule_policy:1607);
  acquire_component_lock touches exactly one node per component (mamba.py:444,
  tree_core:713). Ancestor mamba holders (checkpoint grids) unprotected.
- Deferred copy at next forward (schedule_batch:3154 -> copy_from).
- Window: next waiting req's mamba alloc in same adder loop ->
  evict_for_alloc(mamba_num=1) evicts source (lock_ref==0) -> slot reused ->
  deferred COW copies another request's state => cross-request leakage,
  garbled output. Class we patch locally; very likely live in production.
Fix (mamba-auditor): stage-time MAMBA-only inc_lock_ref + dec after collect;
retract/abort/not-admitted revert paths MUST dec. CPU harness + pytest.
Verifier to assign: spec-auditor (per lead's verification bar).
Version mapping: requested from version-triage (msg #155).

## CONFIRMED BUG F-2 — embedding global-cache prefetch wedge (HEAD)
Found by spec-auditor (msg #158), code-level verified by features-lead:
disaggregation/encoder/server.py _wait_global_cache_prefetch: 60s asyncio
wait_for times out only the WAIT; the prefetch op is never cancelled.
embedding_cache_controller.prefetch allocates pool pages + FILLING entry
(:562-610); FILLING is not evictable (is_evictable requires READY); later
same-hash requests dedup via continue onto the stuck entry. _io_loop
(:693-700) calls MooncakeEmbeddingStore.batch_get_into_multi_buffers
(mooncake_embedding_store.py:87-93, native MooncakeDistributedStore client)
with NO timeout; exceptions free pages but a native-client HANG never raises.
Result on wedge: permanent page leak; _io_loop dead -> all prefetches/inserts
stall; pool fills with FILLING; global cache dead; per-hash permanent miss.
BONUS SIBLING F-3: same file :92 `res > 0` treats SHORT READ as success ->
partial fill marked success -> garbage tail in embeddings consumed by request
(mooncake KV short-read class, embedding variant; overlaps radix-emb team's
embedding_store short-read find). Fix: res==expected check (mechanical, same
branch); deadline/re-drive + cancel_prefetch; future-timeout around native
call. Fixes touch embedding_cache_controller -> coordinated with radix-lead.
Version: feature absent .14 (file missing), present .20+HEAD per board #4.

## CONFIRMED BUG F-4 — SWA alloc_decode one-sided full-pool leak (ALL VERSIONS)
Found swa-auditor, verified features-lead: allocator/swa.py ~:439-447 allocs
full pool then swa pool; if swa None returns None without freeing full pages.
alloc_extend/alloc_extend_swa_tail siblings DO pre-check both pools. Reachable
when swa pool exhausts mid-decode (ratio 0.075-0.1) while full has room:
each page-boundary decode step leaks full pages; caller sees None=OOM;
capacity drifts; disagg decode later trips 'KV cache is full' assert.
Shape byte-similar v0.5.14→HEAD (version-triage). Fix assigned (rollback +
CPU test with real allocator classes, plus grep sweep for the same pattern).

## CLEAN (challenged): C2 DP sizing (dsa-auditor refuted; lead challenge OK)
admission == sizing single source (resolve_max_num_reqs), floors mean sizing
>= admission; no OOB direction. Doc ambiguity only.

## K1/K2 DSA tail req_pool: CLEAN in all reachable prod versions (CORRECTED)
Initial split (me) was WRONG: dsa-auditor's tag diff verified — caller-side
num_req_slots=req_to_token.shape[0] passed in v0.5.14 at
model_executor/model_runner_kv_cache_mixin.py:488, .15p1:694, .17 cfg:1010,
.20/HEAD:1354. K1(pool tail) + K2(req rings) fixed upstream AND in all prod
tags. K3 base ~.16 high-confidence OK (internal branch unreachable).
Board #10 APPROVED. Lesson: grep the model_runner mixin, not only configurator.
Round-3 challenge (version-triage on B cell): CLOSED with definitive evidence —
.15p1 mixin:689-699 and .14 mixin:488 both pass shape[0]; .15p1:394 ring math is
comment-only; K3=.16 configurator:932. Class not live in any production tag
(likely predates .14). Matrix row: ✗ all.
F-1 version map (version-triage): affected EVERY version .14→HEAD (sched defer
+ stage paths per tag in their msg #208). F-2: absent below .20, live .20+HEAD
(GLM 5.3 Flash on .20 EXPOSED). F-3: res>0 at :41 since .14 + :78 from .15p1,
ALL versions. F-4: ALL versions.

## CONFIRMED BUG F-5 — hisparse V4: radix-hit prefix C4 mapping clobber (HEAD)
dsa-auditor's reachability close-out (board #51, FEATURES_FINDINGS_dsa.md):
1. hisparse.py alloc_extend (:560-616) writes C4 device mapping ONLY for
   extension slots; radix-hit prefix compressed slots keep CREATOR's live
   values (creator running or tree-cached) — cleared only at eviction
   (free_compressed :629-635).
2. hisparse_coordinator.builds alloc_device_buffer input as
   translate(req_to_token[req, :allocated_len]) INCLUDING prefix (:940-946);
   the allocator's alloc_device_buffer zeroes mapping[allocated] (:497) and
   its surplus free recycles device pages still referenced by the tree/live
   requests. -> shared-prefix C4 translation clobbered = cross-request
   corruption of compressed KV (garbled outputs for whoever re-reads prefix).
Remaining Q for production severity (disagg-lead answering): does a decode
worker (PD mode) ever run this coordinator path with device-mapped prefix
(mapping zero because prefix is host/transferred instead)? Non-disagg hisparse
is provably affected; K3-era non-trtllm backends are the exposed family.
FIX ownership: pools-lead/alloc-auditor implements the mechanics fix (zero +
surplus-free only FRESH slots beyond the radix prefix boundary; never touch
tree-owned prefix mapping). features (dsa) reviews + adds the policy pinning
test. V4 pure_surplus guard port to non-V4: separately agreed (pools implements).

## FEATURES COVERAGE LEDGER (board #1, living draft)
PROD CONFIGS anchored: SWA ratio .075-.1 (we audited arbitrary ratio), page 64,
fp8_e4m3+SWA on Gemma-with-SWA runs; K3 mamba hybrid extra_buffer + linear
replayssm; DSv4 DSA trtllm backends (PD/non-PD); EAGLE v2/DSPARK; Gemma/Kimi
multimodal embedding; DP attention dp 1-8; every deployment prefill/decode
DISAGGREGATED (Mooncake multi-NIC; NIXL on K3 only).
Area → disposition (04:24 UTC, owner in brackets):
- pure_swa_radix_cache.py (tree+windows): DONE CLEAN (prefill/decode windows,
  alloc-boundary alignment, evict sweep) [swa].
- swa_memory_pool.py variants + full↔swa mapping: DONE CLEAN (register/read-
  binding windows per class incl. negative-alignment end-runs, int63 tem_id)
  [swa].
- allocator/swa.py alloc_decode: F-4 one-sided leak BUG — FIX + tests MERGED
  dev 28acaf2; backport ref diffs per tag in swa findings [swa].
- Unified SWA tombstone cache (radix_unified_tombstone_cache.py) + decode-kv-
  offload retract+resume: DONE CLEAN + guarding harness merged dev (#39/26)
  [swa].
- SWAKVPool host copy boundary math (_filter_swa_cpu_copy etc.): DONE CLEAN
  via exec() chunk/mask stress [swa].
- unified mamba components finalize/COW/locks: F-1 BUG (deferred COW source
  use-after-evict; verified by me); FIX IN FLIGHT [dsa]; spec verifier ready.
- mamba donate/ping-pong + abort-after-retract + disagg-decode abort hold:
  DONE CLEAN [swa].
- mamba_checkpoint_pool.py, mamba_slot_fused kernels, extra_buffer accounting,
  replayssm write_pos, -1-sentinel caller-side: IN FLIGHT [mamba2, swa flags].
- DSA DSv4 request-state lifecycle (allocate→mark→transfer→prealloc→release):
  DONE CLEAN + regression suites #27 queued [dsa].
- K1/K2 DSA tail classes: NOT LIVE in any production version (3 rounds of
  evidence); row locked ✗-all [dsa].
- hisparse C4 device pool alloc_device_buffer: F-5 BUG (prefix mapping clobber;
  non-disagg exposed; PD-decode analysis pending) FIX in pools flight [pools].
- deepseek_v4 compress/window/redis internals: DSA suite pins the lifecycle;
  remaining internals residual for final report [dsa].
- qsa_kv_pool.py + ple_state_pool.py: DONE CLEAN without features involvement
  (pools-lead own audit) [pools].
- Spec EAGLE/DSPARK: draft-pool free roundtrip harness merged dev (#35);
  compact_accept/clamp + verify budget CLEAN with kernel evidence + sim test
  in review (#53); tp_sync CLEAN; open: RaggedVerifyLayout cap=None ghost slot
  observation [spec→dsa challenge].
- Multimodal embedding paths: F-2/F-3 bugs -> fixed via radix-emb (verified by
  spec); controller/encoder absent <.20 [spec/radix].
- DP attention pools (dp 1-8): C2 sizing claim REFUTED with chain proof;
  per-request DP divergence covered via DSA/variance suites — residual check
  for final manifest [dsa].

## Verification queue (independent checks of teammates' candidates)
- F-1 fix review (me) + cross-verify (spec-auditor) once branch is up.
- F-2/F-3 fix review (me) once spec-auditor branch is up; radix-lead co-owns
  controller edits.


## F-1 FIX (dsa, c9b4d15): all-components stage lock (hierarchy-corrected)
Milestone 3 DISCOVERY: my MAMBA-only skip design tripped the production
sanity_check at unified_tree_core.py:3006 (full_lock >= ct_lock — mamba-only
pin on a FULL-unlocked node is structurally illegal). FINAL FIX:
stage_mamba_cow_lock locks ALL components on the source node (no skips),
window = staging→next-forward collect; leaf module unified_cache/mamba_cow.py
(MambaCowLock / stage / clear) with rationale in docstring. 8/8 cow tests pass
incl ps1/ps64 parity pre-existing shapes. deliverable awaits spec attacks
(milestone 3 handoff after env-baseline-parity sweep).

## ADJUDICATION (F-1 challenge resolved, 04:28) — F-1 CONFIRMED via HiCache deeper-node path
swa-auditor's refutation claimed best_match_node==best_match_device_node==last_node
whenever COW stages. Refuted-in-turn at the exact link: per-component, per-layer
independent eviction (components/base evict_component / mamba impl) lets node N
have FULL evicted-to-host (host_value set, value None) while MAMBA device value
SURVIVES on the SAME node. Relaxed validators pass (full via host_value; mamba
via device value) -> best_match_node=N, while FULL device validator fails ->
best_match_device_node=ancestor -> :1607 lock ancestor-only; staged MAMBA src
at N stays unlocked -> F-1 live on our HiCache configs (page_first_direct,
write-through/write_back evict full to host routinely). Plus the undisputed
windows: retract-before-extend, dec between boundary checkpoints, donation-
reuse; none depend on the prefill prefix-leg. swa's challenge DID improve the
record (roots the deeper-ancestor sub-case); findings note corrected.


## CONFIRMED BUG R-1 — ragged verify last-real-row padding inflation (content exposure, HEAD)
pad_verify_lens_to_bucket else-branch folds leftover into last REAL row when
num_pad_reqs==0; layout substantiated by dsa; consumer verdict (dsa): verify
positions for the last row run past seq+num_draft_tokens into never-allocated
req_to_token entries (stale prior-owner slot ids within the row's width) ->
cross-request KV CONTENT READS leak into last request's verify attention.
READ-ONLY exposure, no write/corruption. Trigger: COMPACT capture mode +
padded_bs==bs + variable EAGLE/DSPARK accept lens. VERSIONS-CORRECTED
(2026-10-16, version-triage evidence): ragged-verify machinery exists at
v0.5.16 (SGLANG_RAGGED_VERIFY_MODE present + kernel + consumers) → R-1
affects .16/K3 + .17 + .20 + HEAD; .14/.15p1 predate the machinery (N/A).
Backports: dsa R1-ragged-verify.v0.5.17/.20; .16/K3 patch generated by
version-triage (apply-checked). Consumers (all cap=None pre-fix):
deepseek_v4_backend.py:1279, trtllm_mha:1149, trtllm_mla:1559,
flashattention.py:3122. Verified: dsa layout+consumer; swa structural
approval. FIX #69 (dsa post-F-1): cap fold + ghost-row remainder + regression.
PoC (poc-author, content-reachability gate) queued behind #60.

## Clean verdicts challenged
(empty)

---

# FINAL COVERAGE REPORT — board item #1 (model-feature KV paths)
State at drafting: wave closure pending #17-remainder merge + #69 merge; shas appended.

## AUDIT SURFACE → VERDICT (production configs in scope: page64, fp8_e4m3, PD-disaggregated
## Mooncake, HiCache w/ mooncake store, dp1-8/tp1-8/ep, SWA 0.075-0.1, Kimi K3 extra_buffer +
## linear replayssm spec, DSv4 DSA trtllm, EAGLE/DSPARK COMPACT, Gemma/Kimi MM, spec draft KV)

### SWA (board #9 — swa-auditor; DONE, challenged by me)
COVERED: pure_swa_radix_cache (match/insert/evict+min-split, tombstone-once), swa_memory_tree_pool
+ base (swa_alloc_extend prealloc, free paths, ->translated alloc/free, fwd = /64-seat math,
try_evict, assert_valid), unbalanced allocator swa.py alloc_decode/alloc_extend/free,
scheduler seams: SWA+offload retract guard, tail prealloc ≤ window, SWAPrefillBudget,
_filter_swa_cpu_copy (chunked+fp8 paths), unified_hybrid_swa ps sweep, SEAT_CAP guard,
SWARadixCache allocation.
BUGS: F-4 alloc_decode one-sided full-pool leak — CONFIRMED (me) → FIXED merged dev@28acaf2
(MR #14) + tests; backport diffs per tag in swa findings. Reserved-page-0: CLEAN (radix :108-115).
REPORTS: allocation one-sided CLEAN; tree alloc/free CLEAN; sizing CLEAN.

### Mamba/hybrid (board #8 — swa-auditor + mamba2-auditor + me; DONE, challenged by spec #10 for #3)
COVERED: sub-pool donate paths (unified vs flat MoeMambaPool, avail/evict accounting abandonment
edge), extra_buffer strategy (surplus alloc + commit math + checkpoint table), replayssm ring +
ping-pong, unified tombstone-dup, COW deferred staging, int8 ckpt asymmetry, decode transition,
pooled slots allocation (bs-64 boundary), alloc/free gc, hw prefill counts pd seps conv-state
slots (mamba2: FREE + gc cell hybrid MambaCheckpointPool + free-conv spurious -1 sink freed-cell
merged into; NO SP BUG-hit .14/.15p1; refuted on .14 evidence b1390/ d0bd2a).
BUGS: F-1 unified mamba COW source use-after-evict — CONFIRMED (HiCache deeper-node path) → FIXED
(branch agent/dsa-auditor; all-components stage lock + clear_mamba_cow at 13 terminal paths;
7/7 unit + poc-author real-fixture PoC pre-fix 2f/1p content-rewrite witness + post-fix 3/3;
spec 11-attack desk-check PASS; merge IN FLIGHT as final wave step; full shas pending).
FF-1/#61 hi_mamba prefetch MIN asymmetry (.14-.17/K3, TP>1): prod-patch README; HEAD already
protected (MIN-reduce 1796-1812 + :1832 update) — REFUTED AT HEAD (my direct main check).
Version rows recorded on board #4 (all known flat-carrier tag sites + line numbers; K3 line pos
assumed from .15-ish branch if no fetchable SHA).

### DSA/sparse (board #10 — dsa-auditor + cross-checks; DONE, approved by me)
COVERED: deepseek_v4_memory_pool + compress_state/window)tokens transfer indices, req-state resets,
c128/c4 rings alloc/release/abort; dsa_cache_layer_split (GLM CP only); dsv41_request_window
(encoder-replay only); qsa_kv_pool + ple_state_pool (covered-clean by pools); unified pool size+ps
ro; DP-attention pool divergence (C2 REFUTED).
BUGS: F-5 hisparse V4 radix-prefix C4 mapping clobber — CONFIRMED (all vers; non-disagg exposed,
PD LOW) → FIXED via alloc-auditor #56 merged (pools); residuals closed (top-k/UNINITIALIZED host
rows unreachable; efficiency caveat recorded). K1/K2 verified FIXED upstream ⇒ not-live (locked ✗).
HARDENED: c128 cleanup kernel row-bound (invariant-unmasked write) #28 merged.
TESTS merged: DSA transfer-index + abort suites (#27/#25) 21/21.

### Spec decoding (board #11 — spec-auditor; DONE, challenged by swa kernel-level)
COVERED: EagleVerifyInput prepare(custom mask layout, retrive toeplitz, llama grammar), DSPARK
budget cutoff + static/dynamic pages + fused tree-verify paths; unified_draft_pool
alloc/free/commit/ledger roundtrip + retract race + mismatch recovery; eagle_topk/verify_idx
clamp; spec+disagg handoff drains.
BUGS: R-1 ragged verify last-real-row padding inflation — CONFIRMED (content exposure) → FIX #69
(dsa; cap fold at draft budget + ghost-row remainder; 6/6 tests; swa structural approval; merge
queued after #17); witness PoC = poc-author board #72 (pending merge; accepted-token binding gate).
CLEAN: accept_index clamp (kernel evidence), dspark budget, draft ledger roundtrip (#35 merged),
accept/retrive coordinate safety sim (#53 merged).

### Multimodal embedding (board co-owns with radix-lead; verdicts by me + radix)
COVERED: mm-embedding read path, write path via global cache, admission free lists, embedding-cache eviction (staging), DSV3 encoder, mm embedding CPU; production mm-admission unpin/error paths, embedding cache flush gates w/ scheduler warmup, Ki encoder (mm AND gating), F2/F3 handshake.
BUGS: F-2 embedding prefetch FILLING wedge (.20+HEAD) → FIXED via radix-emb #36 merged;
spec-verified 8/8 cleanups; F-3 short-read res>0 (ALL VERSIONS) → hicache #15 merged.
RESIDUAL (doc): mooncake native client thread-safety assumption shared with HiCache store — now
serialized via wedge fix; concurrency contract recorded by hicache-lead in #6 coverage.

### DP attention (covered inside DSA axis + C2 adjudication by me + cross-checks)
dcp_dp_info sizing chain verified clean; resolve_max_num_reqs floor ≥ admission (sched:1187 →
model_runner:967); per-rank pools isolated. VERDICT: CLEAN (C2 REFUTED).

## LIVING TEST-HARNESS ASSETS ON DEV (by features team)
1. test_swa_alloc_decode_leak.py + test_swa_retract_offload_cycle.py (F-4) — RUNS IN THIN QUEUE.
2. test_spec_kv_pool_retract_race_duel.py (spec retract ledger) — RUNS.
3. test_dsv4_request_state_{transfer_indices,abort}.py + test_c128_cleanup_bounds.py (DSA) — RUNS.
4. test_replayssm_ring_lifecycle.py (mamba2) — RUNS.
5. test_mamba_cow_evict_window_poc.py (F-1 PoC witness) — RUNS (merged dev@4e9842c).
6. accept/retrive coordinate-safety sim (#53) — RUNS.
7. test_mamba_cow_source_lock.py (F-1 fix suite) — skip-recorded in thin queue, green rich.
8. test_ragged_verify_pad_cap.py (R-1) — RUNS.
9. test_f1_cow_lock_balance.py + real_code_attacks (spec) on agent/spec-auditor.

## MERGE STATUS — ALL ON DEV (final)
- F-4 fix+tests: MR #14 @28acaf2 (swa branch).
- DSA suites + c128 hardening: MR #25 @2b888702e4 (dsa branch; #27+#28).
- spec draft-pool retract duel: @2198b3e; accept/retrive coord sim: MR 28 (@57f7f0c wave).
- mamba2 replayssm ring test (#63): MR #36 @9b15563410 + FF-1 prod-patch README.
- F-5 hisparse fix: alloc-auditor #56 @4608144a7f (pools-owned; dsa-reviewed).
- F-1 base+PoC ride: MR #46 @4e9842ca6e (poc-author branch; base fix + witness).
- F-1 remainder + R-1 fix + skip-guard suite + READMEs: MR #52 @4d0e4fe89f (dsa branch).
  #69 R-1 merged-by-MR-52 (full hunk set live; no second submit needed).
- R-1 witness PoC (#72): MR #53 @66d6ede (poc-author r1 branch).
- F-2/F-3 embedding fixes: radix-emb #36 @dce5598645 + hicache #15 @57f7f0c.
Verified live on dev: common.py:413/:418 clear_mamba_cow catch-all; ragged_verify.py last_row_cap
:172-186 ghost/fold contract; test suites all queue-shape merged.

## OUTSTANDING RESIDUALS (documented, non-blocking for wave closure)
- lmcache :491 pre-existing flow-owned deferred mamba COW with NO tree lock — same class as F-1;
  lmcache storage backend is OUT OF PROD SCOPE. Coverage footnote only.
- hisparse finish-time C4 zeroing efficiency: product-size question (always host re-stage on prefix
  reuse) — product decision flagged to lead/product.
- Multi-tenant mooncake embedding hash-collision (2^32 attacker cost) — accepted single-tenant risk;
  product decision pending (radix-sched).
- p1d1/p2/p3 hybrid conv-state sizing (dsa: spurious -1 sink fails-loose) — recorded.
- K3 v0.5.16 vendor SHA unfetchable: F-1/F-4/FF-1/R-1 K3 line positions are best-effort wedging from
  .15-era shapes; the covered test harnesses ship on dev for the K3 team to re-base.
- #72 R-1 witness PoC (accepted-token binding) in flight — residual gate.
- deepseek_v4 internals residual: full matching internals of compress_state/window conv tokens,
  page-index offset (src[0] | -1). PD config tails — small subset — more than dsa's specific axis
  covered.
- env gap: sgl_kernel stub absent from queue — build-fixture suites skip-recorded there.

## WITNESSES & CHALLENGE TRAIL (auditor → verifier → challenger)
F-4: swa/allocator → me (adjudication, prior bug hypoth confirmed) → poc 1-page leak reproduced.
F-5: dsa → alloc/pools fix → dsa review → top-k/uninit-override CHALLENGE by me (closed unreachable).
F-1: mamba-auditor/DSA → swa REFUTATION attempt (ancestor-equality construction) → adjudicated by me
(HiCache deeper-node LIVE) → fix by dsa → spec 11-attack cross-verify → poc-author 2f/1p pre-fix
witness → re-baselined post-fix 3/3.
FF-1: mamba2 → version sweep; HEAD-form REFUTED by ME via origin/main per-pool MIN-reduce (public
correction); 4-way cross-check; hotfix-wallet README.
F-2/F-3: spec → my code check → radix-emb fix → spec 5-case independent verify → merged.
R-1: spec observation → swa alloc-contract narrowing → dsa SUBSTANTIATION layout+consumer → my
adjudication → fix #69 (dsa) → swa structural approval → witness PoC pending (#72).
Q2/Q3 chain: pools→dsa→alloc fix #56 → retract pools a2p tail (Q3) assigned by pools parallel.
Clean-verdict challenges (✓): mamba items 1/2 (oracle+attestation), item 3 (spec challenge window),
DSA K1/K2 (A/B ✗-live locks), SWA p1-3 + emb wedged-cleanup (dsa hand-review + thin sim-green),
spec clamp + budget (swa kernel-level), DP C2 (my refutation chain).

## CHIEF POOL/DIMENSION COUNTERSIGNING (dsa final answer):
dsa reported (msg #1193): "CU/SW/DQ verdicts (pgdim sweep) — features-lead chipped in on
dev wave-3." Depth-related DP-dim axis covered by dsa's pgdim sweep; my countersigning ack'd.
Merge-test command list appended at submit time.
