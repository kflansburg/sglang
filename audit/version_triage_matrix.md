# Version-affected matrix — FINAL v2 (2026-10-16 06:08)
Authoritative live record: board item #4 notes. This snapshot freezes the review state for the endgame.
Versions: A=v0.5.14(49e384ce9d30), B=v0.5.15.post1(0b3bb0cbe318), K=v0.5.16(fdebc938f7f4; k3-PROXY, fork delta vendored), E=v0.5.17(294816854627), F=v0.5.20(94602c9c2b7c), G=HEAD(3420bae before review), dev=bed816b tip under endgame.
Every ✓ has a corresponding per-tag apply-verified backport patch in the hotfix omnibus EXCEPT where marked POLICY-RESIDUAL.
Cell symbols: ✓ = bug present in that tag and patch/verdict resolved below. ✗ = not present. POLICY = bug behavior exists but no machine-patch is shipped (deployment/policy resolution signed off).

| Bug | A | B | K | E | F | G | fix on dev | hotfix patches |
|---|---|---|---|---|---|---|---|---|
| Mooncake KV short-read (N1/N3) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (fixed e766366a→#33 5050af76) | #33 | MC-SHORTREAD ×5 hicache-lead |
| Embedding store short-read (N4) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (#14 chain) | radix-emb | N4 ×5 radix-emb |
| Deferred decode KV release + abort-notif 30s-hold (CLASS-1/V4 family) | ✓ wider | ✓ wider | ✓ wider | ✓ wider | ✓ compound | ✓→fixed (#19/#44/8e731e894a) | #19/#44 chain | V4 .v0.5.20 only; A/B/K/E = POLICY (no machinery; decision-table option) |
| Retract-vs-KV-offload drain+catch (CLASS-5/V8/B1) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓→fixed (fd9845e via #52) | #52 | RETRACT-OFFLOAD-DRAIN ×4 + CLASS-5.v0.5.16 (✓5) |
| RadixCache.checkpoint protected-len shrink (row A) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ fixed (#38) | #38 | 38 ×5 radix-lead |
| Unified checkpoint_into_tree shrink (#59) | ✗ | ✗ | ✗ | ✗ | ✗ (no checkpoint facility) | ✓ fixed (fcdd7d5) | #59 | n/a (HEAD-only) |
| SWA alloc_decode one-sided leak (F-4/D) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ fixed (#26/c0b53d4a lineage) | #26 | F-4 ×4(authors).v0.5.16 fill by version-triage |
| Mamba COW use-after-evict (F/F-1) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ unified-only fixed (#17/4d0e4fe8) | #17 | F1 ×5 dsa (+A needs new clear site — included) |
| Ragged-verify last-row fold (R-1) | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ fixed (#69/279eb8a1) + witness #72 | #69/#72 | R1 ×2(dsa) + .16 fill (version-triage) |
| HiSparse V4 radix-prefix clobber (F-5) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ fixed | #2/Q3 (2cd88ba6 alloc_pads sibling + pools fix) | F-5 ×5 alloc-auditor |
| Mamba-hi_mamba prefetch TP divergence (C4/FF-1) | ✓ | ✓ | ✓ | ✓ | ✗ (removed) | ✗ (folded MIN) | (implicit by unified) + canonical guidance | FF-1 ×3 (swa) + .16 fill; C4-MAMBA-MIN = alternative |
| Host-pool free guards (H-F1/H-F2) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ fixed (#25/13d4f7a2) | #25 | H-FREEGUARDS ×5 hicache-lead |
| NIXL no-reload invalidated peer (class-10) | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ (reload at HEAD) | — | (no separate patch ordered; V2/V3 cover) |
| NIXL send_aux no-validation (V2) / unbounded handle wait (V3) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ fixed (#34) | #34 | V2 ×5, V3 ×5 disagg-nixl |
| Mooncake truncate-on-mismatch (V5) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ fixed (#37/ac0f192) | #37 | V5 ×5 disagg-mooncake |
| batch_register return drops (V6) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ fixed (#37) | #37 | V6 ×5 |
| transfer_worker exception death (V7) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ fixed (#37) | #37 | V7 ×5 (same-file order: V5 then V7) |
| State-drift truncation cap (V12/F6/V9) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ fixed (#41/ac0f192) | #41 | F6 ×5 |
| Aux codec unchecked wire write (V11/F12) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ fixed (#42+#46/bed816b2) | #42/#46 | F12 ×5 |
| get_mha_kv_ptrs_with_pp draft-V mixup (F15/V10) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ fixed (#30/9fa4981e26) | #30 | F15 ×5 disagg-prefill |
| Encoder global-cache family L/#13/#15/#18/EMB36 | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ (F+G-only feature; fixes on dev dce5598 chain) | radix-emb boards #13/#15/#18/#36 | EMB ×4 (.F-only) |
| Embedding prefetch wedge (C1/F-spec-1) | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ fixed (#36) | #36 | (covered by EMB36/.F) |
| #16 embedding io-wedge (availability) | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ → fixed (#52 family) / implementing | — | n/a (availability-side) |

CLEAN/no-patch rows: DSA tail req_pool (class-11) ✗ ALL; H-F5 unified page-envelope CLEAN; SWA reserved page 0 ✗ guarded everywhere; dsv4-F2 verified DiD not-live.

RESIDUALS (signed):
- POLICY: class-1 pre-.20 — decision-table (keep local patch / .20 wholesale / accept risk). K3 + .14/.15p1/.17 fleets.
- K3 vendor fork: patches target public v0.5.16; fork-heavy files (mamba, nixl, common/conn.py) re-derive line refs on vendor checkout; C4-MAMBA-MIN.README.md carries the K3 shape-check walkthrough; prefill-common staging + F18 guidance where applicable.
- F18 register-first refusal: availability-side omnibus (not corruption); where DeferredBootstrap exists (.14+), fix exists on dev (bed816b2 F2/F18 bundle) but per-tag patch files NOT requested/parked.
- _stop_storage_threads uncaught on flush_cache: availability-side followup, unowned at freeze gate.
- R1 TRITON kernel fold remains uncapped under ALL hotfix patches (torch-path-only caps, mirrors dsa's own shape): K3/.16 GPU fleets patched with hotfix only get partial cover; full cover = backport merged HEAD kernel clamp (#69→dev carries it) or accept exposure. FINAL REPORT FLAG.
- .16 CLASS-5: delivered (author+proxy-verified) but module-level functional test blocked by pre-existing env clash (extracted-function execution only).
- C4-arbitration: FF-1 canonical; C4-MAMBA-MIN alternative do-not-apply-both (conditional-collective risk).
- K3 VERIFY-in-flight items: class-5 .v0.5.16 patch semantics run module-level only on extracted function; RADIX-38 .v0.5.16 side; F15 conn.py hunk drift-sensitive (author note).

HOTFIX OMNIBUS INDEX (branch → dir): MC-SHORTREAD+N4+H-FREEGUARDS+C4-alt @ hicache-lead & radix-emb; F1/R1 @ dsa-auditor; F-4/FF-1 @ swa-auditor; .16 fills @ version-triage; F-5 @ hotfix-f5 (alloc); 38 @ radix-lead; V2/V3 @ disagg-nixl; V5/V6/V7/F6/F12 @ disagg-mooncake; F15 @ disagg-prefill; V4 @ disagg-decode; CLASS-5 @ decode-offload-audit-hotfix; EMB @ radix-emb. Apply-order rules per READMEs; mooncake V5→V7 same-file order is the only intra-family order constraint.
