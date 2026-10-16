# Audit deliverables index (swarm 2026-10-16)

This tree collects the memory-safety audit's documentation for the production
production surface (disaggregated prefill/decode, Mooncake/NIXL, HiCache,
page-64 fp8 KV, SWA/mamba/DSA/spec model features). Contents:

## Findings & coverage docs (source branch in parentheses)
- `audit/version_triage_matrix.md`, `audit/version_triage_pass1.md` — version-affected matrix for all known bug classes vs prod tags (agent/version-triage @58849479)
- `audit/hicache_master_findings.md` (agent/hicache-lead), `audit/hicache_mooncake_findings.md` (agent/mooncake-audit), `audit/hicache_prefetch_findings.md`, `audit/cpu_test_env_recipe.md` (agent/prefetch-ctl-audit), `audit/hicache_decode_offload_findings.md` (dev/current, decode-offload team)
- `audit/radix_findings.md` (agent/radix-lead)
- `audit/FEATURES_FINDINGS_lead/dsa/mamba/swa/spec.md` (features team)
- `audit/disagg-findings-{seams,decode,prefill}.md` (disagg team)
- `audit/pools_findings_{allocplan,configurator,dsv4,kvpool}.md`, `audit/pools_coverage_report.md` (pools team)
- `audit/findings-emb.md` (agent/radix-emb)

## Hotfix patch kit — `hotfix/`
Per bug family × production tag (v0.5.14 / v0.5.15.post1 / v0.5.16 / v0.5.16K /
v0.5.17 / v0.5.20), each family with a README: V2..V7 mooncake/nixl/common-conn,
F12/F6/F15 codec/MHA, F-4 SWA alloc leak, F1-mamba-cow, FF-1 mamba-min asym,
F-5 hisparse pad, C4-MAMBA-MIN, H-FREEGUARDS host free guards, MC-SHORTREAD,
CLASS-5 retract-offload-drain, EMB13/15/18/36 + N4 embedding short-read,
38 lock_ref underflow guards, R1 ragged-verify, RETRACT-OFFLOAD-DRAIN.
Union of producer branches: decode-offload-audit-hotfix, disagg-{decode,mooncake,
nixl,prefill}, dsa-auditor, hicache-lead, hotfix-f5, radix-{emb,lead}, swa-auditor,
version-triage. See per-family README*.md files there for apply instructions.

## Coverage report
- `audit/AUDIT_COVERAGE_REPORT.md` — lead's final report (coverage matrix + verdicts).

## Exclusions
- `test/srt/pools_fuzz/README.md` (agent/kvpool-auditor) is intentionally NOT in
  this bundle: it documents the fuzz test suite itself (lives with the tests on
  dev), not an audit deliverable.
