# RETRACT-OFFLOAD-DRAIN — CLASS-5 hotfix (board #4)

Bug: in all production tags the decode-side KV offload path has NO drain guard.
`retract_decode()` frees device KV rows while a queued/in-flight D2H offload
still reads them; re-admitted rows are handed to new prefill-KV RDMA writes
racing the stale reads -> wrong-pair KV persisted to L2/L3 under correct hashes
-> cross-request context leakage. Evidence: `git grep drain_before_retraction
<tag HEAD>` == 0 on all four tags; audit/hicache_decode_offload_findings.md on
agent/decode-offload-audit-fix (the "Versions" section).

Main fixed the UAF with `drain_before_retraction` but that drain raised an
uncaught RuntimeError on a 30-s wedged copy -> scheduler SIGQUIT (class B1,
board #52 -- merged on dev as fd9845e229, plus the drain + catch/refuse).
PROD PATCH THEREFORE CARRIES BOTH CONCEPTS: drain-before-retract AND
call-site catch/refuse (V8+V8.1 analogues). Bare drain without the catch is a
crasher trade, not a fix.

## What each patch adds (3 files, ~90 lines, additive only)

1. `python/sglang/srt/environ.py`: `SGLANG_DECODE_OFFLOAD_DRAIN_TIMEOUT_S
   = EnvFloat(30.0)` (overridable; 0.05 in tests).
2. `decode_kvcache_offload_manager.py`: `drain_before_retraction()`
   VERBATIM from dev — poll check_offload_progress, tp-group MAX-reduce of
   "any pending" (ack_write_queue || ongoing_offload || offload_inflight);
   bounded wait; raise RuntimeError on timeout; 1 ms poll.
3. `scheduler.py`:
   - update_running_batch OOM-retract path: try/catch the drain; on
     RuntimeError set `batch.batch_is_full = False` and `return batch`.
     Tag consumer does not accept a None running_batch (main returns None; tag
     consumer at get_next_batch_to_run dereferences immediately) — keeping the
     batch IS the deferral: retraction is skipped this iteration, in-flight
     pages stay allocated, next loop retries the drain.
   - pause_generation: drain before any batch mutation; on RuntimeError restore
     `_engine_paused = False`, `logger.critical("PAUSE REFUSED ...")`, return.
     Engine keeps serving; operator may retry.

## Apply notes

Per tag (clean checkout; prefix/line drift has been verified per tag):

    git apply --check hotfix/RETRACT-OFFLOAD-DRAIN.v0.5.14.patch       # @ 49e384ce9d30 — VERIFIED 05:57Z
    git apply --check hotfix/RETRACT-OFFLOAD-DRAIN.v0.5.15.post1.patch # @ 0b3bb0cbe318 — VERIFIED 05:57Z
    git apply --check hotfix/RETRACT-OFFLOAD-DRAIN.v0.5.17.patch       # @ 294816854627 — VERIFIED 05:57Z
    git apply --check hotfix/RETRACT-OFFLOAD-DRAIN.v0.5.20.patch       # @ 94602c9c2b7c — VERIFIED 05:57Z

All four are `git apply` (3-way not needed); syntax-checked post-apply
(ast.parse); functional smoke on v0.5.20 patched tree: drain happy path
returns immediately; wedged inflight + SGLANG_DECODE_OFFLOAD_DRAIN_TIMEOUT_S=0.05
raises RuntimeError as designed. Same anchors/semantics on all tags (differences
are local line offsets only; patches hand-diffed, not regexed-copy from main).

## K3 (v0.5.16-base 87dd4596/bd4828e2, Sha per triage fdebc938f7f4 — private branch)

Cannot fetch; no patch artifact shipped. Apply the v0.5.15.post1 patch with
`git apply --3way` against the K3 tree if the three anchor files match
patterns: (i) `decode_kvcache_offload_manager.py` — manager with
offload_inflight/ongoing_offload/cache_controller and NO
`drain_before_retraction` (0 hits = affected; version-triage verified at
fdebc938f7f4: manager exists, drain absent, retract flows call
release_req directly from scheduler); (ii) scheduler pause_generation exists
with `_engine_paused = True`; (iii) environ.py has EnvFloat. On any anchor
mismatch use the manual recipe in section 2 of this README (the four edits are
small and positional in the same functions). If in doubt, ask decode-offload-audit.

## Post-patch operational note

PAUSE REFUSED + deferred retraction appear in scheduler logs only on wedged
copies (>30s D2H); if you see them, the D2H path is unhealthy sooner than
retraction policy — page the decode fleet owner; do NOT raise the timeout
blindly.

## Sibling RESIDUAL (same class, not patched by this item)

`_stop_storage_threads` RuntimeError on flush_cache/cache-reset with in-flight
host->storage IO (>10 s) is still uncaught on tags (see instance-4 findings);
operator-facing only. Decide separately (triage row notes).
