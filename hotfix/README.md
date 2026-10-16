# Hotfix: V4 — decode deferred KV release (board #19, dev 8e731e894a)

Bug: device-destination deferred KV release freed GPU pages at the
`deferred_kv_release_timeout` (30s) deadline **without a prefill drain ack**
("releasing anyway"). A lost AbortNotification or a slow multi-chunk transfer
then lets a still-writing prefill RDMA write land in pages already reissued
to a new request — cross-request KV corruption. Host destinations already
hold forever; device destinations now use the same rule.

Fix (upstream, dev 8e731e894a): `resolve_deferred_releases` holds undrained
device releases **past the deadline** until
`kv_mgr.is_abort_release_safe(room, required_acks)`; the only early release
is when the prefill instance is declared down
(`kv_mgr.prefill_info_table` no longer contains the bootstrap addr — popped
by `_handle_node_failure`). The deadline field only rate-limits a re-log.

## Patches

| Tag | Commit | Status |
|---|---|---|
| v0.5.20 (prod F) | 94602c9c2b7c | **V4.v0.5.20.patch** — apply + py_compile verified fresh |
| v0.5.17 (prod E) | 294816854627 | **RESIDUAL — no deferred machinery on this tag** |
| v0.5.16 (prod K) | fdebc938f7f4 | **RESIDUAL — no deferred machinery on this tag** |
| v0.5.14 (prod A) | — | flagged at audit: no deferred machinery either |

### Why v0.5.16 / v0.5.17 / v0.5.14 get no patch
`grep -rn "deferred_kv_release\|_deferred_releases" python/sglang/srt/` returns
nothing on all three tags: the deferred-release framework
(`_defer_release`, `resolve_deferred_releases`, `is_abort_release_safe`,
drain-ack bookkeeping, `prefill_info_table` watch) **does not exist** there.
On those versions a failed transfer frees pages **immediately** (no drain
wait of any kind) — a larger production hole that a hotfix patch cannot
close without backporting the whole deferred framework (large feature
backport, out of hotfix scope). Recommend: treat as residual; fix by
upgrading those deployments to a version with the deferred machinery
(v0.5.20 + this patch, or the dev merge).

## Apply notes (v0.5.20)

    cd $SGLANG_REPO
    git checkout v0.5.20            # 94602c9c2b7c
    git apply --check /path/to/V4.v0.5.20.patch
    git apply /path/to/V4.v0.5.20.patch
    python3 -m py_compile python/sglang/srt/disaggregation/decode.py

Single file: `python/sglang/srt/disaggregation/decode.py` (43 lines changed).
No metrics_collector change needed on v0.5.20 (the
`observe_decode_deferred_kv_release` metric — and its outcome validation —
arrived on main after v0.5.20 was cut; the patch does not emit metric
outcomes incompatible with v0.5.20).

Behavior change on v0.5.20 after the patch:
- "timed out … releasing anyway" free **can no longer happen**.
- A failed room whose prefill is alive now stays in `_deferred_releases`
  until the drain ack completes; a re-log fires every 30s.
- If the prefill instance goes down (heartbeat/failure detector pops it from
  `prefill_info_table`), the room releases with the down-peer log line.
- Net effect: no free of pages a remote RDMA engine may still be writing;
  at worst a request is held until its prefill dies (then released).

Regression tests (dev): `test_deferred_release_timeout_hold.py` holds /
drain / peer-dead / rate-limited-relog cases — the dev versions live in the
main tree; backport of the test file is optional (it depends on the later
`observe_decode_deferred_kv_release` metric).
