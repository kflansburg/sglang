# HOTFIX C4-MAMBA-MIN — mamba prefetch slot attach/free TP divergence (hi_mamba legacy path)

**Bug (board #61 / #6 C4-class; resolved at HEAD by the hybrid
_reduce_prefetch_ack per-pool MIN; live on the legacy tags):**
hi_mamba_radix_cache.check_prefetch_progress MIN-reduces `completed_tokens`
across TP but reads RANK-LOCAL `extra_pool_hit_pages[PoolName.MAMBA] >= 1`
("mamba_loaded") to attach or free the mamba host slot. On an asymmetric
sidecar read (one rank's engine returns fewer/errored pages — e.g. short
reads, transient mooncake errors), the hit rank attaches the slot and skips
the free while the miss rank frees the same slot and skips the attach →
per-rank mamba host free lists diverge permanently; the freed slot is
reissued while alive elsewhere → cross-request mamba host UAF / H→D garbage.

**Patch (A/B/K/E = v0.5.14/.15p1/.16/.17; F=v0.5.20 removed the file):**
after the rank-local mamba_loaded computation in check_prefetch_progress,
fold it into a TP MIN all_reduce on the same tp_group (identical collective
entry on every rank since the MAMBA-transfer presence is constructed
uniformly at prefetch submit). Every rank then shares one verdict: the slot
is attached only if EVERY rank loaded it; otherwise it is freed on every
rank. Mirrors the unified_radix_cache.py:2021-2033 packed-MIN analogue
(v0.5.15+ SWA path, which is guarded already) and the HEAD hybrid
_reduce_prefetch_ack guard.

**Consistency invariant:** the new all_reduce is entered iff
`mamba_host_indices is not None and tp_world_size > 1` — uniform across
ranks by the prefetch submission path (all ranks of the TP group build the
same pool_transfers for a given prefetch operation). Adding a collective
after an unconditional collective in the same method keeps stage lockstep.

**STATUS (omnibus arbitration, features-lead 06:03 UTC):** swa-auditor's
FF-1 variant is CANONICAL (unconditional fold into the existing tokens
all_reduce tensor); THESE patches are INDEXED-ALTERNATIVE — do NOT apply
both. Why the unconditional form won: this patch enters the mamba-loaded
all_reduce conditionally on `mamba_host_indices is not None` — safe under
the prefetch-submit uniformity invariant, but a hotfix whose job is curing
rank-divergence should not carry ANY rank-conditional collective entry:
under a latent transfer-uniformity violation the conditional variant would
deadlock the TP group. Semantics otherwise equivalent; this folder kept as
the design/reference note.

**Apply:** `git checkout <tag>; git apply C4-MAMBA-MIN.<tag>.patch`
Verified `git apply --check` against clean tags:
v0.5.14 (49e384ce9d30), v0.5.15.post1 (0b3bb0cbe318), v0.5.16 (fdebc938f7f4),
v0.5.17 (294816854627). Not needed on v0.5.20 (file removed) or HEAD
(hybrid override already folds).

**K3 note:** K3 ≈ v0.5.16-base internal fork — check its
hi_mamba_radix_cache.py matches (~:1866-1890); the patch anchors on the
verbatim mamba_host_indices/mamba_loaded loop shown in this README's bug
section and asserts exactly-one-match shape equivalence.
