"""Deferred mamba copy-on-write (COW) source locking (F-1 fix).

A prefix match can stage a deferred COW: the source state lives in a tree
node's mamba slot until the next forward copies it into the request's own
slot. Without a lock the source node is evictable in the window and its slot
can be reallocated to another request's state — the deferred read would then
COPY ANOTHER REQUEST'S MAMBA STATE into this one (cross-request leakage).

The lock covers ALL components of the source node — not MAMBA alone. The
tree-core lock-hierarchy invariant (production sanity check,
unified_tree_core.py:3006) requires full_lock >= ct_lock on every node, so a
MAMBA-only pin is structurally illegal. Pinning the full node for the short
staging->collect window is conservative but correct; the tier's plain
prefix-LRU still evicts everything else.

Clear paths must all funnel through ``clear_mamba_cow`` so the source lock
cannot leak — a leaked lock pins the tree node permanently (unevictable).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType  # noqa: F401

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_radix_cache import TreeNode  # type: ignore


class MambaCowLock:
    """Receipt anchoring the lock taken on the COW source node."""

    __slots__ = ("tree_cache", "node", "receipt")

    def __init__(self, tree_cache, node, receipt) -> None:
        self.tree_cache = tree_cache
        self.node = node
        self.receipt = receipt

    def release(self) -> None:
        self.tree_cache.dec_lock_ref(self.node, self.receipt.to_dec_params())


def stage_mamba_cow_lock(tree_cache, node, req) -> None:
    """Pin the COW source node's mamba slot for the deferred-copy window.

    A fresh staging supersedes any stale receipt (e.g. a rematch before the
    previous window's collect ran).
    """
    prev: Optional[MambaCowLock] = req.kv.mamba_cow_lock
    if prev is not None:
        prev.release()
        req.kv.mamba_cow_lock = None
    # All components locked: full_lock >= ct_lock hierarchy (see module doc).
    receipt = tree_cache.inc_lock_ref(node)
    req.kv.mamba_cow_lock = MambaCowLock(tree_cache, node, receipt)


def clear_mamba_cow(req) -> None:
    """Release the COW source lock (if any) and drop the req-side deferred
    COW/clear metadata.

    Every non-forward consumer of ``mamba_cow_src_index`` must come through
    here (finish / retract / not-admitted revert / pool free / H2D reload /
    post-collect), so the source node cannot leak or hold a stale pin.
    """
    lock: Optional[MambaCowLock] = getattr(req.kv, "mamba_cow_lock", None)
    if lock is not None:
        lock.release()
        req.kv.mamba_cow_lock = None
    req.kv.mamba_cow_src_index = None
    req.kv.mamba_needs_clear = False
