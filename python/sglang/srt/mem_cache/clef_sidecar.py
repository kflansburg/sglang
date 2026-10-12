"""Producer/retriever for CLEF joint-schema-head prefix sidecars on the radix cache.

A sidecar is the head's projected prefix evidence memory for a cached token
prefix, stored on the radix node alongside the backbone KV so independent CLEF
branches reuse it instead of reprojecting the shared state. This module owns:
identity construction (so head/weight/dtype changes invalidate sidecars),
projection of a prefix's final hidden states into head evidence memory, and the
attach/lookup calls that feed JointSchemaHead.forward_batch's prefix_memory.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from sglang.srt.layers.joint_schema_head import JointSchemaHead

# Build identity once per process: it must cover model revision, adapter and
# head weights, dtype / projection format, and mm config so a head change makes
# a stored sidecar stale even while the backbone KV stays valid. The token
# prefix itself is already the radix key, so identity covers everything else.
_identity_cache: Optional[str] = None


def sidecar_identity() -> str:
    """A stable identity string for what produced a sidecar.

    Reuses the server's resolved config (model revision, dtype) plus the head's
    weight signature. The token-prefix / mm parts of cache identity are already
    carried by the radix key (token ids + extra_key/cache_salt), so they are
    not repeated here.
    """
    global _identity_cache
    if _identity_cache is not None:
        return _identity_cache
    parts = []
    try:
        from sglang.srt.runtime_context import get_context

        ctx = get_context()
        for name in ("model_path", "revision", "dtype"):
            try:
                parts.append(f"{name}={ctx.config_leaf(name)}")
            except Exception:
                pass
    except Exception:
        pass
    _identity_cache = "|".join(parts) or "clef"
    return _identity_cache


def identity_with_head(head: JointSchemaHead) -> str:
    """Identity including a signature of the head's weights, so head changes
    invalidate sidecars even when everything else matches."""
    base = sidecar_identity()
    try:
        sig = 0
        for name, p in head.named_parameters():
            # Cheap, deterministic signature: shape + element count + dtype.
            sig = (sig * 1315423911) ^ (hash((tuple(p.shape), str(p.dtype))) & 0x7FFFFFFF)
        return f"{base}|head@{sig:x}"
    except Exception:
        return base


def project_prefix_memory(
    head: JointSchemaHead, prefix_hidden_states: torch.Tensor
) -> torch.Tensor:
    """Project a prefix's final hidden states into the head's evidence memory.

    Uses the head's own hidden_norm + memory_projection so the format exactly
    matches what forward_batch would have computed for those positions.
    """
    with torch.no_grad():
        normalized = head.hidden_norm(prefix_hidden_states)
        return head.memory_projection(normalized)


def attach_prefix_sidecar(
    tree_cache: Any,
    node: Any,
    head: JointSchemaHead,
    prefix_hidden_states: torch.Tensor,
    identity: Optional[str] = None,
) -> torch.Tensor:
    """Project ``prefix_hidden_states`` and attach as the node's sidecar.

    The projected tensor has one row per token of the prefix (1:1 with
    ``node``'s token range), as the storage layer asserts. Returns the memory.
    """
    memory = project_prefix_memory(head, prefix_hidden_states)
    tree_cache.attach_sidecar(node, memory, identity or identity_with_head(head))
    return memory


def lookup_prefix_sidecar(
    tree_cache: Any, node: Any, head: JointSchemaHead
) -> Optional[torch.Tensor]:
    """The cached projected prefix memory for ``node``, or None.

    None means a KV-only hit or an identity mismatch (stale head/weights/
    dtype); the caller then falls back to a complete uncached prefill rather
    than reusing incomplete or stale memory.
    """
    if node is None:
        return None
    return tree_cache.get_sidecar(node, identity_with_head(head))


# The scheduler does not hold the live head (it lives in the model runner's
# pooler), so the runner registers a zero-arg identity provider at startup and
# the scheduler resolves identity through it without importing the model.
_identity_provider = None


def register_identity_provider(provider) -> None:
    """Register the callable returning the head's sidecar identity.

    Called once by the model runner, which owns the JointSchemaHead instance.
    Until a provider is registered, sidecar lookup returns None, so every
    request takes the uncached path.
    """
    global _identity_provider
    _identity_provider = provider


def current_identity() -> Optional[str]:
    """The head's sidecar identity, or None if no provider is registered yet."""
    if _identity_provider is None:
        return None
    try:
        return _identity_provider()
    except Exception:
        return None


# The pooler owns the hidden states but not the radix cache, and the scheduler
# owns the cache but not the head. The model that constructs the head registers
# a resolver locating the tree cache, so the pooler's MISS path can attach by
# calling produce_sidecar without importing the scheduler.
_cache_resolver = None


def register_cache_resolver(resolver) -> None:
    """Register a zero-arg callable returning the RadixCache used for sidecars."""
    global _cache_resolver
    _cache_resolver = resolver


def produce_sidecar(
    head: JointSchemaHead,
    node: Any,
    prefix_hidden_states: torch.Tensor,
) -> bool:
    """MISS path: attach the projected prefix memory for ``node`` if possible.

    Called by the pooler after it has the prefix's final hidden states for a
    request whose matched prefix had no usable sidecar. Returns True when the
    sidecar was attached. Never raises: any failure degrades to the uncached
    path (no sidecar attached), matching the module's non-load-bearing rule.
    """
    try:
        if node is None or prefix_hidden_states is None:
            return False
        if _cache_resolver is None:
            return False
        tree_cache = _cache_resolver()
        if tree_cache is None or not hasattr(tree_cache, "attach_sidecar"):
            return False
        # A node may already carry a sidecar (attached by a concurrent branch);
        # do not overwrite it.
        if tree_cache.get_sidecar(node, current_identity() or "") is not None:
            return False
        attach_prefix_sidecar(
            tree_cache, node, head, prefix_hidden_states, current_identity()
        )
        return True
    except Exception:
        return False


def try_attach_decision_sidecar(req: Any, tree_cache: Any) -> None:
    """HIT path: if ``req`` matched a cached prefix that has a sidecar, carry
    that projected prefix memory onto ``req.decision_prefix_memory``.

    Only decisions (requests with a decision_layout) are considered. On a
    KV-only hit, an identity mismatch, or no registered head, the request is
    left with no sidecar so the pooler skips it and serving falls back to a
    complete uncached prefill. Never raises: a sidecar is never load-bearing.
    """
    try:
        if getattr(req, "decision_layout", None) is None:
            return
        if getattr(req, "prefix_len", 0) <= 0:
            return
        identity = current_identity()
        if identity is None:
            return
        node = getattr(req, "last_device_node", None) or getattr(req, "last_node", None)
        if node is None:
            return
        memory = tree_cache.get_sidecar(node, identity)
        if memory is not None:
            req.decision_prefix_memory = memory
    except Exception:
        try:
            req.decision_prefix_memory = None
        except Exception:
            pass
