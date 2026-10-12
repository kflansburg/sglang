"""Joint schema head memory for prompt prefixes reused from the radix cache.

A Clef request whose prompt prefix hits the radix cache skips that prefix in its
forward, but the joint schema head still attends over the prefix's projected
evidence memory. The memory row of a token depends only on the tokens up to it,
so the rows a request computes serve every later prompt that shares a prefix
with it. This store keeps those rows on the device under a byte budget and
caps each decision request's radix match at the longest prefix it holds rows
for, so a cache hit never lacks memory.
"""

from __future__ import annotations

import hashlib
from array import array
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.layers.joint_schema_head import parse_decision_layout
from sglang.srt.runtime_context import get_memory, mamba_checkpoint_grid


class DecisionPrefixMemory:
    """Projected memory rows of computed prompts, found by any prefix of them.

    Rows are indexed at every multiple of ``grid`` tokens, the depths at which
    the hybrid radix cache keeps recurrent state, so a match can only end where
    rows are indexed. Whole row tensors are dropped least recently used first.
    """

    def __init__(self, capacity_bytes: int, grid: int) -> None:
        assert capacity_bytes > 0 and grid > 0
        self.capacity_bytes = capacity_bytes
        self.grid = grid
        self.used_bytes = 0
        self._index: Dict[bytes, int] = {}
        self._entries: OrderedDict[int, Tuple[torch.Tensor, List[bytes]]] = (
            OrderedDict()
        )
        self._next_id = 0

    def _digests(
        self, salt: bytes, token_ids: Sequence[int], lengths: List[int]
    ) -> List[bytes]:
        data = _token_bytes(token_ids)
        hasher = hashlib.blake2b(salt, digest_size=16)
        digests, done = [], 0
        for length in lengths:
            hasher.update(data[done * 8 : length * 8])
            done = length
            digests.append(hasher.copy().digest())
        return digests

    def longest_prefix(
        self, salt: bytes, token_ids: Sequence[int], limit: int
    ) -> Tuple[int, Optional[torch.Tensor]]:
        """The longest held prefix of at most ``limit`` tokens, and its rows."""
        lengths = list(range(self.grid, limit + 1, self.grid))
        for length, digest in reversed(
            list(zip(lengths, self._digests(salt, token_ids, lengths)))
        ):
            entry = self._index.get(digest)
            if entry is not None:
                self._entries.move_to_end(entry)
                return length, self._entries[entry][0][:length]
        return 0, None

    def store(self, salt: bytes, token_ids: Sequence[int], rows: torch.Tensor) -> None:
        """Keep the rows of a computed prompt, one per token of ``token_ids``."""
        lengths = list(range(self.grid, rows.shape[0] + 1, self.grid))
        digests = [
            digest
            for digest in self._digests(salt, token_ids, lengths)
            if digest not in self._index
        ]
        size = rows.numel() * rows.element_size()
        if not digests or size > self.capacity_bytes:
            return
        while self.used_bytes + size > self.capacity_bytes:
            _, (evicted, evicted_digests) = self._entries.popitem(last=False)
            self.used_bytes -= evicted.numel() * evicted.element_size()
            for digest in evicted_digests:
                del self._index[digest]
        entry = self._next_id
        self._next_id += 1
        self._entries[entry] = (rows, digests)
        for digest in digests:
            self._index[digest] = entry
        self.used_bytes += size

    def clear(self) -> None:
        self._index.clear()
        self._entries.clear()
        self.used_bytes = 0


def create_decision_prefix_memory(
    *, model_config: Any, tree_cache: Any
) -> Optional[DecisionPrefixMemory]:
    capacity_mb = get_memory().joint_schema_prefix_memory_mb
    if model_config.joint_head_config is None or capacity_mb <= 0 or tree_cache.disable:
        return None
    return DecisionPrefixMemory(
        capacity_mb << 20, mamba_checkpoint_grid(tree_cache.page_size)
    )


def cache_salt(extra_key: Optional[str], salt: Optional[str]) -> bytes:
    """What besides the tokens keys the radix tree, so prefixes that must not
    share KV do not share memory."""
    return f"{extra_key or ''}\0{salt or ''}".encode()


def first_span_start(layout: List[int], num_tokens: int) -> int:
    """The first prompt position the joint schema head reads span means from."""
    return min(
        min(question.question_span[0], question.option_spans[0][0])
        for question in parse_decision_layout(layout, num_tokens)
    )


def decision_prefix_limit(
    layout: List[int],
    token_ids: Sequence[int],
    limit: int,
    salt: bytes,
    memory: Optional[DecisionPrefixMemory],
) -> Tuple[int, Optional[torch.Tensor]]:
    """How much of a decision prompt may come from the radix cache, and the
    memory rows of that much.

    The match must end before the first question or option span, whose hidden
    states the head reads, and within the prefix that ``memory`` holds rows for.
    """
    limit = min(limit, first_span_start(layout, len(token_ids)))
    if memory is None or limit <= 0:
        return 0, None
    return memory.longest_prefix(salt, token_ids, limit)


def _token_bytes(token_ids: Sequence[int]) -> bytes:
    if isinstance(token_ids, array) and token_ids.typecode == "q":
        return token_ids.tobytes()
    return array("q", token_ids).tobytes()
