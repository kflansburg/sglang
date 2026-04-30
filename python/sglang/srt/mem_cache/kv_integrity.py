import collections
import logging
import math
import os
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_VALID_MODES = {"off", "host", "device"}


class _NullTracker:

    enabled = False

    def on_alloc(self, req_pool_idx: int, slot_indices: Any) -> None:
        return None

    def on_prefix_hit(self, req_pool_idx: int, slot_indices: Any) -> None:
        return None

    def on_free(self, slot_indices: Any) -> None:
        return None

    def on_req_free(self, req_pool_idx: Optional[int]) -> None:
        return None

    def validate_batch(self, batch: Any) -> Sequence[Any]:
        return []


class KvIntegrityTracker:
    """Host-side per-page request-ownership tracker. See
    `docs/superpowers/specs/2026-04-30-sglang-kv-integrity-tagging-design.md`.
    """

    enabled = True

    def __init__(self, num_pages: int, page_size: int, req_pool_size: int):
        self.num_pages = num_pages
        self.page_size = page_size
        self.req_pool_size = req_pool_size
        self.owner_words = math.ceil(req_pool_size / 64)
        self.page_owners = np.zeros((num_pages, self.owner_words), dtype=np.uint64)
        self.req_pages: dict[int, set[int]] = {}
        self.transition_log: collections.deque = collections.deque(maxlen=10_000)

    def _pages_from_slots(self, slot_indices: Any) -> np.ndarray:
        if hasattr(slot_indices, "detach"):
            slot_indices = slot_indices.detach().cpu().numpy()
        slots = np.asarray(slot_indices, dtype=np.int64).ravel()
        if slots.size == 0:
            return np.empty(0, dtype=np.int64)
        return np.unique(slots // self.page_size)

    def _bit_for(self, req_pool_idx: int) -> tuple[int, np.uint64]:
        word_idx = req_pool_idx // 64
        bit = np.uint64(1) << np.uint64(req_pool_idx % 64)
        return word_idx, bit

    def _record(self, kind: str, page: int, req_pool_idx: Optional[int]) -> None:
        self.transition_log.append((kind, page, req_pool_idx))

    def on_alloc(self, req_pool_idx: int, slot_indices: Any) -> None:
        pages = self._pages_from_slots(slot_indices)
        if pages.size == 0:
            return
        word_idx, bit = self._bit_for(req_pool_idx)
        self.page_owners[pages, word_idx] |= bit
        bucket = self.req_pages.setdefault(req_pool_idx, set())
        for p in pages.tolist():
            bucket.add(p)
            self._record("ALLOC", p, req_pool_idx)

    def on_free(self, slot_indices: Any) -> None:
        pages = self._pages_from_slots(slot_indices)
        if pages.size == 0:
            return
        self.page_owners[pages, :] = 0
        for p in pages.tolist():
            self._record("FREE", p, None)


def make_tracker(num_pages: int, page_size: int, req_pool_size: int):
    mode = os.environ.get("SGLANG_KV_INTEGRITY", "off").lower()
    if mode == "off":
        return _NullTracker()
    if mode == "host":
        return KvIntegrityTracker(
            num_pages=num_pages,
            page_size=page_size,
            req_pool_size=req_pool_size,
        )
    if mode in _VALID_MODES:
        logger.warning(
            "SGLANG_KV_INTEGRITY=%s is reserved but not implemented; falling back to off.",
            mode,
        )
        return _NullTracker()
    logger.warning(
        "SGLANG_KV_INTEGRITY=%s is not a recognized mode (expected off|host|device); "
        "falling back to off.",
        mode,
    )
    return _NullTracker()
