import logging
import os
from typing import Any, Optional, Sequence

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
        raise NotImplementedError("KvIntegrityTracker body lands in Task 3")


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
