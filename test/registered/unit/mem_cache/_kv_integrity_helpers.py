"""Test fixtures shared across kv_integrity unit tests.

These helpers operate on `KvIntegrityTracker` instances directly without
spinning up an SGLang engine. They are deliberately decoupled from any
torch tensor / GPU concern: tests pass plain numpy arrays or python lists.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np


def build_tracker(num_pages: int = 1024, page_size: int = 64, req_pool_size: int = 128):
    from sglang.srt.mem_cache.kv_integrity import KvIntegrityTracker

    return KvIntegrityTracker(
        num_pages=num_pages,
        page_size=page_size,
        req_pool_size=req_pool_size,
    )


def slot_tensor(slots: Iterable[int]) -> np.ndarray:
    return np.asarray(list(slots), dtype=np.int64)


def slots_for_pages(pages: Iterable[int], page_size: int) -> np.ndarray:
    out: List[int] = []
    for p in pages:
        out.extend(range(p * page_size, (p + 1) * page_size))
    return np.asarray(out, dtype=np.int64)


def owners_of(tracker, page: int) -> set[int]:
    word_count = tracker.page_owners.shape[1]
    bits = set()
    for word in range(word_count):
        w = int(tracker.page_owners[page, word])
        for b in range(64):
            if (w >> b) & 1:
                bits.add(word * 64 + b)
    return bits


def assert_owners(tracker, page: int, expected: set[int]) -> None:
    actual = owners_of(tracker, page)
    if actual != expected:
        raise AssertionError(
            f"page {page} owners mismatch: expected={sorted(expected)} "
            f"actual={sorted(actual)}"
        )


def assert_no_owner(tracker, page: int) -> None:
    assert_owners(tracker, page, set())
