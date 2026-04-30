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


def fake_batch_with_pool(
    reqs,
    req_to_token,
    seq_lens,
    *,
    page_size: int,
    max_context: int = 128,
    req_pool_size: int = 64,
):
    """Build a batch fixture exposing the same surface validate_batch needs:
    `batch.reqs`, `batch.req_to_token_pool.req_to_token`, `batch.seq_lens_cpu`.

    Args:
        reqs: list of (rid, req_pool_idx) pairs.
        req_to_token: list of (req_pool_idx, slot_array) populating
            req_to_token[idx, :len(slots)]. Other rows stay zero.
        seq_lens: list of int, one per req in `reqs`. Determines the
            slice req_to_token[idx, :seq_len] that validate_batch reads.

    Other rows of req_to_token are left as zero — slot 0 maps to page 0,
    so callers that want validation to silently pass on row idx must also
    record page 0 as owned by that idx.
    """
    from types import SimpleNamespace

    import torch

    rt = torch.zeros((req_pool_size, max_context), dtype=torch.int64)
    for idx, slots in req_to_token:
        slots_t = torch.as_tensor(slots, dtype=torch.int64)
        rt[idx, : slots_t.numel()] = slots_t
    pool = SimpleNamespace(req_to_token=rt)
    reqs_ns = [SimpleNamespace(rid=rid, req_pool_idx=idx) for rid, idx in reqs]
    return SimpleNamespace(
        reqs=reqs_ns,
        req_to_token_pool=pool,
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int64),
    )
