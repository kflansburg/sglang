"""Regression: new_tokens_required_next_decode must not under-estimate paged
KV pages for requests one in-flight token from a page boundary under overlap
scheduling. Under-estimation made update_running_batch skip retraction and the
decode step then raised RuntimeError("Decode out of memory") once every
evictable page was gone (radix-sched audit, board item #16).
"""
from types import SimpleNamespace

import pytest

from sglang.srt.managers.schedule_batch import ScheduleBatch


class _Kv:
    def __init__(self, committed, allocated):
        self.kv_committed_len = committed
        self.kv_allocated_len = allocated


class _Allocator:
    def __init__(self, page_size):
        self.page_size = page_size


def _batch(enable_overlap: bool, reqs, page_size: int = 64) -> ScheduleBatch:
    b = ScheduleBatch.__new__(ScheduleBatch)
    for r in reqs:
        r.beam_group = None
    b.reqs = list(reqs)
    b.enable_overlap = enable_overlap
    b.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    b.token_to_kv_pool_allocator = _Allocator(page_size)
    return b


@pytest.mark.parametrize("enable_overlap", [True, False])
def test_non_boundary_request_needs_no_new_page(enable_overlap):
    # committed=10: far from boundary; no new page either way.
    b = _batch(enable_overlap, [SimpleNamespace(kv=_Kv(10, 64))])
    assert b.new_tokens_required_next_decode() == 0


def test_exact_boundary_request_needs_page():
    # committed=64: next token straddles into a fresh page.
    b = _batch(False, [SimpleNamespace(kv=_Kv(64, 64))])
    assert b.new_tokens_required_next_decode() == 64


def test_overlap_pending_token_at_boundary_needs_page():
    # Overlap: one token is in flight but not committed, so committed=63 while
    # seq_lens is already 64. prepare_for_decode allocates with seq_lens+1 = 65
    # -> a new page. The capacity check must see that page.
    b = _batch(True, [SimpleNamespace(kv=_Kv(63, 64))])
    assert b.new_tokens_required_next_decode() == 64


def test_overlap_pending_token_before_boundary_needs_no_page():
    b = _batch(True, [SimpleNamespace(kv=_Kv(62, 64))])
    assert b.new_tokens_required_next_decode() == 0


def test_mixed_batch_counts_each_straddler():
    reqs = [SimpleNamespace(kv=_Kv(63, 64)), SimpleNamespace(kv=_Kv(100, 128))]
    b = _batch(True, reqs)
    assert b.new_tokens_required_next_decode() == 64
