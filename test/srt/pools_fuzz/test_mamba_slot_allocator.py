"""MambaSlotAllocator double-free hardening (routed from features C1).

Caller contract: a slot is freed at most once; a duplicated slot in the free
list is later handed to two requests sharing one mamba state tensor. The
hardening: always-on integer accounting bound, debug-mode unique sweep.
"""
import os
import torch
import pytest

from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.utils import get_bool_env_var


def test_alloc_group_roundtrip():
    a = MambaSlotAllocator(size=4, device="cpu")
    a.alloc_group_begin(3)
    s1, s2 = a.alloc(1), a.alloc(1)
    a.alloc_group_end()
    assert s1 is not None and s2 is not None
    a.free(s1)
    a.free(s2)
    assert a.available_size() == 4


def test_double_free_overflow_assert():
    # Small pool: the second free of the same slot over-credits accounting
    # beyond capacity; the always-on integer assert trips (no sync, prod-safe).
    a = MambaSlotAllocator(size=2, device="cpu")
    x = a.alloc(1)
    a.free(x.clone())
    with pytest.raises(AssertionError, match="overflow"):
        a.free(x.clone())


def test_double_free_debug_unique_check():
    # On a pool large enough that one dup stays under capacity, only the
    # debug-mode unique sweep catches the duplicated slot. (debug_mode is
    # read from SGLANG_DEBUG_MEMORY_POOL at construction; set directly here
    # to stay hermetic under pytest's env propagation.)
    a = MambaSlotAllocator(size=8, device="cpu")
    a.debug_mode = True
    y = a.alloc(1)  # hold one slot out so the dup-ed free stays within accounting
    x = a.alloc(1)
    a.free(x.clone())
    with pytest.raises(AssertionError, match="double free"):
        a.free(x.clone())
    # Note: freeing y now WOULD trip the accounting assert — the corrupted list
    # already sits at capacity (that catch is also correct, and is the
    # production-visible symptom of an un-repaired double free).


def test_double_free_consequence_documented():
    """Without any guarding, the duplicated slot IS handed out twice (the
    pre-hardening behavior / the still-live reachability question)."""
    a = MambaSlotAllocator(size=4, device="cpu")
    a.debug_mode = False
    x = a.alloc(1)
    a.free(x.clone())
    # drop the second free directly into the list, bypassing the accounting
    # assert (models a double-free that stays under the size bound):
    a.free_slots = torch.cat((a.free_slots, x.clone()))
    slots = []
    while (s := a.alloc(1)) is not None:
        slots.append(int(s.item()))
    assert len(set(slots)) < len(slots), "duplicated slot handed to two requests"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
