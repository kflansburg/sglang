"""Adversarial probes for allocator/mamba.py::MambaSlotAllocator (real code, CPU).

Features routed mamba dup-free mechanics to the pools team (pools-lead #356).

Findings demonstrated here (all caller-contract violations — the allocator is a
trusting free list with no dedup or debug checks):

 * MAMBA-1: freeing the same slot twice hands it to TWO future requests
   (slot appears twice in free_slots; two alloc(1) calls return the same id) —
   one live slot shared by two requests = cross-request mamba-state corruption.
   Sibling of the paged free-list double-free class, but unlike the paged debug
   mode there is NO SGLANG_DEBUG_MEMORY_POOL sweep here at all.

 * MAMBA-2: alloc_group_begin() while a previous group is still open silently
   disposes of the previous group's unconsumed pre-allocated slots (the
   iterator is just overwritten: self._alloc_iter = None). Those slots are
   stranded until clear(). Today the only caller (scheduler.get_new_batch_prefill)
   pairs begin/end on one path and scheduler exceptions are fatal, so no
   runtime exposure; defense: assert / drain leftover at begin.

Callers in-tree at HEAD do not exhibit either pattern.
"""

import collections

import pytest
import torch

from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator


def test_double_free_hands_slot_to_two_requests():
    m = MambaSlotAllocator(8, "cpu")
    s = m.alloc(1)
    m.free(s)
    m.free(s)  # caller bug: same slot freed twice
    assert m.free_slots.tolist().count(int(s.item())) == 2
    seen = []
    while True:
        r = m.alloc(1)
        if r is None:
            break
        seen.append(int(r.item()))
    dup = [k for k, v in collections.Counter(seen).items() if v > 1]
    assert dup, "slot must be handed to two owners after double free"


def test_group_free_of_unconsumed_remainder_is_clean():
    """The normal begin -> partial consume -> end cycle returns the remainder
    exactly once (no dup, no loss)."""
    m = MambaSlotAllocator(8, "cpu")
    m.alloc_group_begin(4)
    s1 = m.alloc(1)
    s2 = m.alloc(1)
    m.alloc_group_end()
    slots = m.free_slots.tolist()
    # the two consumed slots are held by callers; the rest returns exactly once
    assert sorted(slots) == [
        x for x in range(1, 9) if x not in (int(s1.item()), int(s2.item()))
    ], slots
    assert len(set(slots)) == len(slots)


def test_group_begin_during_open_group_leaks(DELTA_EXPECTED=True):
    """MAMBA-2: reopening a group without alloc_group_end strands the previous
    group's unconsumed slots (available_size never recovers short of clear())."""
    m = MambaSlotAllocator(8, "cpu")
    m.alloc_group_begin(3)  # carves 3
    m.alloc(1)  # consume 1 of them; 2 unconsumed
    m.alloc_group_begin(2)  # previous iterator dropped without freeing remainder
    assert m.available_size() == 3  # 8 - 3(first carve) - 2(second carve) = 3; the 2 stranded
    m.alloc_group_end()  # returns at most the new group's leftover
    assert m.available_size() == 5  # permanently missing 2 slots until clear()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
