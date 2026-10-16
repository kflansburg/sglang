"""Adversarial probes for allocator/mamba.py::MambaSlotAllocator (real code, CPU).

Features routed mamba dup-free mechanics to the pools team (pools-lead #356).

Findings demonstrated here (all caller-contract violations — the allocator is a
trusting free list with no dedup or debug checks):

 * MAMBA-1 (hardened upstream after our probe landed): freeing the same slot
   twice previously hands it to TWO future requests (slot appears twice in
   free_slots; two alloc(1) calls return the same id) — one live slot shared
   by two requests = cross-request mamba-state corruption. The probe below now
   pins the HARDENING contract on dev: the always-on accounting assertion
   `available_size() <= size` must fire on the second free (fail-closed), and
   SGLANG_DEBUG_MEMORY_POOL mode additionally sweeps for duplicate slots.

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


def test_double_free_is_fail_closed_by_accounting_guard():
    """Hardened on dev: the second free of the same slot must trip the
    always-on accounting assertion (available_size() > size), closing the
    MAMBA-1 primitive (pre-hardening the slot was handed to two requests)."""
    m = MambaSlotAllocator(8, "cpu")
    s = m.alloc(1)
    m.free(s)
    with pytest.raises(AssertionError, match="free-slot accounting overflow"):
        m.free(s)  # caller bug: same slot freed twice


def test_debug_mode_sweeps_duplicate_slots():
    """DEBUG sweep also rejects duplicate slots when debug_mode is enabled.
    Full-coverage pin for the SGLANG_DEBUG_MEMORY_POOL route: whichever
    contract fires first, a duplicate free must raise, never create a dup
    free-list entry."""
    m = MambaSlotAllocator(8, "cpu")
    m.debug_mode = True
    s = m.alloc(1)
    m.free(s)
    with pytest.raises(AssertionError):
        m.free(s)  # accounting guard and/or dup sweep fails closed
    assert m.available_size() <= m.size


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
