"""
Copyright 2026 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Slot allocator for the Mamba state pool.

Mamba caches one whole state tensor per request, so the allocator hands out
fixed-size slots (1 per request) rather than paged token KV indices.  The
underlying tensor storage lives in ``MambaPool``; this class owns only the
free-slot bookkeeping.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch


class MambaSlotAllocator:
    """Free-list of Mamba pool slot indices. Deliberately not a subclass of
    ``BaseTokenToKVPoolAllocator``: slots are per request, not per token."""

    def __init__(self, size: int, device: str):
        self.size = size
        self.device = device
        # Set by alloc_group_begin(); alloc(1) drains it until alloc_group_end().
        self._alloc_iter: Optional[Iterator] = None
        from sglang.srt.utils import get_bool_env_var

        self.debug_mode = get_bool_env_var("SGLANG_DEBUG_MEMORY_POOL")
        self.clear()

    def available_size(self) -> int:
        return len(self.free_slots)

    def schedulable_available_size(self) -> int:
        """Planner-facing free count. Same as ``available_size`` for a static pool;
        byte-coordinated allocators return their byte-limited view instead."""
        return self.available_size()

    def alloc_group_begin(self, num_reqs: int):
        """Pre-allocate a batch of slots for match_prefix to amortize overhead."""
        self._alloc_iter = None
        if num_reqs > 0:
            result = self._do_alloc(num_reqs)
            if result is not None:
                self._alloc_iter = iter(result.split(1))

    def alloc_group_end(self):
        """Return any unused pre-allocated slots from the current group."""
        if self._alloc_iter is not None:
            remaining = list(self._alloc_iter)
            if remaining:
                self.free(torch.cat(remaining))
        self._alloc_iter = None

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if self._alloc_iter is not None and need_size == 1:
            slot = next(self._alloc_iter, None)
            if slot is not None:
                return slot
        return self._do_alloc(need_size)

    def _do_alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        self.free_slots = torch.cat((self.free_slots, free_index))
        # Caller contract: every slot is freed at most once. A duplicated slot
        # in the free list is later handed to two requests, which then share one
        # mamba state tensor (cross-request state corruption). The debug sweep
        # mirrors PagedTokenToKVPoolAllocator._debug_check_no_duplicate_pages;
        # the always-on bound below costs no sync (python ints).
        assert self.available_size() <= self.size, (
            f"mamba free-slot accounting overflow: {self.available_size()} > {self.size}"
        )
        if self.debug_mode:
            self._debug_check_no_duplicate_slots()

    def _debug_check_no_duplicate_slots(self):
        assert len(torch.unique(self.free_slots)) == len(self.free_slots), (
            "double free: duplicate slots in mamba free list"
        )

    def clear(self):
        # Slot 0 is reserved as a dummy write target for padded tokens.
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
