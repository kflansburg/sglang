# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Transfer-index helper hardening (dsv4 F2, defense-in-depth): corrupt local
bookkeeping (negative req_pool_idx / seq_len) must raise, not silently alias
another request's ring rows. Rows must always stay inside the caller's ring."""
import numpy as np
import pytest

from sglang.srt.mem_cache.deepseek_v4_compress_state import (
    c4_state_transfer_indices,
    request_scoped_state_transfer_indices,
)


@pytest.mark.parametrize("ring_size", [8, 12, 16])
def test_c4_rows_stay_in_own_ring(ring_size):
    for req in (0, 1, 7):
        for seq in (0, 1, 3, 4, 5, 20):
            rows = c4_state_transfer_indices(req, seq, ring_size=ring_size)
            assert rows.size >= 0
            if rows.size:
                assert (rows >= req * ring_size).all()
                assert (rows < (req + 1) * ring_size).all()


def test_c4_negative_ids_raise():
    with pytest.raises(ValueError, match="negative req_pool_idx"):
        c4_state_transfer_indices(-1, 8, ring_size=8)
    with pytest.raises(ValueError, match="negative seq_len"):
        c4_state_transfer_indices(0, -1, ring_size=8)


def test_c4_wrap_would_alias():
    # Pre-hardening: req=-1, ring=8, seq=8 -> rows [-8,-7,-6,-5] I.E. another
    # request's rows (or garbage below the buffer); the negative-id raise is
    # the whole point (test_c4_negative_ids_raise). Here pin exact row math of
    # the legal tail-with-wrap case to show the in-ring wrap is INTENTIONAL.
    rows = c4_state_transfer_indices(1, 9, ring_size=8)
    # seq 9: state tail len 9%4+4 = 5 -> positions 4..8 mod 8 = 4,5,6,7,0
    assert rows.tolist() == [12, 13, 14, 15, 8]


def test_request_scoped_negative_ids_raise():
    with pytest.raises(ValueError, match="negative req_pool_idx"):
        request_scoped_state_transfer_indices(-1, 3, ratio=4, online=False, ring_size=16)
    with pytest.raises(ValueError, match="negative seq_len"):
        request_scoped_state_transfer_indices(2, -3, ratio=4, online=True, ring_size=16)


def test_request_scoped_boundaries_unchanged():
    # block boundary -> empty
    assert request_scoped_state_transfer_indices(0, 8, ratio=4, online=False, ring_size=16).size == 0
    assert request_scoped_state_transfer_indices(0, 0, ratio=4, online=True, ring_size=16).size == 0
    # online -> the single row
    assert request_scoped_state_transfer_indices(3, 6, ratio=4, online=True, ring_size=16).tolist() == [3]
    # offline: page = req * pages_per_req + ((seq-1) % ring) // ratio
    got = request_scoped_state_transfer_indices(2, 6, ratio=4, online=False, ring_size=16)
    assert got.tolist() == [2 * 4 + (6 - 1) % 16 // 4]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
