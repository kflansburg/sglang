"""Microbenchmark for KvIntegrityTracker per-step overhead.

Run manually:
    python test/registered/unit/mem_cache/bench_kv_integrity_overhead.py
"""

import time
from types import SimpleNamespace

import numpy as np

from sglang.srt.mem_cache.kv_integrity import KvIntegrityTracker


def main():
    page_size = 64
    num_pages = 200_000
    req_pool_size = 256
    batch_size = 32
    pages_per_req = 64

    tracker = KvIntegrityTracker(
        num_pages=num_pages, page_size=page_size, req_pool_size=req_pool_size
    )
    rng = np.random.default_rng(0)
    pages = rng.choice(num_pages, size=(batch_size, pages_per_req), replace=False)
    for i in range(batch_size):
        slots = pages[i, :, None] * page_size + np.arange(page_size)
        tracker.on_alloc(req_pool_idx=i, slot_indices=slots.ravel())

    batch = SimpleNamespace(
        reqs=[SimpleNamespace(rid=f"r{i}", req_pool_idx=i) for i in range(batch_size)]
    )

    n = 1000
    start = time.perf_counter()
    for _ in range(n):
        tracker.validate_batch(batch)
    elapsed = (time.perf_counter() - start) / n * 1e6
    print(
        f"validate_batch: {elapsed:.2f} µs/call (batch_size={batch_size}, pages_per_req={pages_per_req})"
    )


if __name__ == "__main__":
    main()
