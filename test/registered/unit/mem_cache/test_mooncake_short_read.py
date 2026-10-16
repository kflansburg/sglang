# SPDX-License-Identifier: Apache-2.0
"""Regression: Mooncake store must not treat short reads as success.

Bug: the get paths judged success only by sign of the returned byte count
(`res > 0` / `res >= 0` / `< 0`). A short (partial) read then reported
success while leaving the tail of the destination host buffer stale, so
partially-filled KV was served as valid cache -> garbled output or
cross-request data from the shared store.

Fix: a get is successful iff the engine returned EXACTLY the requested
number of bytes for every slice of the page.
"""

import types
import unittest
from typing import List
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


# ---- fake engine -----------------------------------------------------


class FakeShortReadStore:
    """Fake MooncakeDistributedStore whose get byte counts are scripted."""

    instances = []

    def __init__(self):
        self.existing_keys = set()
        # Either a scalar (all keys) or per-key list of get byte results.
        self.get_results = None
        self.get_multi_results = None
        type(self).instances.append(self)

    def setup(self, *args, **kwargs):
        return 0

    def register_buffer(self, *args, **kwargs):
        return 0

    def put(self, key, value, *args):
        self._warmup_value = value
        return 0

    def get(self, key, *args):
        # Engine-level get used by warmup(); returns stored bytes.
        return getattr(self, "_warmup_value", None)

    def is_exist(self, key):
        return 1

    def batch_is_exist(self, keys, *args):
        return [1 if key in self.existing_keys else 0 for key in keys]

    def batch_put_from(self, keys, ptrs, sizes, *args):
        self.existing_keys.update(keys)
        return [0] * len(keys)

    def batch_put_from_multi_buffers(self, keys, ptrs, sizes, *args):
        self.existing_keys.update(keys)
        return [0] * len(keys)

    def batch_get_into(self, keys, ptrs, sizes) -> List[int]:
        if self.get_results is not None:
            return list(self.get_results)
        return [int(s) for s in sizes]

    def batch_get_into_multi_buffers(self, keys, ptrs, sizes) -> List[int]:
        if self.get_multi_results is not None:
            return list(self.get_multi_results)
        return [sum(int(x) for x in s) for s in sizes]


def _fake_mooncake_modules(fake_store_cls):
    mooncake = types.ModuleType("mooncake")
    mooncake_store = types.ModuleType("mooncake.store")
    mooncake_store.MooncakeDistributedStore = fake_store_cls

    class ReplicateConfig:
        def __init__(self):
            self.group_ids = None

    mooncake_store.ReplicateConfig = ReplicateConfig
    return {"mooncake": mooncake, "mooncake.store": mooncake_store}


def _fake_host_pool_modules():
    pool_host = types.ModuleType("sglang.srt.mem_cache.pool_host")

    class HostKVCache:
        pass

    class HostTensorAllocator:
        pass

    pool_host.HostKVCache = HostKVCache
    pool_host.HostTensorAllocator = HostTensorAllocator

    pool_host_mla = types.ModuleType("sglang.srt.mem_cache.pool_host.mla")

    class MLATokenToKVPoolHost:
        pass

    pool_host_mla.MLATokenToKVPoolHost = MLATokenToKVPoolHost

    return {
        "sglang.srt.mem_cache.pool_host": pool_host,
        "sglang.srt.mem_cache.pool_host.mla": pool_host_mla,
    }


# ---- fake host pool --------------------------------------------------

PAGE_BYTES = 1024


class FakeHostKVCache:
    """MHA page_first host pool: per page, k and v slices of PAGE_BYTES."""

    def __init__(self):
        self.kv_buffer = torch.empty((1,), dtype=torch.uint8)
        self.layout = "page_first"
        self.page_size = 1

    def get_ksize_per_token(self):
        return 1

    def get_page_buffer_meta(self, indices):
        ptrs, sizes = [], []
        for i in range(len(indices)):
            ptrs.extend([1000 + i, 2000 + i])
            sizes.extend([PAGE_BYTES, PAGE_BYTES])
        return ptrs, sizes


class FakeMLAHostKVCache:
    """MLA layer_first-ish pool: one key per page, multi-buffer slices."""

    def __init__(self, slices_per_page=2):
        self.kv_buffer = torch.empty((1,), dtype=torch.uint8)
        self.layout = "page_first"
        self.page_size = 1
        self.slices_per_page = slices_per_page

    def get_ksize_per_token(self):
        return 1

    def get_page_buffer_meta(self, indices):
        ptrs, sizes = [], []
        for i in range(len(indices)):
            # Two slices per key make len(ptr) != len(keys) -> multi-buffer
            # packing kicks in (sum matters for the expected byte count).
            n = self.slices_per_page
            ptrs.append([3000 + i * 10 + j for j in range(n)])
            sizes.append([PAGE_BYTES // n] * n)
        return ptrs, sizes


# ---- store construction ----------------------------------------------


def _make_config(is_mla_model=False):
    extra_config = {
        "master_server_address": "127.0.0.1:50051",
        "check_server": False,
        "global_segment_size": 1024 * 1024,
        "enable_group_semantics": True,
    }
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=is_mla_model,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=None,
        tp_lcm_size=None,
        should_split_heads=False,
        extra_config=extra_config,
    )


def _make_store(is_mla_model=False):
    with patch.dict(
        "sys.modules",
        {
            **_fake_mooncake_modules(FakeShortReadStore),
            **_fake_host_pool_modules(),
        },
    ):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        store = MooncakeStore(_make_config(is_mla_model=is_mla_model))
    return store, FakeShortReadStore.instances[-1]


class TestMooncakeShortRead(CustomTestCase):
    def test_batch_get_v1_short_read_reported_as_miss(self):
        store, fake = _make_store()
        store.register_mem_pool_host(FakeHostKVCache())
        keys = ["p0", "p1", "p2"]
        indices = torch.arange(3)
        # k/v per key: make v-slice of p1 a 1-byte partial read.
        fake.get_results = [
            PAGE_BYTES,
            PAGE_BYTES,  # p0 ok
            PAGE_BYTES,
            1,  # p1 short
            -1,
            PAGE_BYTES,  # p2 error on k -> miss (pre-existing behavior)
        ]
        self.assertEqual(store.batch_get_v1(keys, indices), [True, False, False])

    def test_batch_get_v1_full_reads_all_hit(self):
        store, fake = _make_store()
        store.register_mem_pool_host(FakeHostKVCache())
        fake.get_results = [PAGE_BYTES, PAGE_BYTES] * 2
        self.assertEqual(store.batch_get_v1(["p0", "p1"], torch.arange(2)), [True, True])

    def test_batch_get_v1_mla_multi_buffer_short_total_is_miss(self):
        store, fake = _make_store(is_mla_model=True)
        store.register_mem_pool_host(FakeMLAHostKVCache())
        # Expected bytes per key: 2 * (PAGE_BYTES//2) = PAGE_BYTES.
        fake.get_multi_results = [PAGE_BYTES - 1]
        self.assertEqual(store.batch_get_v1(["p0"], torch.arange(1)), [False])
        fake.get_multi_results = [PAGE_BYTES]
        self.assertEqual(store.batch_get_v1(["p0"], torch.arange(1)), [True])

    def test_raw_batch_get_prefix_count_stops_at_short_read(self):
        store, fake = _make_store(is_mla_model=True)
        store.register_mem_pool_host(FakeMLAHostKVCache())
        # Raw v0 API: returns the successful prefix count.
        fake.get_results = [PAGE_BYTES, 7, PAGE_BYTES]
        sizes = [PAGE_BYTES] * 3
        self.assertEqual(
            store.batch_get(["p0", "p1", "p2"], [1, 2, 3], sizes), 1
        )

    def test_single_get_zero_byte_read_is_miss(self):
        store, fake = _make_store(is_mla_model=True)
        store.register_mem_pool_host(FakeMLAHostKVCache())
        fake.get_results = [0]
        self.assertFalse(store.get("p0", target_location=1, target_sizes=PAGE_BYTES))
        fake.get_results = [PAGE_BYTES]
        self.assertTrue(store.get("p0", target_location=1, target_sizes=PAGE_BYTES))

    def test_batch_get_v2_group_short_read_is_miss(self):
        store, fake = _make_store(is_mla_model=True)
        pool = FakeMLAHostKVCache(slices_per_page=1)
        store.register_mem_host_pool_v2(pool, PoolName.SWA)
        # Expected size per key is PAGE_BYTES (1 slice). This pool produces
        # multi-buffer metadata, so byte counts flow through the multi-buffer
        # engine path.
        fake.get_multi_results = [PAGE_BYTES, PAGE_BYTES - 100]
        transfers = [
            PoolTransfer(
                name=PoolName.SWA,
                keys=["p0", "p1"],
                host_indices=torch.arange(2),
            )
        ]
        results = store.batch_get_v2(transfers)
        self.assertEqual(results[PoolName.SWA], [True, False])


if __name__ == "__main__":
    unittest.main()
