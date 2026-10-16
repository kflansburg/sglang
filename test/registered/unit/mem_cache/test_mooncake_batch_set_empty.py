# SPDX-License-Identifier: Apache-2.0
"""Regression: MooncakeStore.batch_set must not issue an empty batch_put_from
when every key already exists in the store.

Bug (audit P2, LOW): the raw batch_set API filtered out pre-existing keys but
then called _put_batch_zero_copy_impl unconditionally, so an all-hit batch
produced batch_put_from([], [], []) against the native client — a wasted RPC
and a potential crash vector there. batch_set_v1 already had the guard; the
raw path was missing it.
"""

import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeExistAwareStore:
    """Fake MooncakeDistributedStore that records put RPCs."""

    instances = []

    def __init__(self):
        self.existing_keys = set()
        self.batch_put_calls = []
        type(self).instances.append(self)

    def setup(self, *args, **kwargs):
        return 0

    def register_buffer(self, *args, **kwargs):
        return 0

    def put(self, key, value, *args):
        self._warmup_value = value
        return 0

    def get(self, key, *args):
        return getattr(self, "_warmup_value", None)

    def is_exist(self, key):
        return 1

    def batch_is_exist(self, keys, *args):
        return [1 if key in self.existing_keys else 0 for key in keys]

    def batch_put_from(self, keys, ptrs, sizes, *args):
        self.batch_put_calls.append(list(keys))
        self.existing_keys.update(keys)
        return [0] * len(keys)


def _fake_mooncake_modules():
    mooncake = types.ModuleType("mooncake")
    mooncake_store = types.ModuleType("mooncake.store")
    mooncake_store.MooncakeDistributedStore = FakeExistAwareStore

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


def _make_store():
    extra_config = {
        "master_server_address": "127.0.0.1:50051",
        "check_server": False,
        "global_segment_size": 1024 * 1024,
    }
    config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=None,
        tp_lcm_size=None,
        should_split_heads=False,
        extra_config=extra_config,
    )
    with patch.dict(
        "sys.modules", {**_fake_mooncake_modules(), **_fake_host_pool_modules()}
    ):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        store = MooncakeStore(config)
    return store, FakeExistAwareStore.instances[-1]


class FakeHostKVCache:
    def __init__(self):
        # Minimal surface used by register_mem_pool_host / gb_per_page.
        self.kv_buffer = torch.empty((1,), dtype=torch.uint8)
        self.layout = "page_first"
        self.page_size = 1

    def get_ksize_per_token(self):
        return 1

    def get_page_buffer_meta(self, indices):
        return [1000] * len(indices), [1024] * len(indices)


class TestMooncakeBatchSetEmpty(CustomTestCase):
    def test_all_existing_keys_skip_put_rpc(self):
        store, fake = _make_store()
        store.register_mem_pool_host(FakeHostKVCache())
        fake.existing_keys.update(["k0", "k1"])

        result = store.batch_set(
            ["k0", "k1"], target_locations=[1000, 2000], target_sizes=[1024, 1024]
        )
        self.assertTrue(result)
        self.assertEqual(fake.batch_put_calls, [])

    def test_missing_keys_still_put(self):
        store, fake = _make_store()
        store.register_mem_pool_host(FakeHostKVCache())
        fake.existing_keys.update(["k0"])

        result = store.batch_set(
            ["k0", "k1"], target_locations=[1000, 2000], target_sizes=[1024, 1024]
        )
        self.assertTrue(result)
        self.assertEqual(fake.batch_put_calls, [["k1"]])


if __name__ == "__main__":
    unittest.main()
