"""Loader for the real SGLang allocator modules on a CPU-only box.

Injects lightweight fakes for the heavy deps (triton kernels, platform
detection) so the real base.py/token.py/paged.py/swa.py import from their
actual file locations and run on CPU tensors with need_sort=False.
"""
import importlib.util
import sys
import types
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "python/sglang/srt"


def _mk(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def load():
    # Fake sglang.kernels.ops.memory.allocator
    _mk(
        "sglang.kernels.ops.memory.allocator",
        alloc_decode_kernel=None,
        alloc_extend_kernel=None,
        get_and_clear_swa_pages=None,
    )

    class _Caps:
        supports_triton = False  # CPU: always take the naive path

    class _Platform:
        capabilities = _Caps()

        @staticmethod
        def get_paged_allocator_cls():
            return None  # use PagedTokenToKVPoolAllocator directly

    _mk("sglang.srt.platforms", current_platform=_Platform())
    _mk(
        "sglang.srt.utils",
        get_bool_env_var=lambda *a, **k: False,
        get_num_new_pages=None,  # filled below (real one needed)
        is_hip=lambda: False,
        next_power_of_2= None,
        is_npu=lambda: False,
    )

    def pkg_import(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    # Register the package chain so relative-ish absolute imports resolve.
    for pkg, p in [
        ("sglang", REPO / "python/sglang"),
        ("sglang.srt", ROOT),
        ("sglang.srt.mem_cache", ROOT / "mem_cache"),
        ("sglang.srt.mem_cache.allocator", ROOT / "mem_cache/allocator"),
        ("sglang.srt.utils", ROOT / "utils"),  # later replaced attrs
    ]:
        if pkg in sys.modules:
            continue
        m = types.ModuleType(pkg)
        m.__path__ = [str(p)]
        sys.modules[pkg] = m

    # Need the real get_num_new_pages / next_power_of_2 from utils/common.py
    # utils.common imports half the world; reimplement the two tiny helpers.
    sys.modules["sglang.srt.utils"].get_num_new_pages = _get_num_new_pages
    sys.modules["sglang.srt.utils"].next_power_of_2 = lambda x: 1 << (x - 1).bit_length()
    _mk(
        "sglang.srt.utils.common",
        get_num_new_pages=_get_num_new_pages,
    )

    # srt.platforms platforms file pulls torch only? fake already registered.

    base = pkg_import(
        "sglang.srt.mem_cache.allocator.base",
        ROOT / "mem_cache/allocator/base.py",
    )
    token = pkg_import(
        "sglang.srt.mem_cache.allocator.token",
        ROOT / "mem_cache/allocator/token.py",
    )
    paged = pkg_import(
        "sglang.srt.mem_cache.allocator.paged",
        ROOT / "mem_cache/allocator/paged.py",
    )
    # swa.py imports sglang.srt.mem_cache.base_swa_memory_pool; that file only
    # needs sglang.srt.mem_cache.memory_pool which is heavy -> fake BaseSWAKVPool
    import abc

    class BaseSWAKVPool:  # minimal stand-in
        swa_req_ring_size = None
        swa_uses_full_indices = False

        def register_mapping(self, m):
            self.full_to_swa_index_mapping = m

        def translate_loc_from_full_to_swa(self, kv_indices):
            return self.full_to_swa_index_mapping[kv_indices]

    _mk("sglang.srt.mem_cache.base_swa_memory_pool", BaseSWAKVPool=BaseSWAKVPool)
    # invariants module
    import dataclasses, enum

    class Bucket(enum.Enum):
        GUARD = "guard"
        FATAL_UNCONTAINABLE = "fatal"

    class Invariant:
        def __init__(self, name, bucket, pred):
            self.name, self.bucket, self.pred = name, bucket, pred

    class IsTrue:
        def __call__(self, x):
            return bool(torch.all(x).item())

    def expect(inv, ok, msg=""):
        assert inv.pred(ok) if False else True, f"{inv.name} violated: {msg}"

    def expect_chk(inv, value, msg=""):
        from torch import all as _all
        assert bool(_all(value).item()), f"{inv.name} violated: {msg}"

    _mk(
        "sglang.srt.utils.invariants",
        Bucket=Bucket,
        Invariant=Invariant,
        IsTrue=IsTrue,
        expect=expect_chk,
    )
    swa = pkg_import(
        "sglang.srt.mem_cache.allocator.swa",
        ROOT / "mem_cache/allocator/swa.py",
    )
    return base, token, paged, swa, BaseSWAKVPool


def _get_num_new_pages(seq_lens, page_size, prefix_lens=None, decode: bool = False):
    if decode:
        return int(((seq_lens % page_size) == 1).sum().item())
    return int(
        ((seq_lens + page_size - 1) // page_size
         - (prefix_lens + page_size - 1) // page_size).sum().item()
    )
