"""Thin-env loader for the replayssm-ring lifecycle regression test.

Loads the REAL python/sglang/srt/mem_cache/allocator/mamba.py and
python/sglang/srt/mem_cache/memory_pool.py from their repo paths with the
heavy deps (triton, sglang.kernels.*, platform/util singletons, loguru chain)
replaced by minimal sys.modules fakes. Only used when the fat sglang package
import failed — a pytest process with a working sglang never sees the stubs.
Keep this file OUT of a shared pytest process with package-importing tests.
"""

import importlib.util
import os
import sys
import types

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))


class _AnyCallableModule(types.ModuleType):
    """Module whose any attribute is a callable returning a benign default."""

    def __getattr__(self, name):
        def _f(*a, **k):
            return None

        _f.__name__ = name
        self.__dict__[name] = _f
        return _f


class _EnvValue:
    """envs.<X> value: .get() default False, attribute access nests."""

    def get(self, default=None):
        return False

    def __call__(self, *a, **k):
        return False

    def __getattr__(self, name):
        v = _EnvValue()
        self.__dict__[name] = v
        return v


class _EnvModule(types.ModuleType):
    """Provides `envs` whose any (nested) attribute is an _EnvValue."""

    def __init__(self, name):
        super().__init__(name)
        self.envs = _EnvValue()

    def __getattr__(self, name):
        v = _EnvValue()
        self.__dict__[name] = v
        return v


class _Platform:
    is_npu = False
    device_type = "cpu"

    def __getattr__(self, name):
        def _f(*a, **k):
            return False

        return _f


def _stub(name, mod):
    sys.modules[name] = mod
    return mod


def _install_stubs():
    # triton + kernels
    for name in [
        "triton",
        "triton.language",
        "sglang",
        "sglang.kernels",
        "sglang.kernels.ops",
        "sglang.kernels.ops.attention",
        "sglang.kernels.ops.attention.dsa",
        "sglang.kernels.ops.attention.dsa.quant_k_cache",
        "sglang.kernels.ops.kvcache",
        "sglang.kernels.ops.kvcache.cache_move",
        "sglang.kernels.ops.kvcache.cache_ops",
        "sglang.kernels.ops.kvcache.kvcache",
        "sglang.kernels.ops.quantization",
        "sglang.kernels.ops.quantization.fp8_kernel",
        "sglang.srt.layers.radix_attention",
        "sglang.srt.layers.dcp",
        "sglang.srt.layers.dcp.layout",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.fp4_kv_cache_quant_method",
        "sglang.srt.layers.attention.dsa",
        "sglang.srt.layers.attention.dsa.utils",
        "sglang.srt.mem_cache.kv_vmm_backing",
        "sglang.srt.mem_cache.index_key_cache",
        "sglang.srt.mem_cache.layout",
        "sglang.srt.mem_cache.layout.token_major",
        "sglang.srt.mem_cache.utils",
        "sglang.srt.utils",
        "sglang.srt.utils.async_probe",
        "sglang.srt.utils.torch_memory_saver_adapter",
        "sglang.srt.configs",
        "sglang.srt.configs.mamba_utils",
    ]:
        if name not in sys.modules:
            _stub(name, _AnyCallableModule(name))
    # triton.language needs enough structure for decorators/attrs at def time
    tl = sys.modules["triton.language"]
    sys.modules["triton"].language = tl
    # constants
    c = types.ModuleType("sglang.srt.constants")
    c.GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"
    _stub("sglang.srt.constants", c)
    # environ.envs
    e = _EnvModule("sglang.srt.environ")
    _stub("sglang.srt.environ", e)
    # platforms.current_platform
    p = types.ModuleType("sglang.srt.platforms")
    p.current_platform = _Platform()
    _stub("sglang.srt.platforms", p)
    # runtime_context.get_parallel
    rc = types.ModuleType("sglang.srt.runtime_context")
    rc.get_parallel = lambda *a, **k: types.SimpleNamespace(
        dcp_enabled=False, tp_rank=0, tp_size=1
    )
    _stub("sglang.srt.runtime_context", rc)
    # allocation_sizing.get_mamba_tracking_slots (real arithmetic: lazy->1 else 2)
    alsz = types.ModuleType("sglang.srt.mem_cache.allocation_sizing")

    def get_mamba_tracking_slots(enable_extra_buffer, enable_extra_buffer_lazy, spec_algorithm=None):
        if not enable_extra_buffer:
            return 0
        return 1 if enable_extra_buffer_lazy else 2

    get_mamba_tracking_slots._real_note = (
        "mirror of allocation_sizing.get_mamba_tracking_slots; fail loud if the "
        "real function's contract changes (returns initial ping-pong slot count)"
    )
    alsz.get_mamba_tracking_slots = get_mamba_tracking_slots
    _stub("sglang.srt.mem_cache.allocation_sizing", alsz)


def load():
    _install_stubs()
    repo = _REPO

    def _load_mod(modname, path):
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        return mod

    alloc_mod = _load_mod(
        "sglang.srt.mem_cache.allocator.mamba",
        os.path.join(repo, "python/sglang/srt/mem_cache/allocator/mamba.py"),
    )
    mempool_mod = _load_mod(
        "sglang.srt.mem_cache.memory_pool_loaded_for_ring_test",
        os.path.join(repo, "python/sglang/srt/mem_cache/memory_pool.py"),
    )
    return alloc_mod, mempool_mod


if __name__ == "__main__":
    a, m = load()
    print("loader OK:", a.MambaSlotAllocator, m.HybridReqToTokenPool)
