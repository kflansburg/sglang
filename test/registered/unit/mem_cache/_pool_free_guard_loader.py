"""Loader for test_mem_pool_host_free_guards.py (thin queue venv fallback).

Loads the REAL alloc/free/clear release-list machinery from the host-pool
source files WITHOUT importing the sglang package (whose root __init__ chain
needs triton and other heavy deps the merge-queue venv may lack).

Used only when the regular package import fails; a fat CI env never sees
these stubs. The extraction takes verbatim method bodies from the repo files
via regex and exec's them into stub classes, so the code under test is the
same source text CI would import.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Optional

import torch


def _synchronized(func):
    import functools

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return wrapper


_REPO_HERE = os.path.dirname(os.path.abspath(__file__))
_MEM_CACHE = os.path.join(_REPO_HERE, os.pardir, os.pardir, os.pardir, os.pardir,
                          "python", "sglang", "srt", "mem_cache")


def _read(relpath: str) -> str:
    with open(os.path.join(_MEM_CACHE, relpath)) as f:
        return f.read()


def _grab(src: str, pattern: str) -> str:
    m = re.search(pattern, src, re.S)
    if m is None:
        raise RuntimeError(f"pool-guard loader: pattern not found: {pattern[:60]}")
    return m.group(0)


def _base_helper(ns: dict) -> None:
    """exec assert_valid_host_slot_indices from pool_host/base.py."""
    src = _read(os.path.join("pool_host", "base.py"))
    helper = _grab(
        src, r"\ndef assert_valid_host_slot_indices\(.*?(?=\n\nclass |\ndef )"
    )
    exec(helper, ns)


def _stub_class(name: str, bodies: list[str], ns: dict):
    code = f"class {name}:\n" + "\n".join(bodies)
    exec(code, ns)
    return ns[name]


def load():
    """Return (LogicalHostPool, DeepSeekV4PagedHostPool, MambaPoolHost, MHAHostClass).

    Each class has the real clear/available_size/_merge_release_slots/
    alloc/free methods extracted from the repo source.
    """
    ns = {
        "torch": torch,
        "threading": threading,
        "synchronized": _synchronized,
        "Optional": Optional,
    }
    _base_helper(ns)

    base_src = _read(os.path.join("pool_host", "base.py"))
    merge_body = _grab(
        base_src, r"    def _merge_release_slots\(.*?(?=\n    @property)"
    )

    # --- LogicalHostPool (whole class; memory_pool_host.py) ---
    mph_src = _read("memory_pool_host.py")
    logical_src = _grab(
        mph_src, r"class LogicalHostPool:.*?(?=\nclass DeepSeekV4PagedHostPool)"
    )
    exec(logical_src, ns)

    # --- DeepSeekV4PagedHostPool: select methods only ---
    dsv4_block = _grab(
        mph_src, r"class DeepSeekV4PagedHostPool.*?(?=\nclass DeepSeekV4StateHostPool)"
    )
    dsv4 = _stub_class(
        "DeepSeekV4PagedHostPool",
        [
            _grab(dsv4_block, r"    def clear\(self\):.*?(?=\n    def available_size)"),
            _grab(dsv4_block, r"    def available_size\(self\):.*?(?=\n    @synchronized)"),
            merge_body,
            _grab(
                dsv4_block,
                r"    @synchronized\n    def alloc\(.*?(?=\n    @synchronized\n    def free)",
            ),
            _grab(
                dsv4_block, r"    @synchronized\n    def free\(.*?return len\(indices\)\n"
            ),
        ],
        ns,
    )

    # --- MambaPoolHost: select methods only ---
    mamba_src = _read(os.path.join("pool_host", "mamba.py"))
    mamba_block = _grab(mamba_src, r"class MambaPoolHost.*?(?=\nclass |\Z)")
    mamba = _stub_class(
        "MambaPoolHost",
        [
            _grab(
                mamba_block,
                r"    @synchronized\n    def clear\(.*?(?=\n    def available_size)",
            ),
            _grab(mamba_block, r"    def available_size\(self\):.*?(?=\n    @synchronized)"),
            merge_body,
            _grab(
                mamba_block,
                r"    @synchronized\n    def alloc\(.*?(?=\n    @synchronized\n    def free)",
            ),
            _grab(
                mamba_block, r"    @synchronized\n    def free\(.*?return len\(indices\)\n"
            ),
        ],
        ns,
    )

    # --- Base HostKVCache: select methods ---
    mha = _stub_class(
        "MHATokenToKVPoolHost",
        [
            (
                "    @property\n    def logical_size(self):\n        return self.size\n"
                "    @property\n    def logical_page_size(self):\n        return self.page_size\n"
            ),
            _grab(
                base_src,
                r"    @synchronized\n    def clear\(.*?(?=\n    def available_size)",
            ),
            _grab(base_src, r"    def available_size\(self\):.*?(?=\n    def _merge)"),
            merge_body,
            _grab(
                base_src,
                r"    @synchronized\n    def alloc\(.*?(?=\n    @synchronized\n    def free)",
            ),
            _grab(base_src, r"    @synchronized\n    def free\(.*?\Z"),
        ],
        ns,
    )

    classes = (ns["LogicalHostPool"], dsv4, mamba, mha)
    makers = [_make_logical, _make_dsv4, _make_mamba, _make_mha]
    return {cls: mk(cls) for cls, mk in zip(classes, makers)}


def _make_logical(cls):
    def make():
        return cls(size=8, page_size=2)

    return make


def _make_dsv4(cls):
    def make():
        pool = cls.__new__(cls)
        pool.size = 8
        pool.slot_page_size = 2
        pool.lock = threading.RLock()
        pool.clear()
        return pool

    return make


def _make_mamba(cls):
    def make():
        pool = cls.__new__(cls)
        pool.size = 8
        pool.page_size = 1
        pool.device = "cpu"
        pool.lock = threading.RLock()
        pool.clear()
        return pool

    return make


def _make_mha(cls):
    def make():
        pool = cls.__new__(cls)
        pool.size = 8
        pool.dcp_size = 1
        pool.page_size = 2
        pool.device = "cpu"
        pool.lock = threading.RLock()
        pool.clear()
        return pool

    return make
