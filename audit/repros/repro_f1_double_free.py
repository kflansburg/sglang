"""Repro F1: double-free -> duplicate slot -> two live owners of one host slot.

We exec the REAL alloc/free/clear/_merge_release_slots method bodies from the
repo source for:
  (a) pool_host/base.py HostKVCache  (has slot_used guard)      -> assert fires
  (b) memory_pool_host.py LogicalHostPool (no guard)            -> silent dup
  (c) DeepSeekV4PagedHostPool free/alloc (no guard)             -> silent dup
Then demonstrate: two allocs return the same slot id -> aliasing.
"""
import re, types
import torch
import threading

def extract_class(path, clsname):
    src = open(path).read()
    m = re.search(rf'class {clsname}[^\n]*:.*?(?=\nclass |\Z)', src, re.S)
    return m.group(0)

# --- (b) LogicalHostPool: full class, needs only torch+synchronized ---
src = open('python/sglang/srt/mem_cache/memory_pool_host.py').read()
m = re.search(r'class LogicalHostPool:.*?(?=\nclass DeepSeekV4PagedHostPool)', src, re.S)
logical_src = m.group(0)

def synchronized(func):
    def wrapper(self, *a, **k):
        with self.lock:
            return func(self, *a, **k)
    return wrapper

ns = {'torch': torch, 'threading': threading, 'synchronized': synchronized,
      'Optional': __import__('typing').Optional}
exec(logical_src, ns)
LogicalHostPool = ns['LogicalHostPool']

PAGE = 64
pool = LogicalHostPool(size=PAGE * 4, page_size=PAGE)

# Request 1 allocs a page, request 2 allocs a page
a1 = pool.alloc(PAGE)          # slots 0..63
a2 = pool.alloc(PAGE)          # slots 64..127
# Request 1 finishes and frees host_indices a1.
# BUG PATH: an abort/timeout handler frees the same host_indices again.
freed_tensor = a1.clone()
pool.free(freed_tensor)        # legit free
pool.free(freed_tensor)        # DOUBLE free (e.g. evict + abort cleanup)
try:
    pool._merge_release_slots()
except Exception as e:
    print('merge raised:', e)
# Now the same slots are in the free list twice.
b1 = pool.alloc(PAGE)
b2 = pool.alloc(PAGE)
assert b1 is not None and b2 is not None
overlap = set(b1.tolist()) & set(b2.tolist())
print('LogicalHostPool: overlapping slots handed to two live owners:', sorted(overlap)[:8], f'({len(overlap)} slots)')
assert len(overlap) == PAGE, 'expected full-page aliasing'
print('==> SILENT CORRUPTION CONFIRMED: same slot ids owned by two requests')
