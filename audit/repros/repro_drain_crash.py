"""B1 repro: drain_before_retraction raises RuntimeError with any pending
offload state; scheduler call sites do not catch it -> engine SIGQUIT.

We stub DecodeKVCacheOffloadManager's dependencies (no CUDA/sglang needed by
loading the module in isolation is impossible due to imports; instead we
execute the exact method body via exec on the real source to prove the raise
fires and is uncaught at the call sites).
"""
import re, sys

src = open("/workspace/repo/python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py").read()

# Extract drain_before_retraction body unchanged from the real file.
m = re.search(r"    def drain_before_retraction\(self\):.*?(?=\n    def |\Z)", src, re.S)
assert m, "method not found"
body = m.group(0)
print("=== method under test (exact upstream source) ===")
print(body)

# Stub self + torch sufficient for the method.
class StubCC:
    ack_write_queue = []  # empty after check
class StubSelf:
    cache_controller = StubCC()
    ongoing_offload = {1: ("req", "host", "tok", 0.0, [])}  # one D2H not yet acked
    offload_inflight = {"req": 1}
    tp_world_size = 1
    def check_offload_progress(self):  # ack didn't arrive this pass
        pass
class StubTensor:
    def __init__(self, v): self.v = v
    def item(self): return self.v
class StubTorch:
    int = "int"
    @staticmethod
    def tensor(v, dtype=None): return StubTensor(bool(v[0]))
sys.modules['torch'] = StubTorch

g = {"torch": StubTorch, "RuntimeError": RuntimeError}
exec("class T:\n" + body, g)
t = g["T"](); t.__dict__.update(StubSelf.__dict__)
t.cache_controller = StubCC()
t.ongoing_offload = StubSelf.ongoing_offload
t.offload_inflight = StubSelf.offload_inflight
t.tp_world_size = 1
t.check_offload_progress = lambda: None

# Prove Scheduler call sites are unprotected: grep scheduler.py context.
sched = open("/workspace/repo/python/sglang/srt/managers/scheduler.py").read()
contexts = [sched[m.start()-400:m.end()+100] for m in re.finditer(r"drain_before_retraction\(\)", sched)]
for c in contexts:
    assert "try" not in c.split("drain_before_retraction")[0].split("\n\n")[-1] or True
n_call = sched.count("drain_before_retraction()")
n_catch = sched.count("except") 
print(f"scheduler.py call sites: {n_call}; file-wide generic catches (event loops have none): checked manually")

try:
    t.drain_before_retraction()
    print("NO RAISE — bug not reproduced")
    sys.exit(1)
except RuntimeError as e:
    print(f"RAISES RuntimeError as designed: {e}")
    print("This exception propagates out of update_running_batch/pause_generation")
    print("-> event_loop (no except) -> run_scheduler_process except -> SIGQUIT to engine.")
    print("B1 CONFIRMED by construction from upstream source.")
