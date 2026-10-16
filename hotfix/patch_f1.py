#!/usr/bin/env python3
"""F-1 mamba COW hotfix: edit-in-place on a tag worktree.
Usage: patch_f1.py <worktree-root> <tag_kind>
Edits staging sites + rewires existing clear sites to release the staged lock
first; adds a pool-free clear on .14/.15.p1/.17 (HEAD parity).
"""
import os
import sys

WT, TAG = sys.argv[1], sys.argv[2]
RECV_KV = TAG == "v0.5.20"           # .20 uses req.kv.<field>
R = "req.kv" if RECV_KV else "req"   # receipt carrier


def edit(rel, transforms):
    p = os.path.join(WT, rel)
    t = open(p).read()
    for (old, new) in transforms:
        c = t.count(old)
        assert c == 1, f"{rel}: count={c} for snippet {old[:70]!r}"
        t = t.replace(old, new)
    open(p, "w").write(t)
    print("edited", rel)


def stage_snippet(var_node):
    return (
        f"# F-1 HOTFIX: pin the COW source node for the deferral window.\n"
        f"            # Supersede: release any previously staged lock first.\n"
        f"            _f1_lock = getattr({R}, 'mamba_cow_lock', None)\n"
        f"            {R}.mamba_cow_lock = None\n"
        f"            if _f1_lock is not None:\n"
        f"                _tc, _node, _res = _f1_lock\n"
        f"                _tc.dec_lock_ref(_node, _res.to_dec_params())\n"
        f"            _f1_res = self.inc_lock_ref({var_node})\n"
        f"            {R}.mamba_cow_lock = (self, {var_node}, _f1_res)\n"
        f"            {R}.mamba_cow_src_index = {var_node}.mamba_value"
    )


stage_old = f"            {R}.mamba_cow_src_index = {{VAR}}.mamba_value"

# 1) staging sites
if not RECV_KV:
    edit(
        "python/sglang/srt/mem_cache/hi_mamba_radix_cache.py",
        [(stage_old.format(VAR="mamba_node"), stage_snippet("mamba_node"))],
    )
edit(
    "python/sglang/srt/mem_cache/mamba_radix_cache.py",
    [(stage_old.format(VAR="last_node"), stage_snippet("last_node"))],
)

# 2) clear sites: release the receipt at every existing clear-point.
def clear_helper_prefix():
    # inline identical release block used at every clear site
    return (
        f"# F-1 HOTFIX: release the staged mamba COW source lock (if any).\n"
        f"                _f1_lock = getattr({R}, 'mamba_cow_lock', None)\n"
        f"                {R}.mamba_cow_lock = None\n"
        f"                if _f1_lock is not None:\n"
        f"                    _tc, _node, _res = _f1_lock\n"
        f"                    _tc.dec_lock_ref(_node, _res.to_dec_params())\n"
    )


def clear_helper_prefix_self():  # for the SB self.<field> site (reset_for_retract on Req)
    return (
        f"# F-1 HOTFIX: release the staged mamba COW source lock (if any).\n"
        f"        _f1_lock = getattr(self.kv if False else self{'._kv' if False else ''}, 'mamba_cow_lock', None)\n"
    )


sb = open(os.path.join(WT, "python/sglang/srt/managers/schedule_batch.py")).read()

if RECV_KV:
    # .20: self.kv.mamba_cow_src_index = None  (reset site, inside Req class)
    sb_old_reset = (
        "        self.kv.mamba_cow_src_index = None"
    )
    sb_new_reset = (
        "        # F-1 HOTFIX\n"
        "        _f1_lock = getattr(self.kv, 'mamba_cow_lock', None)\n"
        "        self.kv.mamba_cow_lock = None\n"
        "        if _f1_lock is not None:\n"
        "            _tc, _node, _res = _f1_lock\n"
        "            _tc.dec_lock_ref(_node, _res.to_dec_params())\n"
        "        self.kv.mamba_cow_src_index = None"
    )
    sb_old_collect = (
        "                req.kv.mamba_cow_src_index = None"
    )
    sb_new_collect = (
        "                # F-1 HOTFIX\n"
        "                _f1_lock = getattr(req.kv, 'mamba_cow_lock', None)\n"
        "                req.kv.mamba_cow_lock = None\n"
        "                if _f1_lock is not None:\n"
        "                    _tc, _node, _res = _f1_lock\n"
        "                    _tc.dec_lock_ref(_node, _res.to_dec_params())\n"
        "                req.kv.mamba_cow_src_index = None"
    )
else:
    sb_old_reset = "        self.mamba_cow_src_index = None"
    sb_new_reset = (
        "        # F-1 HOTFIX\n"
        "        _f1_lock = getattr(self, 'mamba_cow_lock', None)\n"
        "        self.mamba_cow_lock = None\n"
        "        if _f1_lock is not None:\n"
        "            _tc, _node, _res = _f1_lock\n"
        "            _tc.dec_lock_ref(_node, _res.to_dec_params())\n"
        "        self.mamba_cow_src_index = None"
    )
    sb_old_collect = "                req.mamba_cow_src_index = None"
    sb_new_collect = (
        "                # F-1 HOTFIX\n"
        "                _f1_lock = getattr(req, 'mamba_cow_lock', None)\n"
        "                req.mamba_cow_lock = None\n"
        "                if _f1_lock is not None:\n"
        "                    _tc, _node, _res = _f1_lock\n"
        "                    _tc.dec_lock_ref(_node, _res.to_dec_params())\n"
        "                req.mamba_cow_src_index = None"
    )

transforms = [(sb_old_reset, sb_new_reset), (sb_old_collect, sb_new_collect)]
p = os.path.join(WT, "python/sglang/srt/managers/schedule_batch.py")
t = sb
for (old, new) in transforms:
    c = t.count(old)
    assert c == 1, f"schedule_batch count={c} for {old[:40]!r}"
    t = t.replace(old, new)
open(p, "w").write(t)
print("edited schedule_batch.py")

# 3) scheduler not-admitted revert site (.17 scheduler:3255, .20 :3959; absent .14/.15p1)
scher_path = os.path.join(WT, "python/sglang/srt/managers/scheduler.py")
t = open(sch_path := scher_path).read() if os.path.exists(scher_path) else ""
sched_site_line = None
for _cand in (
    [l for l in t.splitlines() if "mamba_cow_src_index = None" in l and "req" in l and "kv" in l]
    if RECV_KV else
    [l for l in t.splitlines() if "mamba_cow_src_index = None" in l and "req." in l and "self." not in l]
):
    _cand = _cand
    break
sched_site_lines = [
    l for l in t.splitlines()
    if "mamba_cow_src_index = None" in l
    and ("req.kv." in l if RECV_KV else ("req.mamba_cow_src_index" in l))
]
if len(sched_site_lines) == 1:
    sched_site = sched_site_lines[0]
    indent = sched_site[: len(sched_site) - len(sched_site.lstrip())]
    recv = "req.kv" if RECV_KV else "req"
    sched_new = (
        f"{indent}# F-1 HOTFIX\n"
        f"{indent}_f1_lock = getattr({recv}, 'mamba_cow_lock', None)\n"
        f"{indent}{recv}.mamba_cow_lock = None\n"
        f"{indent}if _f1_lock is not None:\n"
        f"{indent}    _tc, _node, _res = _f1_lock\n"
        f"{indent}    _tc.dec_lock_ref(_node, _res.to_dec_params())\n"
        f"{sched_site}\n"
    )
    c = t.count(sched_site)
    assert c == 1, f"scheduler count={c}"
    t = t.replace(sched_site, sched_new)
    open(scher_path, "w").write(t)
    print("edited scheduler.py")
else:
    print("scheduler.py: no clear site on this tag (skip)")

# 4) memory_pool free site: .20 has one (:1709); .14-.17 DO NOT — add to the
#    HybridReqToTokenPool.free near mamba field clears.
mp_path = os.path.join(WT, "python/sglang/srt/mem_cache/memory_pool.py")
t = open(mp_path).read()
f1_pool_release = (
    "# F-1 HOTFIX: release staged mamba COW source lock at pool free.\n"
    "            _f1_lock = getattr({RECV}, 'mamba_cow_lock', None)\n"
    "            {RECV}.mamba_cow_lock = None\n"
    "            if _f1_lock is not None:\n"
    "                _tc, _node, _res = _f1_lock\n"
    "                _tc.dec_lock_ref(_node, _res.to_dec_params())\n"
)
mp_site = (
    "            req.kv.mamba_cow_src_index = None" if RECV_KV
    else "            req.mamba_cow_src_index = None"
)
if mp_site in t:
    recv = "req.kv" if RECV_KV else "req"
    new_block = f1_pool_release.replace("{RECV}", recv) + mp_site
    c = t.count(mp_site)
    assert c == 1, f"memory_pool count={c}"
    t = t.replace(mp_site, new_block)
    open(mp_path, "w").write(t)
    print("edited memory_pool.py (augmented existing clear)")
else:
    # .14-.17: add to free_mamba_cache right before the allocator free.
    anchor = "        self.mamba_allocator.free(mamba_index.unsqueeze(0))"
    c = t.count(anchor)
    assert c == 1, f"memory_pool anchor count={c}"
    release8 = f1_pool_release.replace("{RECV}", "req").replace(
        "            _f1_lock", "        _f1_lock"
    ).replace("            req.mamba_cow_lock", "        req.mamba_cow_lock"
    ).replace("            if _f1_lock", "        if _f1_lock"
    ).replace("                _tc", "            _tc"
    ).replace("                _node", "            _node"
    ).replace("                _res", "            _res"
    ).replace("                    _tc.dec_lock_ref", "            _tc.dec_lock_ref")
    t = t.replace(anchor, release8 + anchor, 1)
    open(mp_path, "w").write(t)
    print("edited memory_pool.py (added pool-free clear)")
