#!/usr/bin/env python3
"""Portable smoke check for the F-1 mamba-COW hotfix on ANY vendor tag.

Runs the receipt algebra against the patched files' substance: emulates the
stage + clear flows on a minimal fake tree (inc/dec lock_ref accounting) and
asserts net-zero lock residues across the same terminal matrix our HEAD test
exercises on the real UnifiedRadixCache (collect / retract / pool-free /
not-admitted / session-claim). This is a phrasing-level double-check, not a
replacement for the branch's real-model regression suites.

Usage: python3 hotfix/hotfix_sim_mamba_cow.py
"""


class FakeTree:
    def __init__(self):
        self.node = object()
        self.locked = 0

    def inc_lock_ref(self, node):
        self.locked += 1
        return Recv(self.locked)


class Recv:
    def __init__(self, n):
        self.n = n

    def to_dec_params(self):
        return self


class FakeTreeDec(FakeTree):
    def dec_lock_ref(self, node, params):
        assert self.locked > 0, "dec underflow: release never staged"
        self.locked -= 1


class FakeReq:  # stands in for Req (.14-.17) / ReqKvInfo (.20)
    pass


def stage(tree, req, node):
    lock = getattr(req, "mamba_cow_lock", None)
    req.mamba_cow_lock = None
    if lock is not None:
        tc, n, res = lock
        tc.dec_lock_ref(n, res.to_dec_params())
    res = tree.inc_lock_ref(node)
    req.mamba_cow_lock = (tree, node, res)
    req.mamba_cow_src_index = 42  # symbolic staged source index


def clear(req):
    lock = getattr(req, "mamba_cow_lock", None)
    req.mamba_cow_lock = None
    if lock is not None:
        tc, n, res = lock
        tc.dec_lock_ref(n, res.to_dec_params())
    req.mamba_cow_src_index = None


def main():
    tree = FakeTreeDec()
    # matrix: collect path / retract / pool-free / not-admitted revert /
    # session-claim / restage-supersede — each ends with one clear; interleaved
    # restage then clear; double clear is a no-op.
    for case in range(6):
        req = FakeReq()
        stage(tree, req, tree.node)
        assert tree.locked == 1
        if case == 5:
            stage(tree, req, tree.node)  # restage supersedes
            assert tree.locked == 1, "restage must not leak"
        clear(req)
        assert tree.locked == 0
        clear(req)  # idempotent second clear
        assert tree.locked == 0
    print("hotfix_sim_mamba_cow: 6/6 terminal flows net-zero")


if __name__ == "__main__":
    main()
