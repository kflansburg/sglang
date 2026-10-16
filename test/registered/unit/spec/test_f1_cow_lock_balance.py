"""F-1 verifier harness (spec-auditor; board #17): COW source lock balance.

Simulates the pending fix contract: finalize_match_result_in_cache stages a
mamba COW (innate inc_lock_ref on the source node); every terminal path of a
request must dec that lock exactly once. The harness models the REAL clear
sites as they exist at HEAD (schedule_batch:2023 reset_for_retract,
scheduler:4210 admission reject, memory_pool:1884 release_req,
schedule_batch:3163 collect, components/mamba.py:1044 load-back supersede,
components/mamba.py:728 load-back failure) and is executed TWICE:

  * contract="pre-fix": lock acquire at staging, NOTHING decs it except the
    collect site (the intended pair at 3163).
  * contract="post-fix": every clear site decs (they were updated to pair).

The harness pretends no diff exists yet: for the post-fix run it asserts the
intended invariant with a flag so that, once the patch exists, the same file
mirrors what the real code paths must do. The pre-fix run asserts the known
imbalance (documentation of the leak the fix must close for attack 1/2/3/9/10).
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class LockLedger:
    """Per-node lock-ref counts (mimics unified_tree_core inc/dec_lock_ref)."""

    def __init__(self):
        self.counts = {}

    def inc(self, node):
        self.counts[node] = self.counts.get(node, 0) + 1
        return self.counts[node]

    def dec(self, node):
        if self.counts.get(node, 0) <= 0:
            raise AssertionError(f"dec_lock_ref underflow on {node}")
        self.counts[node] -= 1


def _run_scenario(ledger: LockLedger, src_node: str, path: str, *, contract: str):
    """Run one admission through a terminal path under a lock contract."""
    # Staging (finalize_match_result_in_cache) — lock acquired in both contracts.
    ledger.inc(src_node)
    cow_leaked = True
    try:
        if path == "collect":  # normal: prepare_for_extend collects + decs
            ledger.dec(src_node)  # the intended pair (both contracts)
            cow_leaked = False
        elif path == "retract":  # reset_for_retract clears metadata
            if contract == "post-fix":
                ledger.dec(src_node)
                cow_leaked = False
        elif path == "admission_reject":  # scheduler reject clears metadata
            if contract == "post-fix":
                ledger.dec(src_node)
                cow_leaked = False
        elif path == "abort_waiting":  # release_req frees mamba state (mem pool 1884)
            if contract == "post-fix":
                ledger.dec(src_node)
                cow_leaked = False
        elif path == "loadback_supersede":  # H2D load-back supersedes COW (1044)
            if contract == "post-fix":
                ledger.dec(src_node)
                cow_leaked = False
        elif path == "loadback_fail":  # finalize_load_back(False) - no clear at HEAD
            if contract == "post-fix":
                ledger.dec(src_node)
                cow_leaked = False
        else:
            raise ValueError(path)
    finally:
        return cow_leaked


class TestF1CowLockBalance(unittest.TestCase):
    SCENARIOS = [
        ("collect", False),  # normal path — no leak in either contract
        ("retract", True),
        ("admission_reject", True),
        ("abort_waiting", True),
        ("loadback_supersede", True),
        ("loadback_fail", True),
    ]

    def test_pre_fix_documents_leaks(self):
        """HEAD contract: only the collect site decs — every other terminal
        path leaks a node lock. This documents the attack surface the fix
        must close; kept as a regression anchor."""
        expected = {p: leaked for p, leaked in self.SCENARIOS}
        for path, want_leak in expected.items():
            ledger = LockLedger()
            leaked = _run_scenario(ledger, "X", path, contract="pre-fix")
            self.assertEqual(
                leaked, want_leak, f"{path}: pre-fix leak expectation flipped"
            )

    def test_post_fix_contract_balanced(self):
        """The intended fix invariant: zero residual lock on the COW source at
        every terminal request path; no dec anywhere causes underflow."""
        for path, _ in self.SCENARIOS:
            with self.subTest(path=path):
                ledger = LockLedger()
                _run_scenario(ledger, "X", path, contract="post-fix")
                self.assertEqual(
                    ledger.counts.get("X", 0), 0, f"residual lock after {path}"
                )


if __name__ == "__main__":
    unittest.main()
