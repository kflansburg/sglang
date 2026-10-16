"""Pin the defensive row bound in the C128 draft-state cleanup kernel.

The kernel writes ``state + (req_pool_idx * ring_size + slot) * 2 * half``
into a shared flat compress-state buffer. Correctness of the row index is an
invariant of pool sizing (PD decode prealloc inflates req slots); an out-of-
bounds row silently clobbers another request's compress state. The kernel
must therefore carry an explicit num_rows guard. This test pins the plumbing
on the host side without requiring CUDA/Triton runtime.
"""

import inspect
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestC128CleanupBounds(CustomTestCase):
    def test_kernel_receives_num_rows_from_state_shape(self):
        from sglang.kernels.ops.attention.dsv4 import c128_cleanup

        src = inspect.getsource(c128_cleanup)
        # Host wrapper derives the bound from the actual buffer.
        self.assertIn("num_rows=state.shape[0]", src)
        # Kernel signature declares and uses the bound guard.
        kernel_src = inspect.getsource(
            c128_cleanup._clear_unaccepted_c128_draft_states_kernel.fn
        )
        self.assertIn("if row < 0 or row >= num_rows:", kernel_src)

    def test_wrapper_validates_half_split(self):
        # half must equal state.shape[-1] // 2 for the KV+score layout; guard
        # against a silent signature drift between the two writes.
        from sglang.kernels.ops.attention.dsv4 import c128_cleanup

        src = inspect.getsource(c128_cleanup.clear_unaccepted_c128_draft_states)
        self.assertIn("half = state.shape[-1] // 2", src)


if __name__ == "__main__":
    unittest.main()
