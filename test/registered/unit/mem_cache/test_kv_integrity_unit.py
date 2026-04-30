import os
import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")

from sglang.srt.mem_cache.kv_integrity import (
    _NullTracker,
    make_tracker,
)


class TestNullTracker(unittest.TestCase):
    def test_default_factory_returns_null_when_env_unset(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_KV_INTEGRITY", None)
            tracker = make_tracker(num_pages=1024, page_size=64, req_pool_size=128)
        self.assertIsInstance(tracker, _NullTracker)

    def test_factory_returns_null_for_explicit_off(self):
        with mock.patch.dict(os.environ, {"SGLANG_KV_INTEGRITY": "off"}):
            tracker = make_tracker(num_pages=1024, page_size=64, req_pool_size=128)
        self.assertIsInstance(tracker, _NullTracker)

    def test_factory_returns_null_for_unknown_value_with_warning(self):
        with mock.patch.dict(os.environ, {"SGLANG_KV_INTEGRITY": "device"}):
            with self.assertLogs("sglang", level="WARNING") as cm:
                tracker = make_tracker(num_pages=1024, page_size=64, req_pool_size=128)
        self.assertIsInstance(tracker, _NullTracker)
        self.assertTrue(any("not implemented" in line for line in cm.output))

    def test_factory_returns_real_tracker_for_host(self):
        with mock.patch.dict(os.environ, {"SGLANG_KV_INTEGRITY": "host"}):
            with self.assertRaises(NotImplementedError):
                make_tracker(num_pages=1024, page_size=64, req_pool_size=128)

    def test_null_tracker_methods_are_no_op(self):
        t = _NullTracker()
        self.assertIsNone(t.on_alloc(0, [1, 2, 3]))
        self.assertIsNone(t.on_prefix_hit(0, [1, 2, 3]))
        self.assertIsNone(t.on_free([1, 2, 3]))
        self.assertIsNone(t.on_req_free(0))
        self.assertEqual(t.validate_batch(batch=None), [])


if __name__ == "__main__":
    unittest.main()
