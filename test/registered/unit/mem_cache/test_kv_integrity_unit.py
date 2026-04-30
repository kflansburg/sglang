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
            tracker = make_tracker(num_pages=1024, page_size=64, req_pool_size=128)
        from sglang.srt.mem_cache.kv_integrity import KvIntegrityTracker

        self.assertIsInstance(tracker, KvIntegrityTracker)

    def test_null_tracker_methods_are_no_op(self):
        t = _NullTracker()
        self.assertIsNone(t.on_alloc(0, [1, 2, 3]))
        self.assertIsNone(t.on_prefix_hit(0, [1, 2, 3]))
        self.assertIsNone(t.on_free([1, 2, 3]))
        self.assertIsNone(t.on_req_free(0))
        self.assertEqual(t.validate_batch(batch=None), [])


# Sibling import: pytest adds the test file's directory to sys.path, so the
# leading-underscore helpers module created in Task 2 is importable directly.
from _kv_integrity_helpers import (
    assert_no_owner,
    assert_owners,
    build_tracker,
    slot_tensor,
    slots_for_pages,
)


class TestOnAllocOnFree(unittest.TestCase):
    def test_on_alloc_sets_bit_for_each_page(self):
        t = build_tracker(num_pages=64, page_size=4, req_pool_size=128)
        slots = slots_for_pages([3, 5, 7], page_size=4)
        t.on_alloc(req_pool_idx=42, slot_indices=slot_tensor(slots))
        for page in (3, 5, 7):
            assert_owners(t, page, {42})
        for page in (0, 1, 2, 4, 6, 8):
            assert_no_owner(t, page)

    def test_on_alloc_updates_req_pages(self):
        t = build_tracker(num_pages=64, page_size=4)
        t.on_alloc(req_pool_idx=42, slot_indices=slots_for_pages([3, 5], page_size=4))
        self.assertEqual(t.req_pages[42], {3, 5})

    def test_on_alloc_high_req_pool_idx_uses_second_word(self):
        t = build_tracker(num_pages=64, page_size=4, req_pool_size=200)
        t.on_alloc(req_pool_idx=130, slot_indices=slots_for_pages([3], page_size=4))
        # bit 130 = word 2, bit 2.
        self.assertEqual(int(t.page_owners[3, 2]) & (1 << 2), 1 << 2)
        assert_owners(t, 3, {130})

    def test_on_free_zeros_page_rows(self):
        t = build_tracker(num_pages=64, page_size=4)
        t.on_alloc(req_pool_idx=1, slot_indices=slots_for_pages([3, 5], page_size=4))
        t.on_alloc(req_pool_idx=2, slot_indices=slots_for_pages([3], page_size=4))
        assert_owners(t, 3, {1, 2})
        t.on_free(slot_tensor(slots_for_pages([3], page_size=4)))
        assert_no_owner(t, 3)
        assert_owners(t, 5, {1})

    def test_on_free_handles_empty(self):
        t = build_tracker(num_pages=64, page_size=4)
        t.on_free(slot_tensor([]))


class TestOnPrefixHitOnReqFree(unittest.TestCase):
    def test_prefix_hit_or_s_in_existing_owner(self):
        t = build_tracker(num_pages=64, page_size=4)
        t.on_alloc(req_pool_idx=1, slot_indices=slots_for_pages([3, 5], page_size=4))
        t.on_prefix_hit(
            req_pool_idx=2,
            slot_indices=slots_for_pages([3, 5], page_size=4),
        )
        assert_owners(t, 3, {1, 2})
        assert_owners(t, 5, {1, 2})
        self.assertEqual(t.req_pages[2], {3, 5})

    def test_prefix_hit_does_not_disturb_unrelated_pages(self):
        t = build_tracker(num_pages=64, page_size=4)
        t.on_alloc(req_pool_idx=1, slot_indices=slots_for_pages([3, 5, 7], page_size=4))
        t.on_prefix_hit(
            req_pool_idx=2,
            slot_indices=slots_for_pages([3], page_size=4),
        )
        assert_owners(t, 3, {1, 2})
        assert_owners(t, 5, {1})
        assert_owners(t, 7, {1})

    def test_req_free_removes_only_that_req_bit(self):
        t = build_tracker(num_pages=64, page_size=4)
        t.on_alloc(req_pool_idx=1, slot_indices=slots_for_pages([3, 5], page_size=4))
        t.on_prefix_hit(
            req_pool_idx=2,
            slot_indices=slots_for_pages([3, 5], page_size=4),
        )
        t.on_req_free(1)
        assert_owners(t, 3, {2})
        assert_owners(t, 5, {2})
        self.assertNotIn(1, t.req_pages)

    def test_final_req_free_leaves_zero_row(self):
        t = build_tracker(num_pages=64, page_size=4)
        t.on_alloc(req_pool_idx=1, slot_indices=slots_for_pages([3], page_size=4))
        t.on_req_free(1)
        assert_no_owner(t, 3)
        self.assertNotIn(1, t.req_pages)

    def test_req_free_unknown_idx_is_safe(self):
        t = build_tracker(num_pages=64, page_size=4)
        t.on_req_free(999)


from types import SimpleNamespace


def fake_batch(pairs):
    """pairs is a list of (rid, req_pool_idx)."""
    reqs = [SimpleNamespace(rid=rid, req_pool_idx=idx) for rid, idx in pairs]
    return SimpleNamespace(reqs=reqs)


class TestValidateBatch(unittest.TestCase):
    def test_clean_batch_returns_no_bad_reqs(self):
        t = build_tracker(num_pages=64, page_size=4)
        t.on_alloc(req_pool_idx=1, slot_indices=slots_for_pages([3, 5], page_size=4))
        t.on_alloc(req_pool_idx=2, slot_indices=slots_for_pages([7], page_size=4))
        bad = t.validate_batch(fake_batch([("a", 1), ("b", 2)]))
        self.assertEqual(bad, [])

    def test_violation_detected_when_req_pages_includes_unowned_page(self):
        t = build_tracker(num_pages=64, page_size=4)
        t.on_alloc(req_pool_idx=1, slot_indices=slots_for_pages([3], page_size=4))
        t.req_pages[1].add(99)
        bad = t.validate_batch(fake_batch([("a", 1)]))
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad[0].req.rid, "a")
        self.assertEqual(bad[0].bad_pages, [99])

    def test_violation_logs_structured_warning(self):
        t = build_tracker(num_pages=64, page_size=4)
        t.on_alloc(req_pool_idx=1, slot_indices=slots_for_pages([3], page_size=4))
        t.req_pages[1].add(99)
        with self.assertLogs("sglang", level="WARNING") as cm:
            t.validate_batch(fake_batch([("a", 1)]))
        joined = "\n".join(cm.output)
        self.assertIn("rid=a", joined)
        self.assertIn("req_pool_idx=1", joined)
        self.assertIn("99", joined)

    def test_validate_skips_reqs_without_pool_idx(self):
        t = build_tracker(num_pages=64, page_size=4)
        bad = t.validate_batch(fake_batch([("a", None)]))
        self.assertEqual(bad, [])

    def test_validate_skips_reqs_with_no_known_pages(self):
        t = build_tracker(num_pages=64, page_size=4)
        bad = t.validate_batch(fake_batch([("a", 5)]))
        self.assertEqual(bad, [])


class TestAllocatorIntegration(unittest.TestCase):
    def test_paged_allocator_free_invokes_tracker_on_free(self):
        import torch

        from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
        from sglang.srt.mem_cache.kv_integrity import KvIntegrityTracker

        page_size = 4
        num_pages = 16
        size = num_pages * page_size
        kvcache = SimpleNamespace(device="cpu", dtype=torch.bfloat16)
        allocator = PagedTokenToKVPoolAllocator(
            size=size,
            page_size=page_size,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=kvcache,
            need_sort=False,
        )
        tracker = KvIntegrityTracker(
            num_pages=num_pages, page_size=page_size, req_pool_size=64
        )
        allocator.tracker = tracker
        slots = torch.arange(8, 16, dtype=torch.int64, device="cpu")
        tracker.on_alloc(req_pool_idx=3, slot_indices=slots)
        assert_owners(tracker, 2, {3})
        assert_owners(tracker, 3, {3})
        allocator.free(slots)
        assert_no_owner(tracker, 2)
        assert_no_owner(tracker, 3)

    def test_default_allocator_has_null_tracker(self):
        import torch

        from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
        from sglang.srt.mem_cache.kv_integrity import _NullTracker

        kvcache = SimpleNamespace(device="cpu", dtype=torch.bfloat16)
        allocator = PagedTokenToKVPoolAllocator(
            size=64,
            page_size=4,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=kvcache,
            need_sort=False,
        )
        self.assertIsInstance(allocator.tracker, _NullTracker)
        allocator.free(torch.tensor([0, 1, 2, 3], dtype=torch.int64))


class TestReqToTokenPoolIntegration(unittest.TestCase):
    def test_req_to_token_pool_free_invokes_tracker_on_req_free(self):
        from sglang.srt.mem_cache.kv_integrity import KvIntegrityTracker
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

        pool = ReqToTokenPool(
            size=8, max_context_len=128, device="cpu", enable_memory_saver=False
        )
        tracker = KvIntegrityTracker(num_pages=64, page_size=4, req_pool_size=8)
        pool.tracker = tracker
        tracker.on_alloc(
            req_pool_idx=3, slot_indices=slots_for_pages([1, 2], page_size=4)
        )
        req = SimpleNamespace(rid="a", req_pool_idx=3)
        pool.free(req)
        self.assertNotIn(3, tracker.req_pages)
        assert_no_owner(tracker, 1)
        assert_no_owner(tracker, 2)


class TestRecordExtend(unittest.TestCase):
    def _make_batch(self, reqs_info, prefix_lens_cpu, seq_lens_cpu):
        import torch

        reqs = [
            SimpleNamespace(
                rid=info["rid"],
                req_pool_idx=info["req_pool_idx"],
                prefix_indices=info["prefix_indices"],
            )
            for info in reqs_info
        ]
        return SimpleNamespace(
            reqs=reqs,
            prefix_lens_cpu=torch.tensor(prefix_lens_cpu, dtype=torch.int64),
            seq_lens_cpu=torch.tensor(seq_lens_cpu, dtype=torch.int64),
        )

    def test_records_extend_alloc_and_prefix_hits_per_req(self):
        import torch

        from sglang.srt.managers.schedule_batch import _record_extend_for_tracker

        tracker = build_tracker(num_pages=64, page_size=4)
        batch = self._make_batch(
            reqs_info=[
                {
                    "rid": "a",
                    "req_pool_idx": 1,
                    "prefix_indices": torch.tensor([], dtype=torch.int64),
                },
                {
                    "rid": "b",
                    "req_pool_idx": 2,
                    "prefix_indices": torch.arange(0, 8, dtype=torch.int64),
                },
            ],
            prefix_lens_cpu=[0, 8],
            seq_lens_cpu=[8, 12],
        )
        # Slots layout: req a gets slots [16..23] (8 new), req b gets slots [24..27] (4 new).
        out_cache_loc = torch.arange(16, 28, dtype=torch.int64)
        _record_extend_for_tracker(tracker, batch, out_cache_loc)
        # req a's pages: 4, 5 (slots 16-19, 20-23 with page_size=4)
        assert_owners(tracker, 4, {1})
        assert_owners(tracker, 5, {1})
        # req b's prefix-hit pages: 0, 1 (slots 0-3, 4-7)
        assert_owners(tracker, 0, {2})
        assert_owners(tracker, 1, {2})
        # req b's new pages: 6 (slots 24-27)
        assert_owners(tracker, 6, {2})


class TestRecordDecode(unittest.TestCase):
    def test_records_one_new_slot_per_req(self):
        import torch

        from sglang.srt.managers.schedule_batch import _record_decode_for_tracker

        tracker = build_tracker(num_pages=64, page_size=4)
        out_cache_loc = torch.tensor([12, 28], dtype=torch.int64)
        batch = SimpleNamespace(
            reqs=[
                SimpleNamespace(rid="a", req_pool_idx=1),
                SimpleNamespace(rid="b", req_pool_idx=2),
            ]
        )
        _record_decode_for_tracker(tracker, batch, out_cache_loc)
        assert_owners(tracker, 3, {1})  # slot 12 // 4 = page 3
        assert_owners(tracker, 7, {2})  # slot 28 // 4 = page 7


class TestEndToEndAbortPlumbing(unittest.TestCase):
    def test_corruption_sets_to_finish_only_on_offender(self):
        import torch

        from sglang.srt.managers.schedule_batch import (
            FINISH_ABORT,
            _record_extend_for_tracker,
        )

        tracker = build_tracker(num_pages=64, page_size=4)
        req_a = SimpleNamespace(
            rid="a",
            req_pool_idx=1,
            prefix_indices=torch.tensor([], dtype=torch.int64),
            to_finish=None,
        )
        req_b = SimpleNamespace(
            rid="b",
            req_pool_idx=2,
            prefix_indices=torch.tensor([], dtype=torch.int64),
            to_finish=None,
        )
        out_cache_loc = torch.arange(16, 32, dtype=torch.int64)
        batch = SimpleNamespace(
            reqs=[req_a, req_b],
            prefix_lens_cpu=torch.tensor([0, 0], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([8, 8], dtype=torch.int64),
            out_cache_loc=out_cache_loc,
        )
        _record_extend_for_tracker(tracker, batch, out_cache_loc)
        # Inject a corruption: pretend req_a's req_to_token now references page 99,
        # which is owned by nobody (and out of range — should still be flagged).
        tracker.req_pages[1].add(99)
        violations = tracker.validate_batch(batch)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].req.rid, "a")

        # Mimic the abort wiring from prepare_for_extend.
        from http import HTTPStatus

        for v in violations:
            v.req.to_finish = FINISH_ABORT(
                f"KV integrity violation on pages {v.bad_pages}",
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "KVIntegrityError",
            )

        self.assertIsNotNone(req_a.to_finish)
        self.assertIsInstance(req_a.to_finish, FINISH_ABORT)
        self.assertIn("integrity", str(req_a.to_finish.message).lower())
        self.assertIsNone(req_b.to_finish)


class TestRecordAllocPerReq(unittest.TestCase):
    def test_distributes_slots_by_per_req_length(self):
        import torch

        from sglang.srt.mem_cache.kv_integrity import record_alloc_per_req

        tracker = build_tracker(num_pages=64, page_size=4)
        # 3 reqs: req 0 gets 4 slots (one page), req 1 gets 0 (skipped),
        # req 2 gets 8 slots (two pages).
        out_cache_loc = torch.tensor([20, 21, 22, 23, 36, 37, 38, 39, 40, 41, 42, 43])
        record_alloc_per_req(
            tracker,
            req_pool_indices=torch.tensor([5, 6, 7]),
            lens_per_req=[4, 0, 8],
            out_cache_loc=out_cache_loc,
        )
        assert_owners(tracker, 5, {5})
        assert_owners(tracker, 9, {7})
        assert_owners(tracker, 10, {7})
        self.assertNotIn(6, tracker.req_pages)

    def test_disabled_tracker_is_no_op(self):
        import torch

        from sglang.srt.mem_cache.kv_integrity import _NullTracker, record_alloc_per_req

        tracker = _NullTracker()
        record_alloc_per_req(
            tracker,
            req_pool_indices=torch.tensor([0, 1]),
            lens_per_req=[2, 2],
            out_cache_loc=torch.tensor([10, 11, 12, 13]),
        )
        # No exception; nothing tracked.

    def test_accepts_python_list_for_indices_and_lens(self):
        import torch

        from sglang.srt.mem_cache.kv_integrity import record_alloc_per_req

        tracker = build_tracker(num_pages=64, page_size=4)
        record_alloc_per_req(
            tracker,
            req_pool_indices=[5, 7],
            lens_per_req=[4, 4],
            out_cache_loc=torch.tensor([20, 21, 22, 23, 36, 37, 38, 39]),
        )
        assert_owners(tracker, 5, {5})
        assert_owners(tracker, 9, {7})


class TestSingleReqAlloc(unittest.TestCase):
    def test_on_alloc_single_req_full_page(self):
        import torch

        tracker = build_tracker(num_pages=64, page_size=4)
        kv_loc = torch.arange(20, 24, dtype=torch.int64)
        tracker.on_alloc(req_pool_idx=42, slot_indices=kv_loc)
        assert_owners(tracker, 5, {42})
        self.assertEqual(tracker.req_pages[42], {5})


if __name__ == "__main__":
    unittest.main()
