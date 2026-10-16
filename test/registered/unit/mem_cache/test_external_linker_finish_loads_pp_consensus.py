"""C6 regression: PP-consistency of the external-linker finish verdict.

Without the pp_group MIN-reduce, a failed middle-stage layer load was silent
on later PP stages: their own layers landed, they committed/sampled garbage
hidden states, and at least one garbage token escaped to the client before the
failing stage's abort reconciled.

These tests simulate per-stage ranks with stubbed cached collectives and verify:
1. The pp all-reduce is entered on EVERY rank for EVERY batch (uniform
   collective ordering), even when a rank has no linkered loads.
2. A failure on one stage reaches every stage (global MIN), and ranks agree on
   the set of aborted rids based on their own inflight list + shared verdict.
3. complete() after fail() is a no-op (does not mask the failure with
   InvalidStateError), and fail-after-complete stays put.
"""

from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock

import torch
import pytest

from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    LayerWiseLoadCounter,
    UnifiedCacheLinkerWrapper,
)


class StubCache:
    """Records collective calls and replays scripted per-rank verdicts."""

    def __init__(self, name: str):
        self.name = name
        self.calls: List[str] = []
        self.broker = None  # shared across ranks

    def _all_reduce_attn_groups(self, tensor: torch.Tensor, op) -> None:
        self.calls.append("attn")
        self.broker.attn_reduce(self.name, tensor)

    def _all_reduce_pp_group(self, tensor: torch.Tensor, op) -> None:
        self.calls.append("pp")
        self.broker.pp_reduce(self.name, tensor)


class CollectiveBroker:
    """MIN-replays across scripted rank inputs, like a real group would."""

    def __init__(self, rank_names, attn_reduce=None, pp_reduce=None):
        self.attn_reduce = attn_reduce or (lambda name, t: None)
        self.pp_reduce = pp_reduce or (lambda name, t: None)


def make_wrapper(name, rids, landed_local, broker):
    wrapper = object.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = StubCache(name)
    wrapper.cache.broker = broker
    cache_linker = MagicMock()
    cache_linker.finish_layer_wise_loading = MagicMock(return_value=landed_local)
    wrapper.cache_linker = cache_linker
    wrapper.inflight_load_rids = list(rids)
    wrapper.inflight_load_index = (3 if rids else -1)
    wrapper.private_load_indices = {rid: rid for rid in rids}
    return wrapper


class TestPPConsensus:
    def test_every_rank_enters_pp_reduce_every_batch(self):
        """Uniform collective ordering: a rank with no linkered loads must still
        join the pp MIN exactly once (and must NOT join the attn collectives)."""
        broker = CollectiveBroker(["stage0", "stage1"])
        loaded = make_wrapper("stage0", ["r1"], True, broker)
        idle = make_wrapper("stage1", [], True, broker)
        assert loaded.finish_loads() == []
        assert idle.finish_loads() == []
        assert loaded.cache.calls == ["attn", "pp"]
        assert idle.cache.calls == ["pp"]
        idle.cache_linker.finish_layer_wise_loading.assert_not_called()

    def test_middle_stage_failure_reaches_all_stages(self):
        """Stage1 fails its layers; stage2 lands fine. With a pp MIN over
        [1, 0, 1], every stage must see 0 and stage2 must abort its rids."""
        scripted = {"stage0": 1, "stage1": 0, "stage2": 1}

        # Each stage reduces over ITS OWN attn group first: no-op here (attn
        # agreement is per-group and already covered by the primary verdict).
        def pp_reduce(name, tensor):
            tensor[0] = min(scripted.values())

        broker = CollectiveBroker(list(scripted), pp_reduce=pp_reduce)
        wrappers = {
            name: make_wrapper(name, ["r1"], scripted[name] == 1, broker)
            for name in scripted
        }
        results = {name: w.finish_loads() for name, w in wrappers.items()}
        # Without the pp reduce, stage2 would have returned [] (its own layers
        # landed) and streamed garbage; with it every stage aborts the req.
        assert results == {"stage0": ["r1"], "stage1": ["r1"], "stage2": ["r1"]}
        for w in wrappers.values():
            assert w.inflight_load_rids == []
            assert w.inflight_load_index == -1
            assert w.private_load_indices == {}

    def test_all_stages_land_yields_no_abort(self):
        broker = CollectiveBroker(["s0", "s1"])
        wrappers = {n: make_wrapper(n, ["r1", "r2"], True, broker) for n in ("s0", "s1")}
        assert {n: w.finish_loads() for n, w in wrappers.items()} == {
            "s0": [],
            "s1": [],
        }


class TestCounterHygiene:
    def test_complete_after_fail_is_noop(self):
        counter = LayerWiseLoadCounter(num_layers=2)
        index = counter.update_producer()
        counter.set_consumer(index)
        counter.fail(index, RuntimeError("mooncake short read"))
        # Completing a failed layer must not clobber the recorded exception.
        counter.complete(index, 0)
        counter.complete(index, 1)
        assert counter.finish(index) is False

    def test_fail_after_partial_complete(self):
        counter = LayerWiseLoadCounter(num_layers=3)
        index = counter.update_producer()
        counter.set_consumer(index)
        counter.complete(index, 0)
        counter.fail(index, RuntimeError("rank1 io"))
        counter.complete(index, 1)  # no-op; failure already recorded
        counter.complete(index, 2)
        assert counter.finish(index) is False

    def test_complete_unknown_index_no_longer_raises(self):
        counter = LayerWiseLoadCounter(num_layers=1)
        counter.complete(99, 0)  # must not KeyError


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
