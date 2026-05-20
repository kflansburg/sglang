from __future__ import annotations

import json
import logging
import threading
import time
from typing import TYPE_CHECKING

import torch

from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class DecodeKVCacheOffloadManager:
    """Manage decode-side KV cache offloading lifecycle and operations."""

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tp_group: torch.distributed.ProcessGroup,
        tree_cache: BasePrefixCache,
        server_args: ServerArgs,
    ) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = server_args.page_size
        self.server_args = server_args
        self.request_counter = 0
        self.tree_cache = tree_cache
        env_stride = envs.SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE.get()
        if env_stride is None or env_stride <= 0:
            self.offload_stride = self.page_size
        else:
            self.offload_stride = max(
                self.page_size, (env_stride // self.page_size) * self.page_size
            )
        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        if isinstance(kv_cache, MHATokenToKVPool):
            self.decode_host_mem_pool = MHATokenToKVPoolHost(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
            )
        elif isinstance(kv_cache, MLATokenToKVPool):
            self.decode_host_mem_pool = MLATokenToKVPoolHost(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
            )
        else:
            raise ValueError("Unsupported KV cache type for decode offload")

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        hicache_storage_backend_extra_config = {}
        if server_args.hicache_storage_backend_extra_config:
            try:
                hicache_storage_backend_extra_config = json.loads(
                    server_args.hicache_storage_backend_extra_config
                )
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid hicache storage backend extra config JSON: {e}"
                )

        self.cache_controller = HiCacheController(
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            mem_pool_host=self.decode_host_mem_pool,
            page_size=self.page_size,
            tp_group=tp_group,
            io_backend=server_args.hicache_io_backend,
            load_cache_event=threading.Event(),
            storage_backend=server_args.hicache_storage_backend,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=hicache_storage_backend_extra_config,
        )

        self.ongoing_offload = {}
        self.ongoing_backup = {}
        self.offloaded_state = {}
        logger.info("Enable offload kv cache for decode side")

    def offload_kv_cache(self, req) -> bool:
        """Offload incremental KV cache for decode side."""

        if self.cache_controller is None or self.decode_host_mem_pool is None:
            return False

        if req.req_pool_idx == -1 or len(req.output_ids) == 0:
            return False

        token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx]
        if token_indices.dim() == 0 or token_indices.numel() == 0:
            return False

        # Prefill side offloads page-aligned origin_input_ids, decode side offloads the incremental part
        all_tokens = req.origin_input_ids + req.output_ids[:-1]
        prefill_offloaded_len = (
            len(req.origin_input_ids) // self.page_size * self.page_size
        )
        state = self.offloaded_state.get(req.rid)
        if state is None:
            prefill_hashes = self._compute_prefix_hash(
                req.origin_input_ids[:prefill_offloaded_len]
            )
            last_prefill_hash = (
                prefill_hashes[-1] if prefill_offloaded_len > 0 else None
            )
            state = OffloadedState(
                prefill_len=prefill_offloaded_len,
                inc_len=0,
                last_hash=last_prefill_hash,
            )
            self.offloaded_state[req.rid] = state
        incremental_total = len(all_tokens) - state.prefill_len
        incremental_new = incremental_total - state.inc_len
        incremental_aligned_len = (
            incremental_new // self.offload_stride * self.offload_stride
        )

        if incremental_aligned_len == 0:
            return False

        # Extract incremental tokens and indices for the newly available chunk
        start = state.prefill_len + state.inc_len
        end = start + incremental_aligned_len
        incremental_tokens = all_tokens[start:end]
        incremental_indices = token_indices[start:end]

        # Early free prefill-offloaded GPU memory.
        #
        # When --disaggregation-decode-enable-radix-cache is on, the slots in
        # [0:cache_protected_len] are owned by the radix tree (TreeNode.value)
        # after cache_unfinished_req inserted the committed prefix; releasing
        # them here would put tree-owned pages back in the allocator pool
        # while the tree still references them, producing cross-request KV
        # pollution when another request prefix-matches the same node and
        # gets recycled slot indices.
        #
        # In radix mode, cache_protected_len is the page-aligned committed
        # length and is typically >= state.prefill_len, so the eligible
        # private region [cache_protected_len:state.prefill_len] is empty
        # and the early free becomes a no-op. The slots actually get
        # released through the radix tree's eviction path later.
        prefix_protected_len = getattr(req, "cache_protected_len", 0)
        if (
            state.prefill_len > 0
            and state.inc_len == 0
            and state.prefill_len > prefix_protected_len
        ):
            self.token_to_kv_pool_allocator.free(
                token_indices[prefix_protected_len : state.prefill_len]
            )

        # Asynchronously offload incremental KV cache from device to host
        self.request_counter += 1
        ack_id = self.request_counter
        host_indices = self.cache_controller.write(
            device_indices=incremental_indices.long(),
            node_id=ack_id,
        )
        if host_indices is None:
            logger.error(f"Not enough host memory for request {req.rid}")
            return False

        self.ongoing_offload[ack_id] = (
            req,
            host_indices,
            incremental_tokens,
            time.time(),
            start,
            end,
        )
        state.inc_len += incremental_aligned_len
        return True

    def check_offload_progress(self):
        """Check the progress of offload from device to host and backup from host to storage."""
        cc = self.cache_controller

        qsizes = torch.tensor(
            [
                len(cc.ack_write_queue),
                cc.ack_backup_queue.qsize(),
            ],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )

        n_write, n_backup = map(int, qsizes.tolist())
        self._check_offload_progress(n_write)
        self._check_backup_progress(n_backup)

    def _check_offload_progress(self, finish_count):
        """Check the progress of offload from device to host."""
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                (
                    req,
                    host_indices,
                    incremental_tokens,
                    start_time,
                    start,
                    end,
                ) = self.ongoing_offload.pop(ack_id)

                if req.finished():
                    self._release_finished_req(req, start)
                else:
                    kv_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, start:end
                    ]
                    self.token_to_kv_pool_allocator.free(kv_indices)

                prior_hash = (
                    self.offloaded_state[req.rid].last_hash
                    if req.rid in self.offloaded_state
                    else None
                )
                last_hash = self._trigger_backup(
                    req, host_indices, incremental_tokens, start_time, prior_hash
                )
                if req.rid in self.offloaded_state:
                    self.offloaded_state[req.rid].last_hash = last_hash
            finish_count -= 1

    def _release_finished_req(self, req: Req, start_offset: int):
        # Guard against double-call. _check_offload_progress and
        # _handle_finished_req → finalize_release_on_finish can both
        # invoke us for the same request when in-flight offload acks
        # straddle request finish (the final offload_kv_cache call returns
        # False when the tail isn't stride-aligned, triggering
        # finalize_release_on_finish immediately; later ack processing
        # would then re-enter _release_finished_req and re-pop the
        # already-popped committed KV, asserting in pop_committed_kv_cache).
        if getattr(req, "kv_committed_freed", False):
            return
        kv_committed_len = req.pop_committed_kv_cache()

        # Skip the radix-tree-protected prefix region [0:cache_protected_len]
        # in any direct allocator frees. With
        # --disaggregation-decode-enable-radix-cache, cache_unfinished_req
        # inserts the committed prefix into the tree and writes the canonical
        # tree-owned slot indices into req_to_token[req.req_pool_idx,
        # :cache_protected_len]. Those slots are shared with TreeNode.value
        # and must NOT be freed through allocator.free here — the tree owns
        # them until eviction. Releasing them directly produces a
        # use-after-free when a concurrent prefix match returns the recycled
        # indices to another request. In chunk-cache mode,
        # cache_protected_len stays at 0 and the original behavior is
        # preserved.
        prefix_protected_len = getattr(req, "cache_protected_len", 0)
        start = max(start_offset, prefix_protected_len)
        end = kv_committed_len
        if start < end:
            # Free the incremental part of the request (NSA-aware)
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, start:end
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)

        # Free over-allocated KV cache slots (e.g. from speculative decoding v2).
        # Without spec v2, start_p == end_p so this is a no-op.
        start_p, end_p = req.pop_overallocated_kv_cache()
        if self.page_size > 1:
            start_p = ceil_align(start_p, self.page_size)
        if start_p < end_p:
            overalloc_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, start_p:end_p
            ]
            self.token_to_kv_pool_allocator.free(overalloc_indices)

        self.req_to_token_pool.free(req)

        # Release the radix-tree lock on the matched prefix.
        #
        # The previous implementation directly mutated
        # tree_cache.protected_size_ -= len(req.prefix_indices), which (a)
        # did not actually decrement the lock_ref on req.last_node, leaking
        # the lock and the prefix node permanently — the tree could never
        # evict its slots — and (b) used the matched-prefix length instead
        # of the cumulative len(node.key) of the locked path, producing a
        # drifting protected_size_ that eventually trips the runtime leak
        # detector with "ValueError: pool memory leak detected!
        # protected=-N".
        #
        # dec_lock_ref(req.last_node) is symmetric to the inc_lock_ref calls
        # made by _match_prefix_and_lock at admission and cache_unfinished_req
        # at each scheduler step. For ChunkCache (radix disabled),
        # dec_lock_ref is a no-op, so the chunk-cache path is unchanged.
        if req.last_node is not None:
            self.tree_cache.dec_lock_ref(req.last_node)

        if req.rid in self.offloaded_state:
            del self.offloaded_state[req.rid]

    def _check_backup_progress(self, finish_count):
        """Check the progress of backup from host to storage."""
        for _ in range(finish_count):
            storage_operation = self.cache_controller.ack_backup_queue.get()
            ack_id = storage_operation.id
            req_id, host_indices, start_time = self.ongoing_backup.pop(ack_id)

            # Release host memory
            self.decode_host_mem_pool.free(host_indices)

            logger.debug(
                f"Finished backup request {req_id}, free host memory, len:{len(host_indices)}, cost time:{time.time() - start_time:.2f} seconds."
            )

    def _trigger_backup(
        self, req, host_indices, incremental_tokens, start_time, prior_hash
    ):
        """Trigger async backup from host to storage."""
        page_hashes = self._compute_prefix_hash(incremental_tokens, prior_hash)
        ack_id = self.cache_controller.write_storage(
            host_indices,
            incremental_tokens,
            hash_value=page_hashes,
        )
        self.ongoing_backup[ack_id] = (req.rid, host_indices, start_time)
        return page_hashes[-1] if len(page_hashes) > 0 else prior_hash

    def _compute_prefix_hash(self, tokens, prior_hash=""):
        page_hashes = []
        last_hash = prior_hash
        for offset in range(0, len(tokens), self.page_size):
            page_tokens = tokens[offset : offset + self.page_size]
            last_hash = self.cache_controller.get_hash_str(page_tokens, last_hash)
            page_hashes.append(last_hash)
        return page_hashes

    def finalize_release_on_finish(self, req: Req):
        """Free any remaining tail KV that was not offloaded due to non-aligned length."""
        if req.req_pool_idx == -1:
            return
        state = self.offloaded_state.get(req.rid)
        if state is None:
            prefill_len = len(req.origin_input_ids) // self.page_size * self.page_size
            inc_len = 0
        else:
            prefill_len = state.prefill_len
            inc_len = state.inc_len
        # If no incremental offload ever happened, the prefill-aligned part
        # was never freed by offload_kv_cache's early-free path.
        #
        # In radix mode (cache_protected_len > 0), slots in
        # [0:cache_protected_len] are owned by the radix tree and must NOT
        # be freed directly here — see offload_kv_cache for full rationale.
        # We only free the region between cache_protected_len and
        # state.prefill_len which is request-private. cache_protected_len is
        # typically >= state.prefill_len in radix mode, so this is normally
        # a no-op; the request's tail is released through
        # _release_finished_req → dec_lock_ref → tree eviction.
        prefix_protected_len = getattr(req, "cache_protected_len", 0)
        if prefill_len > 0 and inc_len == 0 and prefill_len > prefix_protected_len:
            token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx]
            self.token_to_kv_pool_allocator.free(
                token_indices[prefix_protected_len:prefill_len]
            )
            logger.info(
                f"Finalize release: freed prefill-aligned KV for req {req.rid}, "
                f"len:{prefill_len - prefix_protected_len}"
            )
        start_offset = prefill_len + inc_len
        self._release_finished_req(req, start_offset)
