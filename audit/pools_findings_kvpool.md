# kvpool-auditor findings

Scope: memory_pool.py (MHATokenToKVPool, MLATokenToKVPool + subclasses, ReqToTokenPool,
HybridLinearKVPool core), kv_vmm_backing.py, kv_cache_dtype.py, events.py,
layout/ (paged_view, token_major, fused_draft), device_pool_info.py, utils.py,
common.py, cache_init_params.py.
Production: prefill/decode disagg, --page-size 64, fp8_e4m3 (mostly)/bf16,
chunked-prefill 12288-32768, TP 1-8, MLA + GQA models.
Branch: agent/kvpool-auditor.

Format: file:line | severity | trigger | production relevance | status

Boundary notes: unified pool/sub-pools/configurator/sizing = unified-auditor;
allocator/* free-lists = alloc-auditor; SWA = swa-auditor; DSA index pool internals
= dsa-auditor; spec draft KV write callers = spec-auditor. I audit the pool classes
themselves + layout/view/VMM backing + fp8 store math.

---

## CHECKED — NO BUG (verdict + evidence)

1. layout/ view/stride arithmetic — CLEAN.
   CPU fuzz at page_size=64: paged_view, paged_kv_view (slot-major + HND permute),
   paged_row_view, build_dense_views (multi-layer entry), DenseEntryLayout.validate
   (overlap + OOB-part rejection), build_mamba_entry_views (conv/temporal regions).
   No off-by-one, no cross-layer/page aliasing. Test: test/srt/pools_fuzz/test_layout_arithmetic.py (all pass).
   entry_bytes/ROW_ALIGN/vector-store alignment asserts enforced in validate().

2. KvBufferDesc span math <=> get_contiguous_buf_infos consistency — CLEAN.
   MHA NHD: final_span_bytes(size,ps) == (size+ps)*row_bytes; MLA nbytes matches.
   item_len_bytes(ps) = (ps//tokens_per_row)*row_bytes matches MLA kv_item_lens.
   HND tokens_per_row=ps -> whole-page rows accounted correctly.

3. KvVmmBacking monotonic commit/backing — CLEAN.
   commit_range is per-offset monotonic (prev tracked, maps only the delta);
   every buffer has a distinct offset (separate tensors); finalize() is monotonic
   (never unmaps); finalize-guard unified_hybrid_swa asserts ordering w/ resize.

## DEFENSE-IN-DEPTH GAPS (not directly triggerable from pool code; note for rollup)

G1. memory_pool store paths trust `loc` >= 0; the only guards are
   `maybe_detect_oob` (gated behind SGLANG_ENABLE_ASYNC_ASSERT, default OFF) and
   the JIT store_cache kernel's C `assert(index >= 0 && index < size_limit)`
   (kvcache.cuh:75) which is a no-op in release/NDEBUG builds.
   - MHA eager fallback: k_cache[indices]=k with -1 index wraps to the LAST row
     (torch negative indexing) -> silent cross-slot write.
   - MLA set_kv_buffer/set_mla_kv_buffer: self.kv_buffer[..][loc] index_put; -1 wraps.
   - copy_all_layer_kv_cache_tiled (cache_move.py:148): src/tgt_ptr = base + loc*stride
     with no bounds mask at all -> -1/OOB loc is a raw OOB read/write.
   Reaching -1 locs requires an allocator/upstream bug (production allocators return
   valid ids and pad with slot 0). Severity: latent; a single -1 sink = slot-0(OOB)
   corruption. Recommend: enable a cheap runtime clamp or drop the gating so the
   store/&copy kernels bound-check in release.

G2. set_kv_buffer_prefix_valid (tiled fp8/non-fp8 triton) has NO check that
   commit_lens[i] <= loc_2d.shape[1] (only batch dim is validated, memory_pool.py:3241).
   If a caller passes commit_len > block_size, kernel reads loc_2d OOB (next row's
   loc) and writes KV to a wrong slot.
   In-tree callers are spec-decode (dspark.py:713, dflash_worker_v2.py). Handed to
   spec-auditor to verify commit_lens<=page_size in their construction.

G3. dtype!=store_dtype eager paths use torch index_put which wraps negatives; see G1.

4. KvBufferDesc final_span/item_len span invariants — CLEAN (CPU fuzz, page 64,
   NHD tpr=1 + HND tpr=page, edge token counts). final_span always covers the
   trailing padding page AND is exactly ceil((size+ps)/tpr) rows; item_len tiles
   final_span exactly for HND; slot-row item_len == ps*row_bytes. No transfer
   truncation/over-run. Test: test/srt/pools_fuzz/test_kv_buffer_desc_span.py.

5. events.py KVCacheEventRecorder arithmetic — CLEAN (static).
   record_store/record_remove iterate `range(0, logical_len, page_size)`, i.e.
   exactly ceil(logical_len/ps) pages, matching the per-page hash list from the
   native hasher (same self.page_size). Root node (len 0) -> no pages, no writes.
   Bigram loop reads raw[j+1] up to j=logical_len-1 -> raw[logical_len], valid per
   _native_hash_input (requires len(raw)>=logical_len+1 for bigram). No OOB.
   NOTE: relies on radix keeping node.key/hash_value page_size-consistent (radix-lead).

6. ReqToTokenPool alloc/free invariants — CLEAN (static).
   alloc_rows pops from tail O(need), no duplicates; free() asserts holds_kv;
   free_mamba_cache asserts mamba_index is not None (double-free guarded);
   generation counter monotonic (never re-zeroed on clear); +1 pad row at index 0
   for cuda-graph padded batches; alloc() reuses rows only when kv_allocated_len>0.

7. HybridLinearKVPool layer routing — CLEAN (static).
   _transfer_full_attention_id raises ValueError on a non-full layer (no silent OOB);
   get/set/move/cpu_copy delegate to full_kv_pool (MHA/MLA/DSA) or mamba pool; the
   set_kv_buffer write loc uses full_loc-if-present-else-loc, both physical-marked.

## DEFERRED / OUT-OF-SCOPE (confirmed not in my production surface)

- MHATokenToKVPoolFP4 / MLATokenToKVPoolFP4 / NVFP4 flashinfer dequant workspace
  (lines 3046-3226, 3555-3708, 5005-5150): nvfp4, not deployed.
  (Aside: FP4 set_kv_buffer when cache_k.dtype==self.dtype leaves cache_k_fp4_sf
   undefined -> NameError, not memory-unsafe; fp4 never reached in prod.)
- MHATokenToKVPoolMXFP8 (3781-4148): mxfp8 + page_size=128 + FA4, not deployed.
  _read_sf_interleaved ipos math only exercised under page 128; not fuzzed here.
- PageMajorMHATokenToKVPool: non-constructible (ServerArgs rejects at boot).
- DSA index-k / compress-tail internals: DSATokenToKVPool (dsa-auditor owns).
- set_kv_buffer_prefix_valid commit_len>block_size caller invariant: spec-auditor.

## RESIDUAL TOP CONCERN (for the lead's rollup)

The single biggest latent memory-safety gap in these files is G1/G2: the entire
pool store/move surface trusts that `loc` is a valid, non-negative slot id, and
the only bounds checks are behind debug-only gates (SGLANG_ENABLE_ASYNC_ASSERT env
flag, and C `assert` stripped from release builds). This is by design (allocator
owns id validity) and there is a documented padded slot-0 sink, but it means any
single -1/OOB id escaping an allocator or disagg caller becomes silent cross-slot
corruption rather than a clean fault. A cheap always-on clamp/bounds mask in
copy_all_layer_kv_cache_tiled and the store kernel would convert a whole bug class
into a detectable assert — flagged as a hardening recommendation, not a confirmed
repro.
