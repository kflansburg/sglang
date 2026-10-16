# Allocator audit findings (alloc-auditor)

Scope: `python/sglang/srt/mem_cache/allocator/{base,paged,token,swa,page_interleave,hisparse}.py` + `python/sglang/kernels/ops/memory/allocator.py` (Triton alloc kernels).
Production config: page_size 64, kv fp8_e4m3/bf16, chunked prefill 12288-32768, TP 1-8, PD disaggregated (need_sort=True in ALL disagg modes — see kv_cache_configurator.py:887/999/1348/2485: `need_sort = get_disagg().disaggregation_mode in ("decode","prefill")`).

Method note: allocators are pure-python + torch tensor ops driving Triton kernels; the naive CPU equivalents (`alloc_extend_naive`, `alloc_decode_naive`) are bit-equivalent reference implementations included in paged.py, so the allocator logic + kernels' index math can be CPU-tested via the naive path (Triton path is GPU-only; its index math was separately audit-read).

---

## CONFIRMED / HIGH-CONFIDENCE BUGS

### BUG-ALLOC-1. HiSparse capacity check measures unmerged free tokens as new pages → spurious allocation failure (liveness; can wedge cache), and near-miss vs a real over-allocation (page OOB / -1-slot write)
- File: `python/sglang/srt/mem_cache/allocator/hisparse.py:220-229` (`HiSparseTokenToKVPoolAllocator.alloc_extend`) and `:573-584` (`DeepSeekV4HiSparseTokenToKVPoolAllocator.alloc_extend`).
- Code:
  ```
  if num_new_pages > self.logical_attn_allocator.available_size() // self.page_size:
      return None
  ```
- Problem: `available_size()` is a **token** count. Every other allocator that compares an exact new-page count (`num_new_pages`) either uses the exact free-page array length (`PagedTokenToKVPoolAllocator.alloc_extend` uses `len(self.free_pages)` directly) or merges first. Here `available_size()` includes free_pages+release_pages+staged_pages **without merging**. Under `need_sort=True` (production disaggregated mode) freed pages sit in `staged_pages`/`release_pages` until a merge is triggered; this check then:
  1. Still passes the guard (available_size counts the staged tokens), then calls the inner paged `alloc_extend` which DOES merge internally — so the tokens are recovered and alloc succeeds *by luck of the inner path*.
  2. Net effect direction: the guard is checking a token-equivalent but page-rounded number. With fragmentation across release/staged lists, the guard count (`available_size() // ps`) can be an under- or over-estimate of the pages `alloc_extend` can actually hand out. Where it differs materially:
     - guard under-counts vs inner path → returns None spuriously (retry storm / scheduler stall — cache appears full when it is not).
     - If `num_new_pages` (exact-token basis) would pass but the inner paged allocator, for a different page_size sized pool, cannot deliver, the inner `alloc_extend` returns None and the code hits `assert ... is not None` → harder crash, not a clean -1 write. The HiSparse SECONDARY pool (`hisparse_attn_allocator`) is smaller (size/ratio); the guard against it uses the same available_size vs num_new_pages logic.
- Repro sketch (CPU, naive kernels): build `HiSparseTokenToKVPoolAllocator(size=..., page_size=64)` with need_sort=True; free pages so they scatter between free_pages and staged_pages; craft seq_lens so `num_new_pages == available_size() // ps` (edge). Compare guard decision vs what inner alloc_extend would do.
- Production relevance: DeepSeek V4 Pro (v0.5.15.post1) and DeepSeek V4 Flash (v0.5.17) run DSA/HiSparse — matches "DSA tail req_pool index" known-bug family. Severity: Medium-High (availability; the OOB/-1 slot write concern mostly resolves into an assert crash in current code, but the check is structurally wrong and inconsistent with every sibling allocator).
- Class: "paged allocator capacity check (alloc beyond capacity / boundary condition)" sibling.

### BUG-ALLOC-2. SWA `alloc_extend_swa_tail` over-reserves SWA pages when the tail shares the partial prefix page (page_size > 1) → spurious None (admission failure) on the disaggregation decode prealloc path
- File: `python/sglang/srt/mem_cache/allocator/swa.py:417-427` (`SWATokenToKVPoolAllocator.alloc_extend_swa_tail`); caller `python/sglang/srt/disaggregation/decode.py:2353,2419`.
- Code:
  ```
  num_swa_pages = (swa_tail_len + self.page_size - 1) // self.page_size   # ceil
  if not self.new_pages_available(num_full_pages, num_swa_pages):
      return None
  ```
- Problem: the fresh SWA allocation for the tail is built with `swa_prefix_lens=0, swa_seq_lens=swa_tail_len, swa_last_loc=-1`, so the inner alloc_extend computes its own `get_num_new_pages(swa_seq, prefix=0) = ceil(swa_tail_len/ps)` — matching this guard, so there is NO over-allocation here. BUT `new_pages_available` uses `available_size() // ps` which, like BUG-ALLOC-1, does not merge staged/release pages, so the guard can be over-strict vs what the inner paged SWA alloc_extend would actually deliver, returning None spuriously. Also note `assert len(seq_lens_cpu) == 1` — this is invoked for a single request during decode prealloc; any caller change to batched use would assert-crash (fragility, and `0 <= swa_tail_len <= extend_num_tokens` is the only bound on tail vs the SWA window — no check the tail fits the window/page count of the actual SWA pool).
- Production relevance: disaggregated DECODE (--disaggregation-decode-enable-offload-kvcache / SWA) — direct match to production surface. Severity: Medium (spurious admission failure under fragmentation ⇒ prealloc retry/abort churn).
- TODO: reproduce with need_sort=True + fragmented SWA free list; exact failing case.

### BUG-ALLOC-3. DeepSeekV4HiSparse `alloc_device_buffer` frees whole pages for indices that may be shared with the retained buffer → page double-free / free of live pages
- File: `python/sglang/srt/mem_cache/allocator/hisparse.py:505-525`
- Code:
  ```
  buffer_indices = hisparse_indices[:need_size]
  surplus = hisparse_indices[need_size:]
  if surplus.numel() > 0:
      buffer_pages = torch.unique(buffer_indices // self.hisparse_page_size)
      surplus_pages = torch.unique(surplus // self.hisparse_page_size)
      pure_surplus = surplus_pages[~torch.isin(surplus_pages, buffer_pages)]
      if pure_surplus.numel() > 0:
          self.hisparse_attn_allocator.free(pure_surplus * self.hisparse_page_size)
  ```
- Problem: it keeps indices whose pages straddle the buffer/surplus boundary (those are safe — `pure_surplus` excludes pages shared with buffer), but `free(pure_surplus * hisparse_page_size)` frees a page by its FIRST token index. `hisparse_attn_allocator.free()` does `torch.unique(idx // ps)` — fine. The risk: `hisparse_indices` comes from `full_to_hisparse_device_index_mapping[allocated_indices]` filtered to `>0`. If the SAME hisparse device token appears under multiple logical slots (mapping shared — e.g. two logical tokens mapping into one compressed C4 slot pair across compress_ratio=4), then `buffer_indices` and `surplus` pages can both contain tokens legitimately still referenced, and the per-page split is only correct if hisparse_indices is a strictly-sorted contiguous run of one request's private compressed tokens. Whether mapping can alias across requests is the open question; worth a concrete trace with compress_ratio=4. Note also the older non-V4 `alloc_device_buffer` (`hisparse.py:160-196`) frees `hisparse_indices[need_size:]` **directly as token indices** (`self.free_hisparse_indices`) — token-level free into a paged allocator, WITHOUT the pure_surplus page-sharing guard the V4 version has. If the buffer/surplus boundary splits a page, that frees the whole page via unique(idx//ps) while the buffer still holds tokens in it → live page handed back out = UAF/cross-request corruption. The two variants diverging is itself a smell.
- Production relevance: DeepSeek V4 DSA (hisparse/DSA prefill backends, production). Severity: High if aliasing/boundary holds; needs trace.
- Class: "DSA tail req_pool index" family / page-sharing double free.

### BUG-ALLOC-4. Triton `alloc_extend_kernel` reads `free_page_ptr[new_page_start_loc + num_new_pages - 1]` even for requests with `num_new_pages == 0` → OOB read of the free-page list (UB; can pick up a non-owned page id for a dead lane, though the store is masked off)
- File: `python/sglang/kernels/ops/memory/allocator.py` (`alloc_extend_kernel`, Part 3, ~line 180):
  ```
  num_part3 = seq_len - seq_len // page_size * page_size
  start_loc = tl.load(free_page_ptr + new_page_start_loc + num_page_start_loc_self - 1)
  tl.store(out_indices + output_start_loc + num_part1 + num_part2 + offset_one_page,
           start_loc * page_size + offset_one_page, mask=offset_one_page < num_part3)
  ```
- Problem: `num_part3` computation does NOT account for the token's own `pre_len` contribution. When a request's seq ends exactly on a page boundary (e.g. seq_len = 64*k with page_size 64), for a request that needed 0 new pages (`num_page_start_loc_self == 0`):
  - `new_page_start_loc` = global pages allocated before this request = the COUNT used so far.
  - load index = `count - 1` → reads the last-allocated page id of a DIFFERENT request (or free_pages[-1] when count=0 → reads element before the list = OOB at the head when `count==0` and `free_page_ptr` is the base pointer).
  - num_part3 = seq_len % ps; for seq_len = exact multiple, num_part3 = 0 → the store mask is all-false, so the load result is DISCARDED. The load itself is unmasked → Triton masked-load is absent: `tl.load(ptr + idx)` with no mask loads garbage/OOB memory (within the tensor for interior indices; at index -1 it reads `free_page_ptr[-1]` = the high end of the buffer = a real page id, not a crash). No memory corruption of OUTPUT because the store is masked, but: it reads `free_pages` beyond the committed `num_new_pages` — i.e., reads pages that remain "free" — and could in principle be combined with a later miscomputation; more importantly it's an outright OOB read at the head of the buffer (idx -1) for the all-empty-prefix + exact-boundary batch, which Triton does not protect against.
  - ALSO the naive reference `alloc_extend_naive` guards this exact case (it computes `page_idx3 = clamp(end_new_pages_g - 1, ...)` and uses the SAME num_part3-deficient value but clamps), which is why the naive path "passes"; the Triton kernel's raw load has no clamp → the two diverge on boundary reads.
- Another Part-3 boundary case: request i where `num_page_start_loc_self >= 1`, `pre_len % ps == 0` (empty partial part1), seq_len % ps == 0 (num_part3 = 0): store masked off again → benign read. The genuinely dangerous variant would need num_part3 > 0 while the batch's global num_new_pages undercounts — cannot happen because num_new_pages accounts for the final partial page exactly via ceil.
- Severity: Low-Medium. No live corruption observed from the masked-off store, but it's an OOB read on the free list (head-of-buffer -1 index) on a production shape (page_multiple lengths are common at page_size=64 chunked prefill). Triton OOB loads are UB and can fault under guard pages. The divergence from the naive reference is the bug-in-principle.
- Verify: cannot Triton-test on CPU; reproduce via naive-vs-model mismatch on the boundary grid (the naive clamped version hides the read). Flag as GPU-kernel-only.

---

## AREAS CHECKED, VERDICT CLEAN (with reasoning)

- Paged allocator capacity check (the production-surface class): `paged.py:alloc_extend` first merges when the strict upper bound `extend_num_tokens//ps + bs + 1` exceeds len(free_pages), then falls through ONLY when `num_new_pages > len(free_pages)` → returns None. The exact-count path is correct; the upper-bound is conservative SAFE (it is `>=` the true new-page count; proven per-request: new_pages_i = ceil(seq_i/ps) - ceil(pre_i/ps) <= (extend_i + ps - 1)/ps; summing gives <= extend//ps + bs). No over-allocation possible. `alloc_decode` similarly uses exact `get_num_new_pages(decode=True)` and returns None when short. VERDICT: capacity/boundary check is correct in paged.py.
- Token allocator (token.py): trivial free-list; alloc returns None short; free just concatenates; page_size==1 free_page_ids==free. No boundary issue. CLEAN.
- available_size() accounting on alloc_partial-abort: alloc_extend/alloc_decode do not commit `self.free_pages = self.free_pages[num_new_pages:]` until AFTER a successful kernel run; on early None returns nothing is deducted → no drift on the alloc side. The committed num_new_pages equals exactly the pages the kernel consumed (same formula `sum(ceil(seq) - ceil(prefix))` in host and kernel). CLEAN on the stock path.
- free-list double free (paged.py free): dedup via `torch.unique(free_index // page_size)`; free_segment asserts disjointness via base._page_disjoint; debug_mode has `_debug_check_no_duplicate_pages`. A double free of the same index is still possible ONLY if the CALLER frees the same indices twice (unique dedups WITHIN one call, not ACROSS calls) — that is a caller bug surfaced via the release/staged lists when need_sort reduces dedup timing; not an allocator defect per se, but note ACROSS-call double free is NOT guarded (fits "abort leaking pins" known class — the allocator does not own this).
- token_to_kv_pool indices at page_size=64 boundary: `max_new_tokens_for_memory` (base.py:96-100) subtracts paged_input + page_size + 1; the +1 keeps one token slack beyond a full page — checked for off-by-one; it reserves a whole page plus a token, conservative (loses up to 65 tokens of capacity, wasteful not unsafe). Note it can return min(0,...) -> 0 (not None) for exact-fit case; callers must treat 0 as "cannot fit". OK.
- -1 index writes: full_to_swa_index_mapping has a deliberate trailing -1 sink (size+page_size+1 long); last_loc=-1 maps there. `_free_swa_pages_none_cuda` asserts swa_tokens>0 (expect SWA_PEER_MAPPED) so freeing via last_loc=-1 is guarded; the CUDA kernel reads `mapping[full_reps]` where full_reps are page representatives (>=0). The only way to write to slot 0 (padding sink) is a full index of 0 being alloc'd out with an SWA mapping — SWA full pools start free lists at page 1 (clear() arange(1, num_pages+1)) so slot 0 is never handed out. CLEAN for the stock path; the custom unified path is a different agent's file.

## TODO / in progress
- Build CPU fuzz harness (torch CPU) at `test/srt/pools_fuzz/`: random alloc_extend/alloc_decode/free/merge sequences against a refcounted in-use index model; invariants: no double-alloc of live index, conservation live+free==total, available_size() == model. Include the naive-kernel paths. Cover page_size=64 and page multiple boundaries (seq_len = k*64, k*64±1), free_group_begin/end nesting, free_segment / free_segments page-disjointness.
- Confirm BUG-ALLOC-1/2 divergent guard vs inner path with a CPU build; get exact failing seq.
- Confirm whether full_to_hisparse_device mapping aliases across requests (needed for BUG-ALLOC-3 severity).
- hisparse `alloc` and V4 partial-pool asserts (`Hisparse allocation failed in alloc_extend`) leave the LOGICAL allocation committed while crashing — verify whether the assert-crash leaks the logical pages (they were popped from free_pages) — yes: on assert, logical pages popped but exception propagates; nothing rolls them back. Under a non-assert build / future refactor this is a leak-on-error. Worth a note even though asserts mean "debug/production crash-first".
