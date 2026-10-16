# Configurator sizing-math audit findings (configurator-auditor)

Scope: kv_cache_configurator.py, pool_configurator.py, allocation_sizing.py.
Production configs: page 64, fp8_e4m3/bf16, DP 1-8, TP 1-8, SWA ratio 0.075-0.1, mamba extra_buffer, DSA, EAGLE/DSPARK, disagg, HiCache.

## Inspected, verdicts

### kv_cache_configurator.py
- `calculate_mla_kv_cache_dim` (L3083-3146): fp8 DSA adds scale storage (`kv_lora_rank//128*4`) + rope in bf16 for non-trtllm backends. Users of this (DefaultPoolConfigurator._compute_cell_size MLA branch, HybridSWA MLA branch) multiply the WHOLE dim by `kv_size` (fp8=1). rope term already multiplied by `rope_storage_dtype.itemsize` inside; `*1` is identity -> consistent ONLY because fp8 element size == 1. If kv_cache_dtype==bf16 with a DSA scaled layout, dim would be under-counted? bf16 DSA returns kv_lora+rope (raw layout) and pool stores bf16 — consistent. VERDICT: clean for fp8_e4m3/bf16 prod dtypes; note fragility is benign since scaled layout only exists in fp8 mode.
- `_profile_available_bytes` (L2668-2738): budget = free - slack - mm_reservation; mamba subtracted via _handle_max_mamba_cache. int() truncation at end loses <1GiB (conservative). Clean-ish; relies on get_available_gpu_memory accuracy at this point (weights loaded? pre_model_load_memory is pre-load free).
- `_apply_token_constraints` (L2768): min with user cap, PP all-reduce MIN. Never raises capacity (asserted in config_from_budget). Clean.
- `config_from_budget` (L2870): on token-cap re-derive calls calculate_pool_sizes_from_max_tokens; comment says capacity must not rise; assert present. Note: re-derived config recomputes swa/full from max_tokens WITHOUT re-subtracting fixed pools (mamba bytes etc.) — see mamba note below.
- `_handle_max_mamba_cache` (L2907-3081): GLB units. See findings F1/F2 below.
- `resolve_max_num_reqs` (L2797-2850): estimated = token_capacity/context_len*512 clamped [2048,4096]; explicit max_running_requests: `requested_per_worker = max_num_reqs // attn_dp_size` — floor division. Also capped by mamba ratio. See F4 (DP floor) — check scheduler side for matching division.
- `_calculate_mamba_ratio` (L2740): returns 1 if radix disabled; else base+extra slots. Used both for state-slot multiplier and budget divisor. See F2.
- `logical_token_capacity` (L366): DCP widening `max_total * attn_dcp_size // loc_space_scale`; draft skips widening (loc_space_scale=dcp). `hybrid_swa_token_capacity` takes min(allocator.size_full, max_total_tokens) under unified memory.
- `_derive_pool_sizes` (L488-543): draft worker zeroes c4/c128/state sizes while keeping dtypes — matches instruction note; verify no draft path allocates c4/c128 pools. (Confirmed: `_build_dsa_kv_pool`/`_build_dsv4_kv_pool` only reached for non-draft? NEED TO CHECK call path.)

### pool_configurator.py
- `DefaultPoolConfigurator.calculate_pool_sizes` (L614): `available_bytes // cell_size` then page-floor. Floor = conservative. Clean.
- `_compute_cell_size` MLA (L352-394): dim*layers*kv_size + indexer(head_dim scale terms). mxfp8 MHA adds `//scale_block_size` floored scale bytes: pool ACTUAL scale allocation must be checked (kv pool mxfp8 branch). fp8_e4m3 (non-mxfp8, non-DSA) has NO scale storage in pool — fp8_e4m3 MHA/MLA storage is raw, correct.
- `HybridSWAPoolConfigurator._recompute_cell_size` (L760): cell_size = full_cell + ratio*swa_cell uses FLOAT ratio then L881 `int(available_bytes // cell_size)` truncates float; then _solve_pool_sizes page-floors full and floors `int(full*ratio)` for swa. Actual bytes used = full_cell*full + swa_cell*swa <= budget? full <= budget/cell_size. bytes = full_cell*full + swa_cell*floor(full*ratio) <= full_cell*full + swa_cell*ratio*full = cell_size*full <= budget. SAFE (float monotonicity edge: floor after float mult is >= exact? float rounding of cell_size could round UP making the bound looser but computed `full` smaller — direction safe. If cell_size rounds DOWN, full could be +1 token over real budget; per-token cell sizes are large ints so float rel error ~1e-16 vs int truncation magnitude — negligible). VERDICT: clean.
- `_max_unified_full_tokens` (L801): bisect over pages with draft virtual-span overhang accounted (ceil_align(-1 span)+page). Looks carefully done — draft overhang charged. VERDICT: clean on read.

## FINDINGS (ranked)

(to be filled)

## FINDINGS (ranked, final)

### F1 (LOW, systematic budget-direction): KV pools allocate `size + page_size` rows; configurator budgets only `size`
- memory_pool.py `_kv_buffer_shapes` (L2518-2529), MLA `_create_buffers` (L~4738), IndexKeyCache (index_key_cache.py L18: `num_pages = (buf + page + 1)//page` — one extra page) all allocate one extra page of rows for the dummy/padded-token sink. Cell-math budget (`DefaultPoolConfigurator.calculate_pool_sizes` L614: `available_bytes // cell_size`) charges only `max_total` tokens. Runtime allocates `(1 + page/max_total)x` of budget (page64 & 1M tokens ≈ +0.006%). Deliberate padding-page pattern; absorbed by mem_fraction slack. Not corruption; worst case a marginal OOM at extreme mem_fraction (slack also grows then). VERDICT: benign, note for budget truthfulness. Same +1-page pattern: DSA index buffer, HiSparse host-expanded spaces rounded UP for draft pools (kv_cache_configurator.py L678-684) — all in the "runtime slightly bigger than budget" direction, all small.

### F2 (LOW): DSA compress-tail buffers unbudgeted
- DSATokenToKVPool._init_kpool_compress_tail_buffers (memory_pool.py L5292+): `_compress_tail_k/_compress_tail_score` = (req_pool_size, tail_width=index_kpool+tail_extra_slots, index_head_dim) bf16 x2 per non-skipped indexer layer. Bytes are NOT in `_compute_dsa_indexer_cell_size` or anywhere in the budget. Only allocated when kpool compress active. req_pool_size = max_running + 2 (DSA pool is handed req_to_token.shape[0] = max_running+1, then +1 again L5319 — one extra row, safe side). Size example: 512 reqs x (1+? ) x 128 x 2B x 2 x 61 layers ≈ tens of MB — OOM-margin only when kpool-compress models run at very tight mem_fraction. VERDICT: budget leak (under), not corruption; worth charging in _compute_dsa_indexer_cell_size as fixed bias.

### F3 (INFORMATIONAL): DSV4 GPU pool asserts swa_page_size==256 (kv_cache_configurator.py L1769)
- `_build_dsv4_kv_pool` asserts page==256 (non-NPU). Prod page-64 DeepSeek-V4/DSA therefore cannot take the DSV4 compressed-pool path on GPU; it must be on the DSA path (_build_dsa_kv_pool, trtllm backends) — consistent with prod config. If anyone launches a `deepseek_v4` (compressed-attention) arch model at page 64 it fails loudly at boot (not silent). VERDICT: no bug; config-compat note. The DSV4 configurator's `assert page_size % 128 == 0` (pool_configurator L1571) passes at 64? NO: 64%128 != 0 -> fails loudly too. Both loud.

### F4 (CLEAN, verified): DP-attention floor-division consistency
- `resolve_max_num_reqs` (kv_cache_configurator.py L2806): `requested_per_worker = max_num_reqs // attn_dp_size`; mamba per-worker caps (L2980/2988/3005/3044) use `// attn_dp_size` floor; `compute_swa_request_cap` (pool_configurator L941): `num_reqs = max_running_requests // attn_dp_size`. All same floor formula, and the scheduler's per-worker SchedulePolicy enforces max_running_requests = the resolved per-worker value, so a worker can never hold ceil(global/dp) requests. VERDICT: consistent, no OOB direction. DSV4 request window (L1186): `slots = requested_max_running_requests_per_worker + 1` — +1 for the sentinel row, matches req pool's +1 padding row. DeepSeek V4 pool also receives `num_req_slots = req_to_token.shape[0]` (real capacity incl. PD decode pre-alloc) — comment L1802-1804 explicitly handles the PD-decode inflate. Consistent.

### F5 (CLEAN, verified): bs1 floor / reserved floor single source
- `_reserved_floor_bytes` (unified_memory_pool.py L359) is the single source both for UnifiedKVPool min_slot_index reservation (L458,510: ceil((reserved)/entry_bytes)) and the factories' `_check_bs1_feasibility_floor` sink terms (L1545-1549, 2071-2072, 2282). Same function, same page_size=1 exemption for MambaSubPoolSpec. min_slot_index = ceil division — a sub-pool entry straddling the sink boundary is excluded. VERDICT: matches exactly; bs=1 crash class guarded.

### F6 (CLEAN): num_pages vs max_total vs admission triple
- Paged allocator `num_pages = size // page_size` (paged.py L153) on a page-aligned size, so num_pages*page == size == max_total_num_tokens; highest handed-out loc = size-1; pool buffer valid range [0, size+page). Admission (scheduler) uses same max_total_num_tokens. DCP branch: allocator size widened by attn_dcp_size AND page widened equally (kv_cache_configurator.py L2592-2603) — num_pages unchanged, consistent. Draft loc_space_scale (L480-486) widens max_total only for the draft replica, matching the allocator's virtual space. VERDICT: no token loss, no mismatch direction.

### F7 (CLEAN, arithmetic-verified): fp8_e4m3 scale storage in DSA dims
- `calculate_mla_kv_cache_dim` fp8 scaled layout adds kv_lora//128*4 + rope*bf16.itemsize, then callers multiply whole dim by kv_size=1 (identity). bf16 DSA returns raw dim and pool stores bf16. mxfp8 MHA adds //32 scale bytes; fp4 uses quant_method.compute_cell_size. DSATokenToKVPool index buffer: per-token 128+128//128*4=132 B matches `_compute_dsa_indexer_cell_size` ceil_div(132, kpool)/kpool... with kpool=1 exact; with kpool>1 the SPLIT (ceil_div by kpool) matches IndexKeyCache's row of (page/kpool)*132 per page. Times num_indexer_layers (skip_topk layers get 0-row buffers — matches `_should_allocate_index_layer`). VERDICT: consistent for prod dtypes/backends.

### F8 (CLEAN): Hybrid SWA ratio sizing direction at page 64
- full = page-floor(budget/cell); swa = page-floor(int(full*ratio)); bytes used = full_cell*full + swa_cell*swa ≤ cell_size*full ≤ budget (float-rel-err ~1e-16 « int truncation). Unified-memory variant bisects with draft virtual-span overhang charged (ceil_align(span-1)+page). SWARequestCap mode: swa = ceil_align(cap) — ceil is correct direction (pool must hold cap). `_get_swa_tokens` DSV4: int(full*ratio)//page*page floors (<=). VERDICT: all floors conservative; no underestimate.

### F9 (VERIFIED OK): mamba pool sizing vs MambaPool allocation
- `_handle_max_mamba_cache`: final rest = rest - (max_mamba_cache_size+1)*(stage_per_req+extra)/2^30; MambaPool allocates (size+1) conv + (size+1) temporal; mamba_cache_per_req (mamba_utils.py L116) = (conv+temporal)*layers exactly. envelope_layout charged same via mamba_entry_bytes. PP: stage_per_req scaled by MAX stage fraction (over-charges symmetric ranks - safe, no collective). spec intermediate_ssm (size+1)*D charged separately under reserves_spec_state; replayssm GDN fixed/ring variants charged. Joint-solve integer discounts: (budget - per_req*(1+D) - extra) // (per_req*(1+D/ratio)+extra) then re-subtracts (capped+1)*D — capped_reqs min(global//dp, size//ratio) and size solved to cover intermediate for capped+1 slots; direction: allocated ≤ budget. VERDICT: conservative/correct for prod (extra_buffer, K3).
- Minor note: `_handle_max_mamba_cache` works in GiB floats; `total_rest_memory - intermediate_size/(1<<30)` then int() byte-floor in _profile_available_bytes — floor is safe.

### F10 (CLEAN): config_from_budget re-derivation
- On user-cap/PP-min the config is re-derived from token count only; assert `max_tokens <= profiled` (kv_cache_configurator.py L2897) forbids capacity growth, and `calculate_pool_sizes_from_max_tokens` never re-adds fixed pools — byte use only shrinks. unified_total_bytes only set on the profiling path. VERDICT: safe direction.

### Coverage vs instructions
1. Traced: config_from_budget, _resolve_memory_pool_config, _profile_available_bytes, _handle_max_mamba_cache, _calculate_mamba_ratio, _apply_token_constraints, resolve_max_num_reqs + all 4 pool configurators. Only budget-direction exceptions: F1 (+1 page rows, deliberate), F2 (DSA tail buffers unbudgeted).
2. page-64 floor/ceil: all audited paths floor capacities (safe); ceil used only where the pool must CONTAIN a worst case (swa cap, IndexKeyCache pages, draft virtual space). DSV4 asserts page%128==0 / swa page==256 → page-64 DSV4 compressed path boot-fails loudly (F3).
3. fp8 scale storage: F7 verified.
4. num_pages/num_tokens triple: F6 verified.
5. bs1 floor/reserved_floor: F5 verified exact.
6. DSA tail req_pool index: pool gets req_to_token.shape[0] (real capacity) and adds its own +1 → over-allocated by one row (safe); DSV4 num_req_slots uses real shape incl. PD decode pre-alloc (F4). matches known-class patch.
7. DP attn: F4 verified.

### NOT found in my files (checked for siblings): no capacity-raising rounding anywhere; no path where scheduler admission exceeds allocator slots; no double-count that shrinks usable cache materially beyond +1-page slop.
