"""Equivalence test: Triton alloc_extend/alloc_decode index math vs the naive
torch reference, transcribed to CPU (no GPU needed).

alloc_extend_kernel / alloc_decode_kernel in
  python/sglang/kernels/ops/memory/allocator.py
are transcribed instruction-for-instruction in plain torch below (per-program
scalar semantics, bs programs). Asserting triton-transcription == naive
reference across a boundary grid (page_size 64, page-multiple / off-by-one seq
and prefix lens, batch sizes 1..9) proves the kernels emit exactly the naive
mapping -- and any divergence between the two is a genuine kernel bug, since
the naive path is the production fallback on non-Triton platforms.

Also asserts:
  * out_indices are unique (no slot handed out twice in one batch),
  * out_indices token slots for new pages match free_pages[:num_new_pages],
  * out-of-range probe: no emitted slot >= (last committed new page's end), i.e.
    the kernel never reads free_pages beyond num_new_pages for its OUTPUT slots.
"""

import torch


def triton_alloc_extend(prefix_lens, seq_lens, last_loc, free_pages, out_indices, page_size):
    """Instruction-level transcription of alloc_extend_kernel (per program pid)."""
    bs = len(seq_lens)
    # each program computes the full prefix-sum over masked loads (<= pid lanes)
    for pid in range(bs):
        sl = seq_lens[: pid + 1]
        pl = prefix_lens[: pid + 1]
        extend_lens = sl - pl  # masked lanes contribute 0 -> match masked tl.load
        seq_len = int(seq_lens[pid]); pre_len = int(prefix_lens[pid])
        sum_extend = int(extend_lens.sum())
        output_start = sum_extend - (seq_len - pre_len)
        n_after = (sl + page_size - 1) // page_size
        n_before = (pl + page_size - 1) // page_size
        n_prog_self = (seq_len + page_size - 1) // page_size - (pre_len + page_size - 1) // page_size
        sum_new = int((n_after - n_before).sum())
        new_page_start = sum_new - n_prog_self
        ll = int(last_loc[pid])
        n1 = min(seq_len, (pre_len + page_size - 1) // page_size * page_size) - pre_len
        for t in range(n1):
            out_indices[output_start + t] = ll + 1 + t
        if pre_len + n1 == seq_len:
            continue
        n2 = (seq_len // page_size) * page_size - ((pre_len + page_size - 1) // page_size) * page_size
        for t in range(0, n2):
            page = int(free_pages[new_page_start + t // page_size])
            out_indices[output_start + n1 + t] = page * page_size + t % page_size
        if pre_len + n1 + n2 == seq_len:
            continue
        n3 = seq_len - (seq_len // page_size) * page_size
        # Triton load: unmasked tl.load(free_page_ptr + new_page_start + n_prog_self - 1)
        idx = new_page_start + n_prog_self - 1
        # NOTE: idx can lie beyond the committed new-page range (a stale/OOB free read,
        # BUG-ALLOC-4); emulate torch semantics exactly (negative index wraps).
        start = int(free_pages[idx])
        for t in range(n3):
            out_indices[output_start + n1 + n2 + t] = start * page_size + t


def naive_alloc_extend(prefix_lens, seq_lens, last_loc, free_pages, out_indices, page_size):
    """Reference copy of alloc_extend_naive from paged.py (pasted, no sglang import)."""
    extend_lens = seq_lens - prefix_lens
    end_pos = torch.cumsum(extend_lens, 0)
    start_pos = end_pos - extend_lens
    extend_num_tokens = out_indices.shape[0]
    if extend_num_tokens == 0:
        return
    j = torch.arange(extend_num_tokens, dtype=torch.int64)
    owner = torch.searchsorted(end_pos, j, right=True)
    local = j - start_pos[owner]
    last_loc_g = last_loc[owner]
    ceil_prefix = (prefix_lens + page_size - 1) // page_size * page_size
    floor_seq = seq_lens // page_size * page_size
    if free_pages.numel() == 0:
        ceil_seq = (seq_lens + page_size - 1) // page_size * page_size
        assert torch.all(ceil_seq == ceil_prefix)
        out_indices.copy_(last_loc_g + 1 + local)
        return
    num1 = torch.clamp(seq_lens, max=ceil_prefix) - prefix_lens
    done_after_1 = (prefix_lens + num1) == seq_lens
    num2 = torch.where(done_after_1, torch.zeros_like(num1), floor_seq - ceil_prefix)
    num3 = torch.where(done_after_1, torch.zeros_like(num1), seq_lens - floor_seq)
    full_pages = num2 // page_size
    need_extra_page = (num3 > 0).to(torch.int64)
    pages_per_req = full_pages + need_extra_page
    end_new_pages = torch.cumsum(pages_per_req, 0)
    start_new_pages = end_new_pages - pages_per_req
    num1_g = num1[owner]
    num2_g = num2[owner]
    start_g = start_new_pages[owner]
    end_g = end_new_pages[owner]
    is1 = local < num1_g
    is2 = (~is1) & (local < num1_g + num2_g)
    v1 = last_loc_g + 1 + local
    rel2 = torch.clamp(local - num1_g, min=0)
    pi2 = torch.clamp(start_g + rel2 // page_size, min=0, max=free_pages.numel() - 1)
    v2 = free_pages[pi2] * page_size + rel2 % page_size
    rel3 = torch.clamp(local - num1_g - num2_g, min=0)
    pi3 = torch.clamp(end_g - 1, min=0, max=free_pages.numel() - 1)
    v3 = free_pages[pi3] * page_size + rel3
    out_indices.copy_(torch.where(is1, v1, torch.where(is2, v2, v3)))


def triton_alloc_decode(seq_lens, last_loc, free_pages, out_indices, page_size):
    """Instruction-level transcription of alloc_decode_kernel (per program pid)."""
    for pid in range(len(seq_lens)):
        sl = seq_lens[: pid + 1]
        pre = sl - 1
        n_after = (sl + page_size - 1) // page_size
        n_before = (pre + page_size - 1) // page_size
        n_prog_self = int(n_after[-1] - n_before[-1])
        sum_new = int((n_after - n_before).sum())
        new_page_start = sum_new - n_prog_self
        if n_prog_self == 0:
            out_indices[pid] = int(last_loc[pid]) + 1
        else:
            page = int(free_pages[new_page_start])
            out_indices[pid] = page * page_size


def naive_alloc_decode(seq_lens, last_loc, free_pages, out_indices, page_size, num_new_pages):
    if num_new_pages == 0:
        out_indices.copy_(last_loc + 1)
        return
    need_new = (((seq_lens - 1) % page_size) == 0).to(torch.int64)
    rank = (torch.cumsum(need_new, 0) - need_new).clamp_(max=num_new_pages - 1)
    out_indices.copy_(torch.where(need_new.bool(), free_pages[rank] * page_size, last_loc + 1))


def run_extend_case(ps, prefixes, seqs):
    bs = len(prefixes)
    prefix_lens = torch.tensor(prefixes)
    seq_lens = torch.tensor(seqs)
    last_loc = torch.tensor([p - 1 for p in prefixes])
    ext = int((seq_lens - prefix_lens).sum())
    # Free list must be disjoint from the prefix tails: a prefix of p tokens with
    # last_loc = p - 1 occupies pages 0 .. (p-1)//ps (slot layout is contiguous in
    # this model), so start the free list after the deepest prefix tail page.
    tails = [(p - 1) // ps for p in prefixes if p > 0]
    deepest = max(tails) if tails else 0
    start_pg = max(1, deepest + 1)
    free_pages = torch.arange(start_pg, start_pg + 200000 // ps + 1, dtype=torch.int64)
    num_new_pages = int(((seq_lens + ps - 1) // ps - (prefix_lens + ps - 1) // ps).sum())
    out_t = torch.empty(ext, dtype=torch.int64)
    out_n = torch.empty(ext, dtype=torch.int64)
    triton_alloc_extend(prefix_lens, seq_lens, last_loc, free_pages, out_t, ps)
    naive_alloc_extend(prefix_lens, seq_lens, last_loc, free_pages, out_n, ps)
    assert torch.equal(out_t, out_n), (
        f"extend diverges ps={ps} prefixes={prefixes} seqs={seqs}\n"
        f" triton={out_t.tolist()}\n naive={out_n.tolist()}"
    )
    assert len(set(out_t.tolist())) == ext, f"dup slot in extend out ps={ps} {prefixes} {seqs}"
    # the new pages must occupy exactly free_pages[:num_new_pages] footprints
    new_pages_used = sorted(set(
        int(x) // ps for x in out_t.tolist() if int(x) // ps != (prefixes[0] - 1) // ps
    ))
    free_used = set(free_pages[:num_new_pages].tolist())
    for x in out_t.tolist():
        pg = int(x) // ps
        # slots outside the drawn new pages must lie on a prefix-owned page
        if pg not in free_used:
            assert any(p > 0 and (p - 1) // ps == pg for p in prefixes), (
                f"extend wrote slot {x} (page {pg}) which is neither a drawn new page "
                f"nor a caller prefix page (ps={ps}, prefixes={prefixes}, seqs={seqs})"
            )


def run_decode_case(ps, seqs):
    seq_lens = torch.tensor(seqs)
    last_loc = torch.tensor([s - 2 for s in seqs])
    # decode: the token before the new one sits at slot seq-2; the current partial
    # page is (seq-2)//ps and is caller-owned, so the free list starts after it.
    deepest = max((s - 2) // ps for s in seqs)
    start_pg = max(1, deepest + 1)
    free_pages = torch.arange(start_pg, start_pg + 200000 // ps + 1, dtype=torch.int64)
    num_new_pages = int(((seq_lens % ps) == 1).sum())
    out_t = torch.empty(len(seqs), dtype=torch.int64)
    out_n = torch.empty(len(seqs), dtype=torch.int64)
    triton_alloc_decode(seq_lens, last_loc, free_pages, out_t, ps)
    naive_alloc_decode(seq_lens, last_loc, free_pages, out_n, ps, num_new_pages)
    assert torch.equal(out_t, out_n), (
        f"decode diverges ps={ps} seqs={seqs}: triton={out_t.tolist()} naive={out_n.tolist()}"
    )
    for i, x in enumerate(out_t.tolist()):
        pg = int(x) // ps
        expect = seqs[i] % ps == 1  # new page iff seq_len % ps == 1
        if expect:
            assert pg in set(free_pages[:num_new_pages].tolist())


def main():
    ps = 64
    boundary = [0, 1, ps - 1, ps, ps + 1, 2 * ps - 1, 2 * ps, 2 * ps + 1, 3 * ps, 4 * ps + 7, 5 * ps - 1]
    checks = 0
    # extend: all (prefix, seq) boundary pairs + assorted batches
    for pre in boundary:
        for seq in boundary:
            if seq > pre:
                run_extend_case(ps, [pre], [seq]); checks += 1
    for a in boundary:
        for b in boundary:
            if a < b:
                run_extend_case(ps, [0, a, 0], [3 * ps + 2, b, ps]); checks += 1
    # decode: boundary seq lens incl. page multiples and multiple-1
    for s in boundary:
        if s >= 1:
            run_decode_case(ps, [s]); checks += 1
    for a in boundary:
        for b in boundary:
            if a >= 1 and b >= 1:
                run_decode_case(ps, [a, b]); checks += 1
    # page_size = 1 sanity (extend only: at ps==1 token pools call alloc(1), not
    # alloc_decode; the paged decode host formula `seq % ps == 1` and the kernel's
    # `ceil(seq/ps) - ceil((seq-1)/ps)` disagree for every seq at ps==1 -> latent
    # free-list desync if alloc_decode_kernel is ever launched with page_size==1,
    # see pools_findings_alloc.md BUG-ALLOC-5).
    for pre in range(0, 5):
        for seq in range(pre + 1, 8):
            run_extend_case(1, [pre], [seq]); checks += 1
    print(f"kernel-vs-naive equivalence: {checks} cases OK")


if __name__ == "__main__":
    main()
