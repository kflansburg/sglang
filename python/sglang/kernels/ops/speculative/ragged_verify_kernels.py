from __future__ import annotations

import msgspec
from typing import Optional
import torch
import triton
import triton.language as tl


class PaddedToBucket:
    @classmethod
    def execute(
        cls,
        *,
        verify_lens: torch.Tensor,
        graph_num_tokens: int,
        bs: int,
        padded_bs: int,
        last_row_cap: Optional[int] = None,
    ) -> torch.Tensor:
        impl = cls.triton if verify_lens.is_cuda else cls.torch
        return impl(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            bs=bs,
            padded_bs=padded_bs,
            last_row_cap=last_row_cap,
        )

    @classmethod
    def torch(
        cls,
        *,
        verify_lens: torch.Tensor,
        graph_num_tokens: int,
        bs: int,
        padded_bs: int,
        last_row_cap: Optional[int] = None,
    ) -> torch.Tensor:
        return pad_verify_lens_to_bucket(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            bs=bs,
            padded_bs=padded_bs,
            last_row_cap=last_row_cap,
        )

    @classmethod
    def triton(
        cls,
        *,
        verify_lens: torch.Tensor,
        graph_num_tokens: int,
        bs: int,
        padded_bs: int,
        last_row_cap: Optional[int] = None,
    ) -> torch.Tensor:
        return pad_verify_lens_to_bucket_triton(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            bs=bs,
            padded_bs=padded_bs,
            last_row_cap=last_row_cap,
        )


def pad_verify_lens_to_bucket(
    *,
    verify_lens: torch.Tensor,
    graph_num_tokens: int,
    bs: int,
    padded_bs: int,
    last_row_cap: Optional[int] = None,
) -> torch.Tensor:
    assert padded_bs >= bs, (
        f"padded_bs {padded_bs} < bs {bs}: the captured tier cannot hold this "
        "batch's requests"
    )
    device = verify_lens.device
    num_pad_reqs = padded_bs - bs
    padded = verify_lens.to(torch.int32)
    leftover = graph_num_tokens - padded.to(torch.int64).sum()
    if num_pad_reqs > 0:
        base = leftover // num_pad_reqs
        rem = leftover - base * num_pad_reqs
        pad_block = base + (
            torch.arange(num_pad_reqs, device=device, dtype=torch.int64) < rem
        )
        padded = torch.cat([padded, pad_block.to(torch.int32)])
    else:
        # R-1: without a padding row to absorb the tier's leftover, folding it
        # into the LAST REAL row can push that row past its drafted width
        # (sum of other rows' rejected-drafts budget): the verify positions
        # then run past seq + num_draft into never-allocated table entries
        # (stale prior-owner slot content leaks into this row's attention).
        # Cap the fold at last_row_cap (the per-request draft budget); the
        # uncovered remainder collapses into the consumer's ghost handling,
        # exactly as the cap=N layout variant semantics.
        padded = padded.clone()
        fold = leftover
        if last_row_cap is not None:
            fold = min(int(leftover), last_row_cap - int(padded[-1]))
            fold = max(fold, 0)
        padded[-1] = (padded[-1].to(torch.int64) + fold).to(torch.int32)
    return padded


@triton.jit
def _padded_to_bucket_kernel(
    verify_lens_ptr,
    out_ptr,
    bs,
    padded_bs,
    graph_num_tokens,
    last_row_cap,  # constexpr(-1 = uncapped legacy fold)
    BLOCK: tl.constexpr,
):
    idx = tl.arange(0, BLOCK)
    valid = idx < padded_bs
    is_real = idx < bs
    vl = tl.load(verify_lens_ptr + idx, mask=is_real, other=0).to(tl.int64)
    leftover = graph_num_tokens - tl.sum(vl)
    num_pad = padded_bs - bs
    num_pad_safe = tl.maximum(num_pad, 1)
    base = leftover // num_pad_safe
    rem = leftover - base * num_pad_safe
    pad_len = base + tl.where((idx - bs) < rem, 1, 0)
    final = tl.where(is_real, vl, pad_len)
    # R-1: cap the last-real-row fold at last_row_cap (per-request draft
    # budget); the uncovered remainder falls to the consumer's ghost
    # handling instead of widening the last row's verify past its drafts.
    if last_row_cap != -1:
        last_len = tl.load(verify_lens_ptr + (bs - 1)).to(tl.int64)
        fold = tl.minimum(leftover, tl.maximum(last_row_cap - last_len, 0))
        final = final + tl.where((num_pad == 0) & (idx == bs - 1), fold, 0)
    else:
        final = final + tl.where((num_pad == 0) & (idx == bs - 1), leftover, 0)
    tl.store(out_ptr + idx, final.to(tl.int32), mask=valid)


def pad_verify_lens_to_bucket_triton(
    *,
    verify_lens: torch.Tensor,
    graph_num_tokens: int,
    bs: int,
    padded_bs: int,
    last_row_cap: Optional[int] = None,
) -> torch.Tensor:
    assert padded_bs >= bs, (
        f"padded_bs {padded_bs} < bs {bs}: the captured tier cannot hold this "
        "batch's requests"
    )
    device = verify_lens.device
    verify_lens = verify_lens.to(torch.int32).contiguous()
    out = torch.empty(padded_bs, dtype=torch.int32, device=device)
    BLOCK = triton.next_power_of_2(max(padded_bs, 1))
    _padded_to_bucket_kernel[(1,)](
        verify_lens,
        out,
        bs,
        padded_bs,
        graph_num_tokens,
        -1 if last_row_cap is None else last_row_cap,
        BLOCK=BLOCK,
    )
    return out


class QoIndptrResult(msgspec.Struct):
    qo_indptr: torch.Tensor
    extend_start_loc: torch.Tensor


class BuildQoIndptr:
    @classmethod
    def execute(cls, *, verify_lens: torch.Tensor) -> QoIndptrResult:
        impl = cls.triton if verify_lens.is_cuda else cls.torch
        return impl(verify_lens=verify_lens)

    @classmethod
    def torch(cls, *, verify_lens: torch.Tensor) -> QoIndptrResult:
        return build_qo_indptr(verify_lens=verify_lens)

    @classmethod
    def triton(cls, *, verify_lens: torch.Tensor) -> QoIndptrResult:
        return build_qo_indptr_triton(verify_lens=verify_lens)


def build_qo_indptr(*, verify_lens: torch.Tensor) -> QoIndptrResult:
    verify_lens = verify_lens.to(torch.int32)
    cumsum = torch.cumsum(verify_lens, dim=0).to(torch.int32)
    zero = torch.zeros(1, dtype=torch.int32, device=verify_lens.device)
    qo_indptr = torch.cat([zero, cumsum])
    extend_start_loc = qo_indptr[:-1].clone()
    return QoIndptrResult(qo_indptr=qo_indptr, extend_start_loc=extend_start_loc)


@triton.jit
def _qo_indptr_kernel(
    verify_lens_ptr,
    qo_indptr_ptr,
    extend_start_loc_ptr,
    bs,
    BLOCK: tl.constexpr,
):
    idx = tl.arange(0, BLOCK)
    valid = idx < bs
    vl = tl.load(verify_lens_ptr + idx, mask=valid, other=0).to(tl.int32)
    incl = tl.cumsum(vl, axis=0)
    excl = incl - vl
    tl.store(qo_indptr_ptr, 0)
    tl.store(qo_indptr_ptr + 1 + idx, incl, mask=valid)
    tl.store(extend_start_loc_ptr + idx, excl, mask=valid)


def build_qo_indptr_triton(*, verify_lens: torch.Tensor) -> QoIndptrResult:
    bs = verify_lens.shape[0]
    device = verify_lens.device
    verify_lens = verify_lens.contiguous()
    qo_indptr = torch.empty(bs + 1, dtype=torch.int32, device=device)
    extend_start_loc = torch.empty(bs, dtype=torch.int32, device=device)
    BLOCK = triton.next_power_of_2(max(bs, 1))
    _qo_indptr_kernel[(1,)](
        verify_lens,
        qo_indptr,
        extend_start_loc,
        bs,
        BLOCK=BLOCK,
    )
    return QoIndptrResult(qo_indptr=qo_indptr, extend_start_loc=extend_start_loc)
