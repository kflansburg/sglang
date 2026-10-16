#!/usr/bin/env python3
"""R-1 ragged-verify hotfix: edit-in-place on a tag worktree.
Usage: patch_r1.py <worktree-root> <tag>  (v0.5.17 | v0.5.20 only)
Edits: ragged_verify.py padded_to_bucket (+last_row_cap), pad_verify_lens_to_bucket
        (+last_row_cap arg + capped fold), and the 3 full-coverage consumer
        sites (dsv4, trtllm_mha, fa) — passes last_row_cap=speculative_num_draft_tokens.
"""
import os
import sys

WT, TAG = sys.argv[1], sys.argv[2]


def edit(rel, transforms):
    p = os.path.join(WT, rel)
    t = open(p).read()
    for (old, new) in transforms:
        c = t.count(old)
        assert c == 1, f"{rel}: count={c} for {old[:60]!r}"
        t = t.replace(old, new)
    open(p, "w").write(t)
    print("edited", rel)


LEFT = """    else:
        padded = padded.clone()
        padded[-1] = (padded[-1].to(torch.int64) + leftover).to(torch.int32)
    return padded"""
NEW = """    else:
        # R-1 HOTFIX: cap the last-real-row fold at last_row_cap (the
        # per-request draft budget). Legacy uncapped fold widened the row's
        # verify past its drafted width and read stale prior-owner table
        # content (cross-request exposure). Remainder collapses into the
        # consumer's ghost handling (same contract as the cap=N layout).
        padded = padded.clone()
        fold = leftover
        if last_row_cap is not None:
            fold = min(int(leftover), last_row_cap - int(padded[-1]))
            fold = max(fold, 0)
        padded[-1] = (padded[-1].to(torch.int64) + fold).to(torch.int32)
    return padded"""

edit(
    "python/sglang/kernels/ops/speculative/ragged_verify_kernels.py",
    [
        (
            """def pad_verify_lens_to_bucket(
    *,
    verify_lens: torch.Tensor,
    graph_num_tokens: int,
    bs: int,
    padded_bs: int,
) -> torch.Tensor:""",
            """def pad_verify_lens_to_bucket(
    *,
    verify_lens: torch.Tensor,
    graph_num_tokens: int,
    bs: int,
    padded_bs: int,
    last_row_cap=None,
) -> torch.Tensor:""",
        ),
        (LEFT, NEW),
    ],
)

# ragged_verify.py padded_to_bucket: add last_row_cap param + thread + None assembly
RV_OLD = """    def padded_to_bucket(
        self, *, padded_bs: int, cap: Optional[int] = None
    ) -> RaggedVerifyLayout:"""
RV_NEW = """    def padded_to_bucket(
        self, *, padded_bs: int, cap: Optional[int] = None, last_row_cap: Optional[int] = None
    ) -> RaggedVerifyLayout:"""
RV_CALL = """        padded = PaddedToBucket.execute(
            verify_lens=self.verify_lens,
            graph_num_tokens=self.graph_num_tokens,
            bs=self.bs,
            padded_bs=padded_bs,
        )"""
RV_CALL_NEW = """        padded = PaddedToBucket.execute(
            verify_lens=self.verify_lens,
            graph_num_tokens=self.graph_num_tokens,
            bs=self.bs,
            padded_bs=padded_bs,
            last_row_cap=last_row_cap,
        )"""
RV_ASM = "            total_verify_tokens=None if cap is not None else self.graph_num_tokens,"
RV_ASM_NEW = "            total_verify_tokens=None if (cap is not None or last_row_cap is not None) else self.graph_num_tokens,"

edit(
    "python/sglang/srt/speculative/ragged_verify.py",
    [(RV_OLD, RV_NEW), (RV_CALL, RV_CALL_NEW), (RV_ASM, RV_ASM_NEW)],
)

# consumers
edit(
    "python/sglang/srt/layers/attention/deepseek_v4_backend.py",
    [(
        "        layout = layout.padded_to_bucket(padded_bs=bs)",
        "        # R-1 HOTFIX: cap the no-padding-row fold at the draft budget.\n"
        "        layout = layout.padded_to_bucket(\n"
        "            padded_bs=bs, last_row_cap=self.speculative_num_draft_tokens\n"
        "        )",
    )],
)
edit(
    "python/sglang/srt/layers/attention/trtllm_mha_backend.py",
    [(
        "        padded_layout = ragged_layout.padded_to_bucket(padded_bs=bs)",
        "        # R-1 HOTFIX\n"
        "        padded_layout = ragged_layout.padded_to_bucket(\n"
        "            padded_bs=bs, last_row_cap=self.speculative_num_draft_tokens\n"
        "        )",
    )],
)
edit(
    "python/sglang/srt/layers/attention/flashattention_backend.py",
    [(
        "                    padded = ragged_layout.padded_to_bucket(padded_bs=bs)",
        "                    # R-1 HOTFIX\n"
        "                    padded = ragged_layout.padded_to_bucket(\n"
        "                        padded_bs=bs,\n"
        "                        last_row_cap=self.speculative_num_draft_tokens,\n"
        "                    )",
    )],
)
