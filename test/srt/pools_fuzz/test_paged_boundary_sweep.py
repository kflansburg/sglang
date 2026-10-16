"""Exhaustive alloc_extend boundary sweep at production page size 64:
every prefix_len mod 64, every extend 0..66. Verifies phase-1 tokens stay on
the prefix tail page, page counts are exact, and no alloc fails."""
import torch
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

PS = 64

def _mk(n=40):
    return PagedTokenToKVPoolAllocator(size=n*PS, page_size=PS, dtype=torch.float16,
                                       device="cpu", kvcache=object(), need_sort=False)

def test_extend_boundary_sweep_page64():
    for prefix_len in range(0, 3*PS+1):
        for ext in range(0, PS+2):
            seq_len = prefix_len + ext
            if seq_len == 0:
                continue
            a = _mk()
            pre_out = None
            ll = torch.tensor([-1])
            if prefix_len:
                pre_out = a.alloc_extend(torch.tensor([0]), torch.tensor([0]),
                                         torch.tensor([prefix_len]), torch.tensor([prefix_len]),
                                         torch.tensor([-1]), prefix_len)
                assert pre_out is not None and len(pre_out) == prefix_len
                ll = pre_out[-1:].clone()
            out = a.alloc_extend(torch.tensor([prefix_len]), torch.tensor([prefix_len]),
                                 torch.tensor([seq_len]), torch.tensor([seq_len]),
                                 ll, ext)
            if ext == 0:
                assert out is not None and len(out) == 0
                continue
            need_pages = (seq_len + PS - 1)//PS - (prefix_len + PS - 1)//PS
            assert out is not None, f"need {need_pages} pages prefix={prefix_len} ext={ext}, got None"
            out_l = out.tolist()
            assert len(out_l) == ext
            for i, idx in enumerate(out_l):
                tok = prefix_len + i
                within_old_partial = tok < ((prefix_len + PS - 1)//PS)*PS and prefix_len % PS != 0
                if within_old_partial:
                    assert idx // PS == (pre_out[-1].item())//PS, f"phase-1 token {tok} left the tail page"
            all_pages = (set((pre_out//PS).tolist()) if pre_out is not None else set()) | set((out//PS).tolist())
            assert len(all_pages) == (seq_len+PS-1)//PS

if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q"]))
