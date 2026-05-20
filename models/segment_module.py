import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentModule(nn.Module):
    """Segment-based trend extractor with intra/inter segment transforms."""

    def __init__(self, hidden_dim: int, seq_len: int, pred_len: int, seg_len: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = int(seq_len)
        self.pred_len = pred_len
        self.seg_len = max(2, int(seg_len))
        self.n_seg_in = math.ceil(self.seq_len / self.seg_len)
        self.n_seg_out = math.ceil(self.pred_len / self.seg_len)

        self.intra = nn.Linear(self.seg_len, self.seg_len)
        self.inter = nn.Linear(self.n_seg_in, self.n_seg_out, bias=False)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, N, D)
        bsz, seq_len, num_nodes, dim = x.shape
        if seq_len != self.seq_len:
            raise ValueError(f"SegmentModule expected seq_len={self.seq_len}, but got {seq_len}")
        n_seg_in = self.n_seg_in
        n_seg_out = self.n_seg_out
        pad_len = n_seg_in * self.seg_len - seq_len
        x_pad = F.pad(x, (0, 0, 0, 0, 0, pad_len), mode="replicate")

        x_seg = x_pad.view(bsz, n_seg_in, self.seg_len, num_nodes, dim).permute(0, 3, 4, 1, 2)
        x_intra = self.intra(x_seg)

        x_inter = self.inter(x_intra.transpose(-1, -2)).transpose(-1, -2)

        y = x_inter.permute(0, 3, 4, 1, 2).reshape(bsz, n_seg_out * self.seg_len, num_nodes, dim)
        y = y[:, : self.pred_len]
        return self.out_norm(y)
