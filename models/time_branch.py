import torch.nn as nn

from .encoder import Encoder
from .segment_module import SegmentModule


class TimeBranch(nn.Module):
    """Temporal branch based on segment trend extraction."""

    def __init__(self, input_dim: int, hidden_dim: int, seq_len: int, pred_len: int, seg_len: int):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.segment_module = SegmentModule(
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            seg_len=seg_len,
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.segment_module(x)
