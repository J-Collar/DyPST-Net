import torch.nn as nn

from .encoder import Encoder
from .freq_module import FreqModule


class FreqBranch(nn.Module):
    """Frequency branch with FFT and low-rank spectral filtering."""

    def __init__(self, input_dim: int, hidden_dim: int, seq_len: int, pred_len: int, lpf_k: int, rank: int):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.freq_module = FreqModule(
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            lpf_k=lpf_k,
            rank=rank,
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.freq_module(x)
