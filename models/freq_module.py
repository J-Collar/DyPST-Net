import torch
import torch.nn as nn


class FreqModule(nn.Module):
    """FFT pathway with adaptive low-rank filter and temporal projection."""

    def __init__(self, hidden_dim: int, seq_len: int, pred_len: int, lpf_k: int = 16, rank: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = int(seq_len)
        self.pred_len = pred_len
        self.lpf_k = lpf_k
        self.rank = rank

        self.u = nn.Parameter(torch.randn(lpf_k, rank) * 0.02)
        self.v = nn.Parameter(torch.randn(rank, lpf_k) * 0.02)

        self.time_proj = nn.Linear(self.seq_len, self.pred_len, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, N, D)
        bsz, seq_len, num_nodes, dim = x.shape
        if seq_len != self.seq_len:
            raise ValueError(f"FreqModule expected seq_len={self.seq_len}, but got {seq_len}")

        x_bn = x.permute(0, 2, 3, 1)  # (B, N, D, L)
        x_fft = torch.fft.rfft(x_bn, dim=-1)
        n_freq = x_fft.shape[-1]

        k = min(self.lpf_k, n_freq)
        low = x_fft[..., :k]

        w = torch.matmul(self.u[:k, :], self.v[:, :k]).to(low.dtype)  # (k, k)
        low_filtered = torch.einsum("bndk,kl->bndl", low, w)

        out_fft = x_fft.clone()
        out_fft[..., :k] = low_filtered
        x_ifft = torch.fft.irfft(out_fft, n=seq_len, dim=-1)

        x_time = self.time_proj(x_ifft.reshape(-1, seq_len)).reshape(bsz, num_nodes, dim, self.pred_len)
        x_time = x_time.permute(0, 3, 1, 2)
        return self.norm(x_time)
