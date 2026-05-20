from typing import Optional

import torch
import torch.nn as nn


class ExplicitODEFunc(nn.Module):
    """Discrete diffusion-advection ODE function for SST latent states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.diff_coeff = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.adv_coeff = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.diff_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.adv_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _nodewise_matmul(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # adj: (N, N) or (B, N, N), x: (B, N, H)
        if adj.dim() == 2:
            return torch.einsum("ij,bjh->bih", adj, x)
        if adj.dim() == 3:
            return torch.einsum("bij,bjh->bih", adj, x)
        raise ValueError(f"adj must be 2D or 3D tensor, got shape={tuple(adj.shape)}")

    def forward(
        self,
        x: torch.Tensor,
        adj_diff: torch.Tensor,
        adj_adv: Optional[torch.Tensor] = None,
    ):
        diff_term = self._nodewise_matmul(adj_diff, x) - x
        diff_term = self.diff_proj(diff_term)

        if adj_adv is None:
            adv_term = torch.zeros_like(diff_term)
        else:
            adv_term = self._nodewise_matmul(adj_adv, x) - x
            adv_term = self.adv_proj(adv_term)

        grad = self.diff_coeff * diff_term + self.adv_coeff * adv_term + self.beta * x
        return grad
