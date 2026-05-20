from typing import Tuple

import torch
import torch.nn as nn


class TriBranchFusion(nn.Module):
    """Adaptive branch weighting at each (time, node)."""

    def __init__(self, dim: int, fusion_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(dim * 3, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 3),
        )
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        y_phy: torch.Tensor,
        y_freq: torch.Tensor,
        y_time: torch.Tensor,
        branch_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # all branches: (B, T, N, D)
        stacked = torch.stack([y_phy, y_freq, y_time], dim=-2)  # (B, T, N, 3, D)
        logits = self.router(torch.cat([y_phy, y_freq, y_time], dim=-1))  # (B, T, N, 3)
        if branch_mask is not None:
            # branch_mask is shape (3,) with True meaning enabled branch.
            mask = branch_mask.to(device=logits.device, dtype=torch.bool).view(1, 1, 1, 3)
            logits = logits.masked_fill(~mask, -1e9)
        weights = torch.softmax(logits, dim=-1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=-2)
        return self.out(fused), weights
