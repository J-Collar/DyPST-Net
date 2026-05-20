import math

import torch
import torch.nn as nn

from .encoder import Encoder
from .ode_func import ExplicitODEFunc
from .diff_eq_solver import DiffEqSolver
from .diff_adv_fusion import DiffAdvFusion


class PhyBranch(nn.Module):
    """Physics branch inspired by Air-DualODE's diffusion-advection dynamics."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        pred_len: int,
        ode_steps: int = 2,
        dt: float = 0.5,
        wind_speed_idx: int = 0,
        wind_dir_idx: int = 1,
        wind_speed_mean: float = 0.0,
        wind_speed_scale: float = 1.0,
        wind_dir_mean: float = 0.0,
        wind_dir_scale: float = 1.0,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.wind_speed_idx = wind_speed_idx
        self.wind_dir_idx = wind_dir_idx
        self.wind_speed_mean = float(wind_speed_mean)
        self.wind_speed_scale = max(float(wind_speed_scale), 1e-6)
        self.wind_dir_mean = float(wind_dir_mean)
        self.wind_dir_scale = max(float(wind_dir_scale), 1e-6)

        self.encoder = Encoder(input_dim, hidden_dim)
        self.odefunc_diff = ExplicitODEFunc(hidden_dim)
        self.odefunc_adv = ExplicitODEFunc(hidden_dim)
        self.solver = DiffEqSolver(ode_steps=ode_steps, dt=dt)
        self.fusion = DiffAdvFusion(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def _build_adv_adj(self, x: torch.Tensor, adj_diff: torch.Tensor) -> torch.Tensor:
        # Estimate directed transport per sample to avoid cross-sample leakage.
        # x: (B, L, N, C), adj_diff: (N, N)
        eps = 1e-6
        wind_speed_std = x[:, -1, :, 1 + self.wind_speed_idx]  # (B, N)
        wind_dir_std = x[:, -1, :, 1 + self.wind_dir_idx]      # (B, N)

        wind_speed = wind_speed_std * self.wind_speed_scale + self.wind_speed_mean
        wind_dir = wind_dir_std * self.wind_dir_scale + self.wind_dir_mean
        wd_rad = wind_dir * math.pi / 180.0
        wind_vec = torch.stack([torch.cos(wd_rad), torch.sin(wd_rad)], dim=-1)  # (B, N, 2)

        sim = torch.matmul(wind_vec, wind_vec.transpose(1, 2))
        sim = torch.relu(sim)

        mag = torch.relu(wind_speed).unsqueeze(-1)  # (B, N, 1)
        adv = sim * mag * adj_diff.unsqueeze(0)
        adv = adv / (adv.sum(dim=-1, keepdim=True) + eps)
        return adv

    def forward(self, x: torch.Tensor, adj_diff: torch.Tensor) -> torch.Tensor:
        x_enc = self.encoder(x)
        x0 = x_enc[:, -1, :, :]  # (B, N, H)
        adj_adv = self._build_adv_adj(x, adj_diff)

        y_diff = self.solver(self.odefunc_diff, x0, self.pred_len, adj_diff=adj_diff, adj_adv=None)
        y_adv = self.solver(self.odefunc_adv, x0, self.pred_len, adj_diff=adj_diff, adj_adv=adj_adv)
        y = self.fusion(y_diff, y_adv)
        return self.out_proj(y)
