from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        if edge_index.numel() == 0 or h.shape[0] == 0:
            return h
        out = torch.zeros_like(h)
        deg = torch.zeros(h.shape[0], device=h.device)
        src, dst = edge_index[0], edge_index[1]
        out.index_add_(0, dst, h[src])
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
        deg = deg.clamp(min=1.0).unsqueeze(-1)
        return out / deg


class GATEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 128) -> None:
        super().__init__()
        self.conv1 = MeanGraphConv(in_dim, hidden_dim)
        self.conv2 = MeanGraphConv(hidden_dim, out_dim)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(node_features, edge_index))
        h = self.conv2(h, edge_index)
        if h.shape[0] == 0:
            return torch.zeros(h.shape[-1], device=node_features.device)
        return h.mean(dim=0)


@dataclass
class GNNEncoderConfig:
    in_dim: int = 32
    hidden_dim: int = 64
    out_dim: int = 128


def build_gnn_encoder(cfg: GNNEncoderConfig | None = None) -> GATEncoder:
    cfg = cfg or GNNEncoderConfig()
    return GATEncoder(cfg.in_dim, cfg.hidden_dim, cfg.out_dim)
