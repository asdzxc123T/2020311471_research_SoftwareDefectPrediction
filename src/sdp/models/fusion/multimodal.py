from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FusionConfig:
    text_dim: int = 768
    graph_dim: int = 128
    metric_dim: int = 4
    hidden_dim: int = 256
    dropout: float = 0.2


class MultimodalClassifier(nn.Module):
    def __init__(self, cfg: FusionConfig, *, use_text: bool = True, use_graph: bool = True, use_metrics: bool = True) -> None:
        super().__init__()
        self.use_text = use_text
        self.use_graph = use_graph
        self.use_metrics = use_metrics
        in_dim = 0
        if use_text:
            in_dim += cfg.text_dim
        if use_graph:
            in_dim += cfg.graph_dim
        if use_metrics:
            in_dim += cfg.metric_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

    def forward(
        self,
        text_emb: torch.Tensor | None = None,
        graph_emb: torch.Tensor | None = None,
        metric_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.use_text:
            if text_emb is None:
                raise ValueError("text_emb required")
            parts.append(text_emb)
        if self.use_graph:
            if graph_emb is None:
                raise ValueError("graph_emb required")
            parts.append(graph_emb)
        if self.use_metrics:
            if metric_vec is None:
                raise ValueError("metric_vec required")
            parts.append(metric_vec)
        x = torch.cat(parts, dim=-1)
        return self.mlp(x).squeeze(-1)


class TextOnlyClassifier(nn.Module):
    def __init__(self, text_dim: int = 768, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        return self.net(text_emb).squeeze(-1)


class GraphOnlyClassifier(nn.Module):
    def __init__(self, graph_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_emb: torch.Tensor) -> torch.Tensor:
        return self.net(graph_emb).squeeze(-1)
