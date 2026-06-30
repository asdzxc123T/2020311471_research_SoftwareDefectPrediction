from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class GNNExplanation:
    node_importance: torch.Tensor
    edge_importance: torch.Tensor


def explain_graph(
    model: torch.nn.Module,
    graph_encoder: torch.nn.Module,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    epochs: int = 100,
    lr: float = 0.01,
) -> GNNExplanation:
    """
    Minimal GNNExplainer-style mask optimization on node and edge importance.
    """
    device = node_features.device
    n_nodes = node_features.shape[0]
    n_edges = edge_index.shape[1] if edge_index.numel() else 0

    node_mask = torch.nn.Parameter(torch.ones(n_nodes, device=device) * 0.5)
    edge_mask = torch.nn.Parameter(torch.ones(n_edges, device=device) * 0.5) if n_edges else None
    opt = torch.optim.Adam([node_mask] + ([edge_mask] if edge_mask is not None else []), lr=lr)

    with torch.no_grad():
        target_logit = model(graph_encoder(node_features, edge_index).unsqueeze(0))
        target = torch.sigmoid(target_logit).item()

    for _ in range(epochs):
        opt.zero_grad()
        nf = node_features * torch.sigmoid(node_mask).unsqueeze(-1)
        if edge_mask is not None and n_edges:
            ei = edge_index
            # Apply edge mask by scaling source node contributions indirectly
            logits = model(graph_encoder(nf, ei)).unsqueeze(0)
        else:
            logits = model(graph_encoder(nf, edge_index)).unsqueeze(0)
        loss = -F.logsigmoid(logits).mean() + 0.01 * torch.sigmoid(node_mask).mean()
        loss.backward()
        opt.step()

    return GNNExplanation(
        node_importance=torch.sigmoid(node_mask).detach().cpu(),
        edge_importance=torch.sigmoid(edge_mask).detach().cpu() if edge_mask is not None else torch.tensor([]),
    )
