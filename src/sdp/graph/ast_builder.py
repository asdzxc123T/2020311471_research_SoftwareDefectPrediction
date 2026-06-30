from __future__ import annotations

import ast
from dataclasses import dataclass

import torch

FEAT_DIM = 32


@dataclass
class GraphSample:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    num_nodes: int


def _python_ast_to_graph(code: str, *, max_nodes: int = 128) -> GraphSample:
    try:
        tree = ast.parse(code or "pass")
    except SyntaxError:
        tree = ast.parse("pass")

    nodes: list[str] = []
    edges: list[tuple[int, int]] = []

    def visit(node: ast.AST, parent: int | None) -> None:
        if len(nodes) >= max_nodes:
            return
        idx = len(nodes)
        nodes.append(node.__class__.__name__)
        if parent is not None:
            edges.append((parent, idx))
        for child in ast.iter_child_nodes(node):
            visit(child, idx)

    visit(tree, None)

    x = torch.zeros((len(nodes), FEAT_DIM), dtype=torch.float32)
    for i, t in enumerate(nodes):
        x[i, hash(t) % FEAT_DIM] = 1.0

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return GraphSample(node_features=x, edge_index=edge_index, num_nodes=len(nodes))


def code_to_ast_graph(code: str, *, max_nodes: int = 128) -> GraphSample:
    from sdp.graph.builder import build_code_graph

    return build_code_graph(code, max_nodes=max_nodes)
