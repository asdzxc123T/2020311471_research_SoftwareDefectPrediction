from __future__ import annotations

import torch

from sdp.graph.ast_builder import FEAT_DIM, GraphSample


def _node_features(node_types: list[str]) -> torch.Tensor:
    x = torch.zeros((len(node_types), FEAT_DIM), dtype=torch.float32)
    for i, t in enumerate(node_types):
        x[i, hash(t) % FEAT_DIM] = 1.0
    return x


def _spans_to_graph(nodes: list[str], edges: list[tuple[int, int]]) -> GraphSample:
    x = _node_features(nodes)
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return GraphSample(node_features=x, edge_index=edge_index, num_nodes=len(nodes))


def java_code_to_ast_graph(code: str, *, max_nodes: int = 128) -> GraphSample:
    try:
        import javalang
    except ImportError:
        return _spans_to_graph(["Empty"], [])

    try:
        tree = javalang.parse.parse(code or "class X {}")
    except Exception:
        return _spans_to_graph(["ParseError"], [])

    nodes: list[str] = []
    edges: list[tuple[int, int]] = []

    def add_node(label: str, parent: int | None) -> int:
        if len(nodes) >= max_nodes:
            return parent if parent is not None else 0
        idx = len(nodes)
        nodes.append(label)
        if parent is not None:
            edges.append((parent, idx))
        return idx

    root = add_node("CompilationUnit", None)
    for _path, node in tree.filter(javalang.tree.TypeDeclaration):
        if len(nodes) >= max_nodes:
            break
        type_idx = add_node(node.__class__.__name__, root)
        for _p2, member in node.filter(javalang.tree.MethodDeclaration):
            if len(nodes) >= max_nodes:
                break
            m_idx = add_node(f"Method:{member.name}", type_idx)
            if member.body:
                for stmt in member.body:
                    if len(nodes) >= max_nodes:
                        break
                    add_node(stmt.__class__.__name__, m_idx)

    if len(nodes) <= 1:
        return _spans_to_graph(["EmptyJava"], [])

    return _spans_to_graph(nodes, edges)
