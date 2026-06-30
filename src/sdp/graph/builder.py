from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from sdp.graph.ast_builder import GraphSample, _python_ast_to_graph


class PythonASTGraphBuilder:
    def build(self, code: str, *, language: str | None = None, cpg_path: str | None = None) -> GraphSample:
        return _python_ast_to_graph(code)


class JavaASTGraphBuilder:
    def build(self, code: str, *, language: str | None = None, cpg_path: str | None = None) -> GraphSample:
        from sdp.graph.java_ast_builder import java_code_to_ast_graph

        return java_code_to_ast_graph(code)


class JoernGraphBuilder:
    """Placeholder for future Joern CPG integration."""

    def build(self, code: str, *, language: str | None = None, cpg_path: str | None = None) -> GraphSample:
        if cpg_path and Path(cpg_path).exists():
            import json

            meta = json.loads(Path(cpg_path).read_text(encoding="utf-8"))
            if meta.get("status") == "joern_export_complete" and meta.get("edge_index"):
                pass  # Future: load real CPG edges from Joern export
        return build_code_graph(code, language=language)


def build_code_graph(
    code: str,
    *,
    language: str | None = None,
    cpg_path: str | None = None,
    max_nodes: int = 128,
) -> GraphSample:
    lang = (language or "").lower()
    if cpg_path and Path(cpg_path).exists():
        return JoernGraphBuilder().build(code, language=language, cpg_path=cpg_path)
    if lang in {"java"}:
        from sdp.graph.java_ast_builder import java_code_to_ast_graph

        return java_code_to_ast_graph(code, max_nodes=max_nodes)
    if lang in {"py", "python"} or "def " in (code or ""):
        return _python_ast_to_graph(code, max_nodes=max_nodes)
    sample = _python_ast_to_graph(code, max_nodes=max_nodes)
    if sample.num_nodes <= 1:
        from sdp.graph.java_ast_builder import java_code_to_ast_graph

        return java_code_to_ast_graph(code, max_nodes=max_nodes)
    return sample


def code_to_ast_graph(code: str, *, language: str | None = None, max_nodes: int = 128) -> GraphSample:
    return build_code_graph(code, language=language, max_nodes=max_nodes)
