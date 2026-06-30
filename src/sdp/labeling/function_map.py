from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_PARSE_STATS = {"java_javalang_ok": 0, "java_regex_ok": 0, "java_parse_fail": 0}


@dataclass(frozen=True)
class FunctionSpan:
    name: str
    start_line: int
    end_line: int


_PY_DEF = re.compile(r"^(\s*)def\s+([A-Za-z_]\w*)\s*\(", re.M)
_JAVA_METHOD = re.compile(
    r"^\s*(?:public|private|protected|static|\s)+[\w\<\>\[\]]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{",
    re.M,
)


def get_parse_stats() -> dict[str, int]:
    return dict(_PARSE_STATS)


def reset_parse_stats() -> None:
    for k in _PARSE_STATS:
        _PARSE_STATS[k] = 0


def _parse_python_functions(source: str) -> list[FunctionSpan]:
    lines = source.splitlines()
    matches = list(_PY_DEF.finditer(source))
    spans: list[FunctionSpan] = []
    for i, m in enumerate(matches):
        start = source[: m.start()].count("\n") + 1
        base_indent = len(m.group(1))
        end = len(lines)
        for j in range(start, len(lines)):
            line = lines[j]
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            if j > start - 1 and indent <= base_indent and line.lstrip().startswith(("def ", "class ")):
                end = j
                break
        spans.append(FunctionSpan(name=m.group(2), start_line=start, end_line=end))
    return spans


def _method_end_line(source: str, start_line: int) -> int:
    lines = source.splitlines()
    brace_depth = 0
    end = start_line
    started = False
    for j in range(start_line - 1, len(lines)):
        line = lines[j]
        brace_depth += line.count("{") - line.count("}")
        if "{" in line:
            started = True
        end = j + 1
        if started and brace_depth <= 0:
            break
    return max(start_line, end)


def _parse_java_functions_javalang(source: str) -> list[FunctionSpan] | None:
    try:
        import javalang
    except ImportError:
        return None
    try:
        tree = javalang.parse.parse(source)
    except Exception:
        _PARSE_STATS["java_parse_fail"] += 1
        return None

    spans: list[FunctionSpan] = []
    for _path, node in tree.filter(javalang.tree.MethodDeclaration):
        name = node.name
        if not name or name in ("if", "for", "while", "switch", "catch"):
            continue
        pos = node.position
        if pos is None:
            continue
        start = int(pos.line)
        end = _method_end_line(source, start)
        spans.append(FunctionSpan(name=name, start_line=start, end_line=end))
    if spans:
        _PARSE_STATS["java_javalang_ok"] += 1
        return spans
    return None


def _parse_java_functions_regex(source: str) -> list[FunctionSpan]:
    lines = source.splitlines()
    spans: list[FunctionSpan] = []
    for m in _JAVA_METHOD.finditer(source):
        name = m.group(1)
        if name in ("if", "for", "while", "switch", "catch"):
            continue
        start = source[: m.start()].count("\n") + 1
        end = _method_end_line(source, start)
        spans.append(FunctionSpan(name=name, start_line=start, end_line=end))
    if spans:
        _PARSE_STATS["java_regex_ok"] += 1
    return spans


def _parse_java_functions(source: str) -> list[FunctionSpan]:
    javalang_spans = _parse_java_functions_javalang(source)
    if javalang_spans:
        return javalang_spans
    return _parse_java_functions_regex(source)


def parse_functions(source: str, language: str | None) -> list[FunctionSpan]:
    lang = (language or "").lower()
    if lang in {"py", "python"}:
        return _parse_python_functions(source)
    if lang in {"java"}:
        return _parse_java_functions(source)
    if "def " in source:
        return _parse_python_functions(source)
    return _parse_java_functions(source)


def map_lines_to_functions(changed_lines: list[int], functions: list[FunctionSpan]) -> list[FunctionSpan]:
    hit: list[FunctionSpan] = []
    for fn in functions:
        if any(fn.start_line <= ln <= fn.end_line for ln in changed_lines):
            hit.append(fn)
    return hit


def infer_language(file_path: str) -> str | None:
    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    return {"py": "python", "java": "java", "js": "javascript", "ts": "typescript"}.get(ext, ext or None)
