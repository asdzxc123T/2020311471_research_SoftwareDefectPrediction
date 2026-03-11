from __future__ import annotations

import re


_DEFAULT_PATTERNS = [
    r"\bfix\b",
    r"\bbug\b",
    r"\bdefect\b",
    r"\bhotfix\b",
    r"\bpatch\b",
    r"\bissue\b",
]


def is_bug_fixing_message(message: str, patterns: list[str] | None = None) -> bool:
    pats = patterns or _DEFAULT_PATTERNS
    m = message.lower()
    return any(re.search(p, m) for p in pats)

