from __future__ import annotations

import re


_ISSUE_PATTERNS = [
    re.compile(r"#(\d+)\b"),
    re.compile(r"\bissue\s*#?\s*(\d+)\b", re.I),
    re.compile(r"\bfixes\s+#(\d+)\b", re.I),
    re.compile(r"\bcloses\s+#(\d+)\b", re.I),
]

_FORMAT_ONLY_PATTERNS = [
    re.compile(r"^\s*(style|format|lint|whitespace|prettier|black)\b", re.I),
    re.compile(r"^\s*merge\b", re.I),
]


def extract_issue_numbers(message: str) -> list[int]:
    nums: set[int] = set()
    for pat in _ISSUE_PATTERNS:
        for m in pat.finditer(message):
            nums.add(int(m.group(1)))
    return sorted(nums)


def is_format_only_commit(message: str) -> bool:
    m = message.strip().lower()
    if not m:
        return False
    first_line = m.splitlines()[0]
    return any(p.search(first_line) for p in _FORMAT_ONLY_PATTERNS)


def is_bug_fixing_message(message: str, patterns: list[str] | None = None) -> bool:
    from sdp.labeling.bfc import _DEFAULT_PATTERNS

    pats = patterns or _DEFAULT_PATTERNS
    m = message.lower()
    if is_format_only_commit(message):
        return False
    has_keyword = any(re.search(p, m) for p in pats)
    has_issue = len(extract_issue_numbers(message)) > 0 and any(
        kw in m for kw in ("fix", "bug", "defect", "patch", "hotfix", "issue")
    )
    return has_keyword or has_issue
