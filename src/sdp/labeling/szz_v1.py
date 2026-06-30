from __future__ import annotations

import re
import subprocess
from pathlib import Path


def _run_git(repo_path: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout or ""


def _parse_diff_changed_lines(diff: str) -> tuple[list[int], list[int], str]:
    added: list[int] = []
    deleted: list[int] = []
    chunks: list[str] = []
    old_line = 0
    new_line = 0
    for line in diff.splitlines():
        if line.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if m:
                old_line = int(m.group(1))
                new_line = int(m.group(2))
            chunks.append(line)
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added.append(new_line)
            new_line += 1
            chunks.append(line)
        elif line.startswith("-") and not line.startswith("---"):
            deleted.append(old_line)
            old_line += 1
            chunks.append(line)
        else:
            old_line += 1
            new_line += 1
            if line.startswith(" "):
                chunks.append(line)
    return added, deleted, "\n".join(chunks)


def blame_commit_for_line(repo_path: Path, commit: str, file_path: str, line: int) -> str | None:
    out = _run_git(repo_path, "blame", "-L", f"{line},{line}", commit, "--", file_path)
    if not out or not out.strip():
        return None
    m = re.match(r"([0-9a-f]{4,40})", out.strip())
    return m.group(1) if m else None


def find_bug_inducing_commits_for_fix(
    repo_path: Path,
    fix_commit: str,
    file_path: str,
    diff_text: str,
) -> set[str]:
    """
    Minimal SZZ v1: for deleted lines in a bug-fix diff, blame parent commit to find BIC candidates.
    """
    _, deleted, _ = _parse_diff_changed_lines(diff_text)
    if not deleted:
        return set()

    parent = _run_git(repo_path, "rev-parse", f"{fix_commit}^").strip()
    if not parent:
        return set()

    bic: set[str] = set()
    for line in deleted:
        blamed = blame_commit_for_line(repo_path, parent, file_path, line)
        if blamed and blamed != fix_commit:
            bic.add(blamed)
    return bic
