from __future__ import annotations

import argparse
from datetime import timezone
from pathlib import Path

import pandas as pd
from pydriller import Repository

from sdp.data.io import write_table
from sdp.utils.paths import ensure_dirs, find_repo_root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Local path to a git repository (cloned).")
    ap.add_argument("--repo_name", default=None, help="Override repo_name (default: folder name).")
    ap.add_argument("--repo_url", default=None, help="Optional repo URL for metadata.")
    ap.add_argument("--out", default="data/raw/commits.parquet", help="Output path (parquet/csv/jsonl).")
    ap.add_argument("--max_commits", type=int, default=None)
    args = ap.parse_args()

    root = find_repo_root()
    ensure_dirs(root / "data" / "raw")

    repo_path = Path(args.repo).resolve()
    repo_name = args.repo_name or repo_path.name
    repo_url = args.repo_url or ""

    rows = []
    for i, c in enumerate(Repository(str(repo_path)).traverse_commits()):
        if args.max_commits is not None and i >= args.max_commits:
            break
        commit_time = c.committer_date.replace(tzinfo=timezone.utc) if c.committer_date.tzinfo is None else c.committer_date
        rows.append(
            {
                "repo_url": repo_url,
                "repo_name": repo_name,
                "commit_hash": c.hash,
                "commit_time": commit_time,
                "message": c.msg,
                "author": c.author.name,
                "loc_added": int(c.insertions),
                "loc_deleted": int(c.deletions),
                "churn": int(c.insertions + c.deletions),
            }
        )

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    write_table(df, root / out_path)
    print(f"Wrote commits: {out_path} (rows={len(df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

