from __future__ import annotations

import argparse
from pathlib import Path

from sdp.data.github_pipeline import mine_function_records
from sdp.data.io import write_table
from sdp.utils.paths import ensure_dirs, find_repo_root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Local path to a git repository (cloned).")
    ap.add_argument("--repo_name", default=None, help="Override repo_name (default: folder name).")
    ap.add_argument("--repo_url", default=None, help="Optional repo URL for metadata.")
    ap.add_argument("--out", default="data/raw/function_records.parquet")
    ap.add_argument("--max_commits", type=int, default=None)
    args = ap.parse_args()

    root = find_repo_root()
    ensure_dirs(root / "data" / "raw")

    repo_path = Path(args.repo).resolve()
    repo_name = args.repo_name or repo_path.name
    repo_url = args.repo_url or ""

    df = mine_function_records(
        repo_path,
        repo_name=repo_name,
        repo_url=repo_url,
        max_commits=args.max_commits,
    )
    out_path = Path(args.out)
    write_table(df, root / out_path)
    print(f"Wrote function records: {out_path} (rows={len(df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
