from __future__ import annotations

import argparse
from pathlib import Path

from fp.data.io import read_table, write_table
from fp.labeling.szz_simple import label_bug_inducing_commits_simple
from fp.utils.paths import find_repo_root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True, help="Commits table with is_bug_fixing_commit")
    ap.add_argument("--out_path", default="data/processed/commits_with_szz.parquet")
    args = ap.parse_args()

    root = find_repo_root()
    df = read_table(root / Path(args.in_path))
    out = label_bug_inducing_commits_simple(df)
    out["label_source"] = "szz_simple_v1"
    write_table(out, root / Path(args.out_path))
    print(f"Wrote: {args.out_path} (rows={len(out)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

