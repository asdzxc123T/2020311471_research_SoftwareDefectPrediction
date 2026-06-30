from __future__ import annotations

import argparse
from pathlib import Path

from sdp.data.github_pipeline import apply_szz_v1_labels
from sdp.data.io import read_table, write_table
from sdp.labeling.szz_simple import label_bug_inducing_commits_simple
from sdp.utils.paths import find_repo_root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True, help="Commits/function table with is_bug_fixing_commit")
    ap.add_argument("--out_path", default="data/processed/commits_with_szz.parquet")
    ap.add_argument("--method", choices=["simple", "szz_v1"], default="simple")
    ap.add_argument("--repo", default=None, help="Local repo path (required for szz_v1)")
    args = ap.parse_args()

    root = find_repo_root()
    df = read_table(root / Path(args.in_path))

    if args.method == "simple":
        if "commit_hash" in df.columns and "function_id" in df.columns:
            commit_df = df.drop_duplicates(subset=["repo_name", "commit_hash"]).copy()
            labeled = label_bug_inducing_commits_simple(commit_df)
            bic_map = labeled.set_index(["repo_name", "commit_hash"])["is_bug_inducing"].to_dict()
            out = df.copy()
            out["is_bug_inducing"] = [
                bic_map.get((r["repo_name"], r["commit_hash"]), False) for _, r in out.iterrows()
            ]
            out["label_source"] = "szz_simple_v1"
        else:
            out = label_bug_inducing_commits_simple(df)
            out["label_source"] = "szz_simple_v1"
    else:
        if not args.repo:
            raise ValueError("--repo is required for szz_v1")
        out = apply_szz_v1_labels(df, Path(args.repo).resolve())
        out["label_source"] = "szz_v1"

    write_table(out, root / Path(args.out_path))
    print(f"Wrote: {args.out_path} (rows={len(out)}, bug_inducing={int(out['is_bug_inducing'].sum())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
