from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sdp.config.loader import load_yaml
from sdp.data.io import read_table, write_table
from sdp.utils.paths import ensure_dirs, find_repo_root


def merge_labeled_datasets(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
        df = read_table(p)
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    return merged.drop_duplicates(subset=["repo_name", "commit_hash", "function_id"], keep="first")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp/github_multi.yaml")
    ap.add_argument("--inputs", nargs="*", default=None, help="Override labeled parquet paths")
    ap.add_argument("--output", default=None, help="Override merged output path")
    args = ap.parse_args()

    root = find_repo_root()
    cfg = load_yaml(root / args.config)
    merge_cfg = cfg.get("merge", {}) or {}

    if args.inputs:
        inputs = [root / p for p in args.inputs]
    else:
        inputs = [root / Path(p) for p in merge_cfg.get("inputs", [])]

    out_path = root / Path(args.output or merge_cfg.get("output", "data/processed/github_labeled_merged.parquet"))
    ensure_dirs(out_path.parent)

    merged = merge_labeled_datasets(inputs)
    write_table(merged, out_path)
    print(f"Merged {len(inputs)} files -> {out_path.relative_to(root)} (rows={len(merged)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
