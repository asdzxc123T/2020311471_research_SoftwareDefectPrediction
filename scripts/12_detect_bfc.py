from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sdp.data.io import read_table, write_table
from sdp.labeling.bfc_enhanced import is_bug_fixing_message
from sdp.utils.paths import find_repo_root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True, help="Input commits table (parquet/csv/jsonl)")
    ap.add_argument("--out_path", default="data/processed/commits_with_bfc.parquet")
    args = ap.parse_args()

    root = find_repo_root()
    df = read_table(root / Path(args.in_path))
    if "message" not in df.columns:
        raise KeyError("Input must contain 'message' column")
    df = df.copy()
    df["is_bug_fixing_commit"] = df["message"].astype(str).apply(is_bug_fixing_message)
    if "label_source" not in df.columns:
        df["label_source"] = "bfc_heuristic_v1"

    out_path = Path(args.out_path)
    write_table(df, root / out_path)
    print(f"Wrote: {out_path} (rows={len(df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

