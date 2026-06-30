from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sdp.config.loader import load_yaml
from sdp.data.io import read_table
from sdp.utils.paths import ensure_dirs, find_repo_root


def validate_dataset(df: pd.DataFrame, *, label_col: str = "is_bug_inducing") -> dict:
    report: dict = {
        "n_rows": int(len(df)),
        "n_repos": int(df["repo_name"].nunique()) if "repo_name" in df.columns else None,
        "missing_commit_time": int(df["commit_time"].isna().sum()) if "commit_time" in df.columns else None,
    }

    if label_col in df.columns:
        pos = int(df[label_col].astype(int).sum())
        report["label_positive"] = pos
        report["label_positive_rate"] = float(pos / max(1, len(df)))

    if "split" in df.columns:
        report["split_counts"] = df["split"].value_counts().to_dict()
        if label_col in df.columns:
            report["split_positive"] = df.groupby("split")[label_col].sum().astype(int).to_dict()

    if "commit_hash" in df.columns and "split" in df.columns:
        leak = (
            df.groupby("commit_hash")["split"].nunique().reset_index(name="n_splits")
        )
        violations = int((leak["n_splits"] > 1).sum())
        report["split_leakage_commits"] = violations

    if "function_id" in df.columns:
        fid = df["function_id"].astype(str)
        file_level = int(fid.str.endswith(":__file__").sum())
        report["file_level_records"] = file_level
        report["file_level_rate"] = float(file_level / max(1, len(df)))
        report["function_mapping_rate"] = float(1.0 - file_level / max(1, len(df)))

    if "code_text" in df.columns:
        missing_code = int((df["code_text"].isna() | (df["code_text"].astype(str).str.strip() == "")).sum())
        report["code_text_missing"] = missing_code
        report["code_text_missing_rate"] = float(missing_code / max(1, len(df)))

    if "cyclomatic_complexity" in df.columns:
        report["complexity_missing"] = int(df["cyclomatic_complexity"].isna().sum())

    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp/github_multi.yaml")
    ap.add_argument("--dataset", default=None, help="Override dataset path")
    ap.add_argument("--label_col", default="is_bug_inducing")
    ap.add_argument("--out", default="reports/data/dataset_report.json")
    args = ap.parse_args()

    root = find_repo_root()
    cfg = load_yaml(root / args.config)
    data_cfg = cfg.get("data", {}) or {}
    ds_path = Path(args.dataset or data_cfg.get("output_path", "data/processed/github_dataset.parquet"))

    df = read_table(root / ds_path)
    report = validate_dataset(df, label_col=args.label_col)
    report["dataset_path"] = str(ds_path)

    out_path = root / args.out
    ensure_dirs(out_path.parent)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
