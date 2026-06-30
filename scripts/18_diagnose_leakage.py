from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from sdp.config.loader import load_yaml
from sdp.data.io import read_table
from sdp.utils.paths import ensure_dirs, find_repo_root


def _single_feature_auc(df: pd.DataFrame, col: str, label_col: str) -> float | None:
    if col not in df.columns or label_col not in df.columns:
        return None
    y = df[label_col].astype(int).to_numpy()
    x = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy()
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, x))
    except Exception:
        return None


def diagnose_leakage(df: pd.DataFrame, *, label_col: str = "is_bug_inducing") -> dict:
    report: dict = {"n_rows": int(len(df))}

    numeric_cols = [
        c
        for c in ["loc_added", "loc_deleted", "churn", "loc_current", "cyclomatic_complexity"]
        if c in df.columns
    ]
    report["single_feature_auc"] = {
        c: _single_feature_auc(df, c, label_col) for c in numeric_cols
    }

    if "label_source" in df.columns and label_col in df.columns:
        by_source = {}
        for src, g in df.groupby("label_source"):
            pos_rate = float(g[label_col].astype(int).mean())
            by_source[str(src)] = {"n": int(len(g)), "positive_rate": pos_rate}
        report["label_source_breakdown"] = by_source
        # Leak if any source is 100% or 0% positive with large n
        report["label_source_leak_suspect"] = [
            k
            for k, v in by_source.items()
            if v["n"] >= 10 and (v["positive_rate"] <= 0.001 or v["positive_rate"] >= 0.999)
        ]

    if "commit_hash" in df.columns and "split" in df.columns:
        train_hashes = set(df[df["split"] == "train"]["commit_hash"].astype(str))
        test_hashes = set(df[df["split"] == "test"]["commit_hash"].astype(str))
        report["train_test_commit_overlap"] = int(len(train_hashes & test_hashes))

    if "function_id" in df.columns:
        report["duplicate_function_ids"] = int(df["function_id"].duplicated().sum())

    suspicious_aucs = [
        k for k, v in report.get("single_feature_auc", {}).items() if v is not None and (v >= 0.95 or v <= 0.05)
    ]
    report["suspicious_feature_aucs"] = suspicious_aucs
    report["leakage_clean"] = len(suspicious_aucs) == 0 and not report.get("label_source_leak_suspect")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default="reports/data/leakage_report.json")
    args = ap.parse_args()

    root = find_repo_root()
    cfg = load_yaml(root / args.config)
    data_cfg = cfg.get("data", {}) or {}
    ds_path = Path(args.dataset or data_cfg.get("output_path", "data/processed/dataset.parquet"))
    label_col = str(cfg.get("eval", {}).get("label_col", "is_bug_inducing"))

    df = read_table(root / ds_path)
    report = diagnose_leakage(df, label_col=label_col)
    report["dataset_path"] = str(ds_path)

    out_path = root / args.out
    ensure_dirs(out_path.parent)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
