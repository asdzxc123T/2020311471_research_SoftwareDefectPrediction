from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sdp.config.loader import load_yaml
from sdp.data.io import read_table
from sdp.utils.paths import ensure_dirs, find_repo_root
from sdp.utils.seed import set_global_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--n", type=int, default=200, help="Number of samples to audit")
    ap.add_argument("--seed", type=int, default=None, help="Override seed")
    args = ap.parse_args()

    root = find_repo_root()
    cfg = load_yaml(root / args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    ds_path = Path(cfg.get("eval", {}).get("dataset_path", cfg.get("data", {}).get("output_path", "data/processed/dataset.parquet")))
    df = read_table(root / ds_path)

    # In the real pipeline, a human verifies sampled labels.
    # This script generates an audit sheet + computes precision once human_verified is filled.
    rng = np.random.default_rng(seed)
    n = min(args.n, len(df))
    sample_idx = rng.choice(df.index.to_numpy(), size=n, replace=False)
    sample = df.loc[sample_idx].copy()

    sample["human_verified_is_bug_inducing"] = np.nan
    sample["human_notes"] = ""

    out_dir = root / "reports" / "labeling"
    ensure_dirs(out_dir)
    audit_path = out_dir / "label_audit_sheet.csv"
    sample.to_csv(audit_path, index=False)

    # If user already filled it, compute precision.
    filled = sample.dropna(subset=["human_verified_is_bug_inducing"])
    precision = None
    if len(filled) > 0 and "is_bug_inducing" in filled.columns:
        pred = filled["is_bug_inducing"].astype(int).to_numpy()
        human = filled["human_verified_is_bug_inducing"].astype(int).to_numpy()
        tp = int(((pred == 1) & (human == 1)).sum())
        fp = int(((pred == 1) & (human == 0)).sum())
        precision = float(tp / max(1, tp + fp))

    report = {
        "dataset_path": str(ds_path),
        "seed": seed,
        "n_sampled": int(n),
        "audit_sheet": str(audit_path.relative_to(root)),
        "precision_if_filled": precision,
        "note": "Fill human_verified_is_bug_inducing in audit_sheet then re-run for precision.",
    }
    report_path = out_dir / "label_precision_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

