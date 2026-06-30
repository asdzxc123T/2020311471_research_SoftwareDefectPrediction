from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import load
from sklearn.metrics import precision_recall_curve

from sdp.config.loader import load_yaml
from sdp.data.io import read_table
from sdp.utils.paths import ensure_dirs, find_repo_root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    root = find_repo_root()
    cfg = load_yaml(root / args.config)
    eval_cfg = cfg.get("eval", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    label_col = str(eval_cfg.get("label_col", "is_bug_inducing"))
    ds_path = Path(eval_cfg.get("dataset_path", cfg.get("data", {}).get("output_path")))
    model_path = Path(eval_cfg.get("model_path", "artifacts/models/baseline.joblib"))
    metrics_out = Path(eval_cfg.get("output_metrics_path", "reports/metrics/metrics.json"))

    df = read_table(root / ds_path)
    split_df = df[df["split"] == args.split].copy()
    if len(split_df) == 0:
        raise ValueError(f"No rows for split={args.split}")

    model = load(root / model_path)
    X = split_df.drop(columns=[label_col], errors="ignore")
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = model.predict(X).astype(float)

    y_true = split_df[label_col].astype(int).to_numpy()
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    out_dir = root / "reports" / "metrics"
    ensure_dirs(out_dir)
    stem = metrics_out.stem
    png_path = out_dir / f"{stem}_pr_curve_{args.split}.png"
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR curve ({args.split}, n={len(split_df)})")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    ap_score = float(np.trapezoid(precision, recall)) if len(recall) > 1 else 0.0
    meta = {"split": args.split, "pr_curve_png": str(png_path.relative_to(root)), "pr_auc_trapz": ap_score}
    meta_path = out_dir / f"{stem}_pr_meta_{args.split}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
