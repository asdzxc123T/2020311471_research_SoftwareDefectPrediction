from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from sdp.config.loader import load_yaml
from sdp.data.io import read_table
from sdp.eval.metrics import compute_classification_metrics
from sdp.utils.paths import ensure_dirs, find_repo_root


def flip_labels(y: np.ndarray, flip_prob: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y2 = y.copy()
    mask = rng.random(y2.shape[0]) < flip_prob
    y2[mask] = 1 - y2[mask]
    return y2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--flip_probs", default="0.0,0.1,0.2,0.3,0.4")
    args = ap.parse_args()

    root = find_repo_root()
    cfg = load_yaml(root / args.config)
    seed = int(cfg.get("project", {}).get("seed", 42))

    eval_cfg = cfg.get("eval", {}) or {}
    ds_path = Path(eval_cfg.get("dataset_path", cfg.get("data", {}).get("output_path", "data/processed/dataset.parquet")))
    model_path = Path(eval_cfg.get("model_path", "artifacts/models/baseline.joblib"))
    label_col = str(eval_cfg.get("label_col", "is_bug_inducing"))

    df = read_table(root / ds_path)
    test_df = df[df["split"] == "test"].copy()
    if len(test_df) == 0:
        raise ValueError("No test rows found.")

    model = load(root / model_path)
    X = test_df.drop(columns=[label_col], errors="ignore")
    y_true = test_df[label_col].astype(int).to_numpy()
    y_prob = model.predict_proba(X)[:, 1]

    out_rows = []
    for p in [float(x.strip()) for x in args.flip_probs.split(",") if x.strip()]:
        y_flip = flip_labels(y_true, p, seed=seed)
        m = compute_classification_metrics(y_flip, y_prob)
        out_rows.append({"flip_prob": p, "auc": m.auc, "ap": m.ap, "precision": m.precision, "recall": m.recall, "f1": m.f1})

    out_dir = root / "reports" / "robustness"
    ensure_dirs(out_dir)
    out_path = out_dir / "label_flipping.json"
    out_path.write_text(json.dumps(out_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out_rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

