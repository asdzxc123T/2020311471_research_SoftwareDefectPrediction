from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from fp.config.loader import load_yaml
from fp.data.io import read_table
from fp.eval.metrics import compute_classification_metrics, popt_like, recall_at_topk_loc
from fp.utils.paths import ProjectPaths, ensure_dirs, find_repo_root


def _predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
        return s
    return model.predict(X).astype(float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    args = ap.parse_args()

    root = find_repo_root()
    paths = ProjectPaths(root=root)
    ensure_dirs(paths.reports_dir, root / "reports" / "metrics")

    cfg = load_yaml(root / args.config)
    eval_cfg = cfg.get("eval", {}) or {}
    ds_path = Path(eval_cfg.get("dataset_path", "data/processed/dataset.parquet"))
    model_path = Path(eval_cfg.get("model_path", "artifacts/models/baseline.joblib"))
    label_col = str(eval_cfg.get("label_col", "is_bug_inducing"))
    effort_col = str(eval_cfg.get("effort_col", "churn"))
    topk_loc = float(eval_cfg.get("topk_loc", 0.2))

    df = read_table(root / ds_path)
    test_df = df[df["split"] == "test"].copy()
    if len(test_df) == 0:
        raise ValueError("No test rows found (split=='test').")

    model = load(root / model_path)
    X = test_df.drop(columns=[label_col], errors="ignore")
    y_true = test_df[label_col].astype(int).to_numpy()
    y_prob = _predict_proba(model, X)

    cls = compute_classification_metrics(y_true, y_prob)
    scored = test_df.copy()
    scored["y_prob"] = y_prob

    effort_recall = recall_at_topk_loc(scored, y_col=label_col, prob_col="y_prob", effort_col=effort_col, topk_loc=topk_loc)
    popt = popt_like(scored, y_col=label_col, prob_col="y_prob", effort_col=effort_col)

    out = {
        "n_test": int(len(test_df)),
        "auc": cls.auc,
        "ap": cls.ap,
        "precision": cls.precision,
        "recall": cls.recall,
        "f1": cls.f1,
        "recall_at_topk_loc": effort_recall,
        "popt_like": popt,
    }

    out_path = Path(eval_cfg.get("output_metrics_path", "reports/metrics/metrics.json"))
    out_abs = root / out_path
    out_abs.parent.mkdir(parents=True, exist_ok=True)
    out_abs.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote metrics: {out_path}")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

