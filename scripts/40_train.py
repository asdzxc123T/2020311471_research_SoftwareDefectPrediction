from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fp.config.loader import load_yaml
from fp.data.io import read_table
from fp.models.baseline.train import BaselineTrainConfig, save_model, train_baseline_model
from fp.utils.paths import ProjectPaths, ensure_dirs, find_repo_root
from fp.utils.seed import set_global_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--seed", type=int, default=None, help="Override project seed")
    args = ap.parse_args()

    root = find_repo_root()
    paths = ProjectPaths(root=root)
    ensure_dirs(paths.data_dir, paths.artifacts_dir, paths.reports_dir, root / "artifacts" / "models")

    config = load_yaml(root / args.config)
    seed = int(args.seed if args.seed is not None else config.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    data_cfg = config.get("data", {}) or {}
    ds_path = Path(data_cfg.get("output_path", "data/processed/dataset.parquet"))
    df = read_table(root / ds_path)

    train_cfg = config.get("train", {}) or {}
    task = str(train_cfg.get("task", "baseline_rf"))
    features_cfg = train_cfg.get("features", {}) or {}
    numeric = list(features_cfg.get("numeric", ["loc_added", "loc_deleted", "churn"]))
    text = features_cfg.get("text", None)
    label_col = str(train_cfg.get("label_col", "is_bug_inducing"))

    train_df = df[df["split"] == "train"].copy()
    if len(train_df) == 0:
        raise ValueError("No training rows found (split=='train').")

    model = train_baseline_model(
        train_df,
        BaselineTrainConfig(task=task, numeric_features=numeric, text_feature=text, label_col=label_col, seed=seed),
    )

    out_path = Path(train_cfg.get("output_model_path", "artifacts/models/baseline.joblib"))
    out_abs = root / out_path
    out_abs.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, str(out_abs))
    print(f"Saved model: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

