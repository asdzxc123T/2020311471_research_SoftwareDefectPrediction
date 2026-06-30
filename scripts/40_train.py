from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sdp.config.loader import load_yaml
from sdp.data.io import read_table
from sdp.models.baseline.train import BaselineTrainConfig, save_model, train_baseline_model
from sdp.models.neural.train_utils import NeuralTrainConfig, train_neural_model
from sdp.utils.paths import ProjectPaths, ensure_dirs, find_repo_root
from sdp.utils.seed import set_global_seed

NEURAL_TASKS = {"codebert_only", "gnn_only", "multimodal_fusion"}


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
    label_col = str(train_cfg.get("label_col", "is_bug_inducing"))
    out_path = Path(train_cfg.get("output_model_path", "artifacts/models/baseline.joblib"))
    out_abs = root / out_path
    out_abs.parent.mkdir(parents=True, exist_ok=True)

    train_df = df[df["split"] == "train"].copy()
    if len(train_df) == 0:
        raise ValueError("No training rows found (split=='train').")

    subset = int(data_cfg.get("train_subset", 0) or 0)
    if subset > 0 and len(train_df) > subset:
        train_df = train_df.sample(n=subset, random_state=seed)

    if task in NEURAL_TASKS:
        features_cfg = train_cfg.get("features", {}) or {}
        neural_cfg = NeuralTrainConfig(
            task=task,
            label_col=label_col,
            metric_cols=list(features_cfg.get("numeric", ["loc_added", "loc_deleted", "churn", "cyclomatic_complexity"])),
            text_col=str(features_cfg.get("text", "diff_text")),
            code_col=str(features_cfg.get("code", "code_text")),
            epochs=int(train_cfg.get("epochs", 3)),
            batch_size=int(train_cfg.get("batch_size", 8)),
            lr=float(train_cfg.get("lr", 1e-4)),
            seed=seed,
        )
        model, components, losses = train_neural_model(train_df, neural_cfg)
        torch.save(
            {
                "task": task,
                "model_state": model.state_dict(),
                "components": {k: v.state_dict() if v is not None else None for k, v in components.items()},
                "neural_cfg": neural_cfg.__dict__,
                "losses": losses,
            },
            out_abs,
        )
        print(f"Saved neural model: {out_path} (final_loss={losses[-1] if losses else 'n/a'})")
    else:
        features_cfg = train_cfg.get("features", {}) or {}
        numeric = list(features_cfg.get("numeric", ["loc_added", "loc_deleted", "churn"]))
        text = features_cfg.get("text", None)
        model = train_baseline_model(
            train_df,
            BaselineTrainConfig(task=task, numeric_features=numeric, text_feature=text, label_col=label_col, seed=seed),
        )
        save_model(model, str(out_abs))
        print(f"Saved model: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
