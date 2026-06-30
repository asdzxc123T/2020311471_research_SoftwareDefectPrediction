from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from sdp.config.loader import load_yaml
from sdp.utils.paths import ensure_dirs, find_repo_root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--flip_probs", default="0.0,0.1,0.2,0.3,0.4")
    args = ap.parse_args()

    root = find_repo_root()
    cfg = load_yaml(root / args.config)
    train_cfg = cfg.get("train", {}) or {}
    task = str(train_cfg.get("task", "baseline_rf"))

    if task in {"codebert_only", "gnn_only", "multimodal_fusion"}:
        # Neural models: retrain with flipped labels on train split, evaluate on test
        from sdp.data.io import read_table
        from sdp.data.splits import apply_temporal_split, TemporalSplitConfig
        from sdp.models.neural.train_utils import NeuralTrainConfig, predict_neural, train_neural_model
        from sdp.eval.metrics import compute_classification_metrics
        import numpy as np
        import torch

        ds_path = Path(cfg["data"]["output_path"])
        df = read_table(root / ds_path)
        test_df = df[df["split"] == "test"].copy()
        train_df = df[df["split"] == "train"].copy()
        label_col = train_cfg.get("label_col", "is_bug_inducing")
        flip_probs = [float(x.strip()) for x in args.flip_probs.split(",") if x.strip()]
        seed = int(cfg.get("project", {}).get("seed", 42))
        rng = np.random.default_rng(seed)

        features_cfg = train_cfg.get("features", {}) or {}
        neural_cfg = NeuralTrainConfig(
            task=task,
            label_col=label_col,
            metric_cols=list(features_cfg.get("numeric", ["loc_added", "loc_deleted", "churn", "cyclomatic_complexity"])),
            text_col=str(features_cfg.get("text", "diff_text")),
            code_col=str(features_cfg.get("code", "code_text")),
            epochs=int(train_cfg.get("epochs", 2)),
            batch_size=int(train_cfg.get("batch_size", 4)),
            lr=float(train_cfg.get("lr", 1e-4)),
            seed=seed,
        )

        out_rows = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for p in flip_probs:
            flipped = train_df.copy()
            y = flipped[label_col].astype(int).to_numpy().copy()
            mask = rng.random(len(y)) < p
            y[mask] = 1 - y[mask]
            flipped[label_col] = y
            model, components, _ = train_neural_model(flipped, neural_cfg, device)
            y_prob = predict_neural(model, components, test_df, neural_cfg, device)
            y_true = test_df[label_col].astype(int).to_numpy()
            m = compute_classification_metrics(y_true, y_prob)
            out_rows.append({"flip_prob": p, "auc": m.auc, "ap": m.ap, "precision": m.precision, "recall": m.recall, "f1": m.f1})

        out_dir = root / "reports" / "robustness"
        ensure_dirs(out_dir)
        out_path = out_dir / f"label_flipping_{task}.json"
        out_path.write_text(json.dumps(out_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(out_rows, indent=2, ensure_ascii=False))
        return 0

    # Baseline path via subprocess to existing script
    subprocess.run(
        ["python", "scripts/70_robustness.py", "--config", args.config, "--flip_probs", args.flip_probs],
        cwd=root,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
