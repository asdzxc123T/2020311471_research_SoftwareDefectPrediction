from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import load

from sdp.config.loader import load_yaml
from sdp.data.io import read_table
from sdp.eval.metrics import (
    compute_classification_metrics,
    find_best_threshold,
    popt_like,
    recall_at_topk_loc,
)
from sdp.models.fusion.multimodal import FusionConfig, GraphOnlyClassifier, MultimodalClassifier, TextOnlyClassifier
from sdp.models.gnn.gat_encoder import GNNEncoderConfig, build_gnn_encoder
from sdp.models.neural.train_utils import NeuralTrainConfig, predict_neural
from sdp.models.text.codebert_encoder import CodeBERTConfig, build_codebert_encoder
from sdp.utils.paths import ProjectPaths, ensure_dirs, find_repo_root

NEURAL_TASKS = {"codebert_only", "gnn_only", "multimodal_fusion"}


def _predict_proba_sklearn(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
        return s
    return model.predict(X).astype(float)


def _load_neural_bundle(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    task = ckpt["task"]
    cfg_dict = ckpt["neural_cfg"]
    neural_cfg = NeuralTrainConfig(**cfg_dict)

    if task == "codebert_only":
        text_enc = build_codebert_encoder(CodeBERTConfig(freeze=True)).to(device)
        model = TextOnlyClassifier().to(device)
        components = {"text": text_enc, "graph": None}
    elif task == "gnn_only":
        graph_enc = build_gnn_encoder(GNNEncoderConfig(in_dim=32)).to(device)
        model = GraphOnlyClassifier().to(device)
        components = {"text": None, "graph": graph_enc}
    else:
        text_enc = build_codebert_encoder(CodeBERTConfig(freeze=True)).to(device)
        graph_enc = build_gnn_encoder(GNNEncoderConfig(in_dim=32)).to(device)
        model = MultimodalClassifier(FusionConfig(metric_dim=len(neural_cfg.metric_cols or []))).to(device)
        components = {"text": text_enc, "graph": graph_enc}

    model.load_state_dict(ckpt["model_state"])
    for k, comp in components.items():
        state = ckpt["components"].get(k)
        if comp is not None and state is not None:
            comp.load_state_dict(state)
    return model, components, neural_cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    args = ap.parse_args()

    root = find_repo_root()
    paths = ProjectPaths(root=root)
    ensure_dirs(paths.reports_dir, root / "reports" / "metrics")

    cfg = load_yaml(root / args.config)
    eval_cfg = cfg.get("eval", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    task = str(train_cfg.get("task", "baseline_rf"))
    ds_path = Path(eval_cfg.get("dataset_path", "data/processed/dataset.parquet"))
    model_path = Path(eval_cfg.get("model_path", "artifacts/models/baseline.joblib"))
    label_col = str(eval_cfg.get("label_col", "is_bug_inducing"))
    effort_col = str(eval_cfg.get("effort_col", "churn"))
    topk_loc = float(eval_cfg.get("topk_loc", 0.2))

    df = read_table(root / ds_path)
    test_df = df[df["split"] == "test"].copy()
    val_df = df[df["split"] == "val"].copy()
    if len(test_df) == 0:
        raise ValueError("No test rows found (split=='test').")

    if task in NEURAL_TASKS:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, components, neural_cfg = _load_neural_bundle(root / model_path, device)
        y_prob_test = predict_neural(model, components, test_df, neural_cfg, device)
        y_prob_val = predict_neural(model, components, val_df, neural_cfg, device) if len(val_df) > 0 else None
    else:
        model = load(root / model_path)
        X_test = test_df.drop(columns=[label_col], errors="ignore")
        y_prob_test = _predict_proba_sklearn(model, X_test)
        y_prob_val = None
        if len(val_df) > 0:
            X_val = val_df.drop(columns=[label_col], errors="ignore")
            y_prob_val = _predict_proba_sklearn(model, X_val)

    y_true = test_df[label_col].astype(int).to_numpy()
    cls_default = compute_classification_metrics(y_true, y_prob_test, threshold=0.5)
    tuned_threshold = 0.5
    cls_tuned = cls_default
    if y_prob_val is not None and len(val_df) > 0:
        y_val = val_df[label_col].astype(int).to_numpy()
        tuned_threshold = find_best_threshold(y_val, y_prob_val)
        cls_tuned = compute_classification_metrics(y_true, y_prob_test, threshold=tuned_threshold)

    scored = test_df.copy()
    scored["y_prob"] = y_prob_test

    effort_recall = recall_at_topk_loc(scored, y_col=label_col, prob_col="y_prob", effort_col=effort_col, topk_loc=topk_loc)
    popt = popt_like(scored, y_col=label_col, prob_col="y_prob", effort_col=effort_col)

    out = {
        "task": task,
        "n_test": int(len(test_df)),
        "n_val": int(len(val_df)),
        "threshold_default": 0.5,
        "threshold_tuned": tuned_threshold,
        "auc": cls_default.auc,
        "ap": cls_default.ap,
        "precision": cls_default.precision,
        "recall": cls_default.recall,
        "f1": cls_default.f1,
        "precision_at_tuned": cls_tuned.precision,
        "recall_at_tuned": cls_tuned.recall,
        "f1_at_tuned": cls_tuned.f1,
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
