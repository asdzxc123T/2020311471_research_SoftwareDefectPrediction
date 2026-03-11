from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
from joblib import load
from sklearn.pipeline import Pipeline

from sdp.config.loader import load_yaml
from sdp.data.io import read_table
from sdp.utils.paths import ensure_dirs, find_repo_root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_rows", type=int, default=500)
    args = ap.parse_args()

    root = find_repo_root()
    cfg = load_yaml(root / args.config)
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
    if len(X) > args.max_rows:
        X = X.sample(n=args.max_rows, random_state=0)

    out_dir = root / "reports" / "xai"
    ensure_dirs(out_dir)

    try:
        if isinstance(model, Pipeline) and "pre" in model.named_steps and "clf" in model.named_steps:
            pre = model.named_steps["pre"]
            clf = model.named_steps["clf"]
            Xt = pre.transform(X)
            try:
                feature_names = pre.get_feature_names_out()
            except Exception:
                feature_names = None

            # RandomForest 등 트리 모델은 TreeExplainer가 안정적
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(Xt)

            out_path = out_dir / "shap_summary.png"
            plt.figure()
            shap.summary_plot(shap_values, Xt, feature_names=feature_names, show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
            print(f"Wrote: {out_path.relative_to(root)}")
        else:
            # 일반 케이스: callable 기반으로 kernel 방식(비용 큼). 작은 샘플에만 사용.
            X_small = X.copy()
            if len(X_small) > 200:
                X_small = X_small.sample(n=200, random_state=0)
            f = lambda Z: model.predict_proba(Z)[:, 1]  # noqa: E731
            explainer = shap.KernelExplainer(f, X_small)
            vals = explainer.shap_values(X_small, nsamples=200)
            out_path = out_dir / "shap_summary_kernel.png"
            plt.figure()
            shap.summary_plot(vals, X_small, show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
            print(f"Wrote: {out_path.relative_to(root)}")
    except Exception as e:
        # Minimal fallback
        out_path = out_dir / "shap_error.txt"
        out_path.write_text(str(e), encoding="utf-8")
        print(f"SHAP failed, wrote error: {out_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

