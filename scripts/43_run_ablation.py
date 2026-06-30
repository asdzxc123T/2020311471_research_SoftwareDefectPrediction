from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from sdp.config.loader import load_yaml
from sdp.utils.paths import ensure_dirs, find_repo_root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp/multimodal_fusion.yaml")
    args = ap.parse_args()

    root = find_repo_root()
    cfg = load_yaml(root / args.config)
    ablation_cfg = cfg.get("ablation", {}) or {}
    if ablation_cfg.get("variants"):
        name_to_cfg = {
            "static_only": ablation_cfg.get("static_config", "configs/exp/ablation_static_only.yaml"),
            "text_only": ablation_cfg.get("text_config", "configs/exp/codebert_only.yaml"),
            "graph_only": ablation_cfg.get("graph_config", "configs/exp/gnn_only.yaml"),
            "full_fusion": ablation_cfg.get("fusion_config", args.config),
        }
        variants = [(v, name_to_cfg.get(v, args.config)) for v in ablation_cfg["variants"]]
    elif "github" in args.config:
        variants = [
            ("static_only", "configs/exp/ablation_github_static_only.yaml"),
            ("text_only", "configs/exp/ablation_github_text_only.yaml"),
            ("graph_only", "configs/exp/ablation_github_graph_only.yaml"),
            ("full_fusion", args.config),
        ]
    else:
        variants = [
            ("static_only", "configs/exp/ablation_static_only.yaml"),
            ("text_only", "configs/exp/codebert_only.yaml"),
            ("graph_only", "configs/exp/gnn_only.yaml"),
            ("full_fusion", args.config),
        ]

    out_dir = root / "reports" / "ablation"
    ensure_dirs(out_dir)
    results = []

    for name, vcfg in variants:
        print(f"== Ablation: {name} ==")
        subprocess.run(["python", "scripts/21_build_dataset.py", "--config", vcfg], cwd=root, check=True)
        subprocess.run(["python", "scripts/40_train.py", "--config", vcfg], cwd=root, check=True)
        subprocess.run(["python", "scripts/50_evaluate.py", "--config", vcfg], cwd=root, check=True)
        v = load_yaml(root / vcfg)
        metrics_path = root / v["eval"]["output_metrics_path"]
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        results.append({"variant": name, "config": vcfg, "metrics": metrics})

    out_path = out_dir / "ablation_summary.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {out_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
