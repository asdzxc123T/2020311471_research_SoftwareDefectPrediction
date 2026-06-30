from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from sdp.config.loader import load_yaml
from sdp.utils.paths import ensure_dirs, find_repo_root


def _git_head(root: Path) -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp/github_multi.yaml")
    args = ap.parse_args()

    root = find_repo_root()
    cfg = load_yaml(root / args.config)
    out_dir = root / "reports"
    ensure_dirs(out_dir)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(root),
        "config": args.config,
        "seed": cfg.get("project", {}).get("seed", 42),
        "dataset_path": str(cfg.get("data", {}).get("output_path", "")),
        "artifacts": {
            "github_baseline": "artifacts/models/github_baseline.joblib",
            "defects4j_baseline": "artifacts/models/defects4j_baseline.joblib",
            "gnn_github": "artifacts/models/gnn_github.pt",
            "codebert_github": "artifacts/models/codebert_github.pt",
            "multimodal_github": "artifacts/models/multimodal_github.pt",
        },
        "reports": {
            "dataset_validation": "reports/data/dataset_report.json",
            "label_precision": "reports/labeling/label_precision_report.json",
            "github_baseline_metrics": "reports/metrics/github_baseline_metrics.json",
            "multiseed": "reports/multiseed/run_summary.json",
            "robustness": "reports/robustness/github_label_flipping.json",
            "ablation": "reports/ablation/ablation_summary.json",
        },
    }

    out_path = out_dir / "experiment_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {out_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
