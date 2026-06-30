from __future__ import annotations

import argparse
import json
from pathlib import Path
from subprocess import run

from sdp.config.loader import load_yaml
from sdp.utils.paths import ensure_dirs, find_repo_root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--seeds", default="1,2,3,4,5", help="Comma-separated seeds")
    args = ap.parse_args()

    root = find_repo_root()
    cfg = load_yaml(root / args.config)
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]

    out_dir = root / "reports" / "multiseed"
    ensure_dirs(out_dir)

    eval_cfg = cfg.get("eval", {}) or {}
    metrics_template = Path(str(eval_cfg.get("output_metrics_path", "reports/metrics/metrics.json")))

    results = []
    for seed in seeds:
        print(f"== Seed {seed} ==")
        run(["python", "scripts/21_build_dataset.py", "--config", args.config, "--seed", str(seed)], cwd=root, check=True)
        run(["python", "scripts/40_train.py", "--config", args.config, "--seed", str(seed)], cwd=root, check=True)
        run(["python", "scripts/50_evaluate.py", "--config", args.config], cwd=root, check=True)

        metrics_path = root / metrics_template
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        results.append({"seed": seed, "status": "ok", "metrics": metrics})

    summary_path = out_dir / "run_summary.json"
    summary_path.write_text(json.dumps({"seeds": seeds, "runs": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {summary_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

