from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from sdp.config.loader import load_yaml
from sdp.eval.stats_tests import bootstrap_ci
from sdp.utils.paths import ensure_dirs, find_repo_root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_a", required=True, help="Path to metrics JSON A")
    ap.add_argument("--metrics_b", required=True, help="Path to metrics JSON B")
    ap.add_argument("--out", default="reports/stats/model_comparison.json")
    args = ap.parse_args()

    root = find_repo_root()
    a = json.loads((root / args.metrics_a).read_text(encoding="utf-8"))
    b = json.loads((root / args.metrics_b).read_text(encoding="utf-8"))

    out = {
        "metrics_a": a,
        "metrics_b": b,
        "delta_auc": (b.get("auc") or 0) - (a.get("auc") or 0),
        "delta_f1": (b.get("f1") or 0) - (a.get("f1") or 0),
        "delta_recall_at_topk_loc": (b.get("recall_at_topk_loc") or 0) - (a.get("recall_at_topk_loc") or 0),
    }

    # If multiseed summaries exist, compute CI on f1 series
    ms_dir = root / "reports" / "multiseed"
    if ms_dir.exists():
        f1_vals = []
        for p in ms_dir.glob("**/run_summary.json"):
            data = json.loads(p.read_text(encoding="utf-8"))
            for run in data.get("runs", []):
                if "metrics" in run and run["metrics"].get("f1") is not None:
                    f1_vals.append(float(run["metrics"]["f1"]))
        if len(f1_vals) >= 3:
            ci = bootstrap_ci(np.array(f1_vals))
            out["bootstrap_ci_f1"] = {"mean": ci.mean, "lower": ci.lower, "upper": ci.upper}

    out_dir = root / Path(args.out).parent
    ensure_dirs(out_dir)
    out_path = root / args.out
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
