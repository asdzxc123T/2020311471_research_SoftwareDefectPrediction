from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sdp.eval.stats_tests import bootstrap_ci
from sdp.utils.paths import ensure_dirs, find_repo_root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", required=True, help="Comma-separated numeric values for bootstrap CI")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="reports/stats/bootstrap_ci.json")
    args = ap.parse_args()

    root = find_repo_root()
    ensure_dirs(root / "reports" / "stats")

    vals = np.array([float(x.strip()) for x in args.values.split(",") if x.strip()], dtype=float)
    ci = bootstrap_ci(vals, n_boot=args.n_boot, alpha=args.alpha)
    out = {"mean": ci.mean, "lower": ci.lower, "upper": ci.upper, "n": int(vals.size)}

    out_path = root / Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

