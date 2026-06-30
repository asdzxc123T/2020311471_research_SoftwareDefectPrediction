from __future__ import annotations

import argparse
import subprocess
import sys

from sdp.utils.paths import find_repo_root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp/codebert_only.yaml")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    root = find_repo_root()
    cmd = ["python", "scripts/21_build_dataset.py", "--config", args.config]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    subprocess.run(cmd, cwd=root, check=True)
    cmd = ["python", "scripts/40_train.py", "--config", args.config]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    subprocess.run(cmd, cwd=root, check=True)
    subprocess.run(["python", "scripts/50_evaluate.py", "--config", args.config], cwd=root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
