from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from sdp.config.loader import load_yaml
from sdp.utils.paths import ensure_dirs, find_repo_root


def main() -> int:
    ap = argparse.ArgumentParser(description="Run GitHub mining -> BFC -> SZZ -> dataset build pipeline.")
    ap.add_argument("--config", default="configs/exp/github_szz.yaml")
    ap.add_argument("--repo", default=None, help="Override local repo path")
    ap.add_argument("--skip-mine", action="store_true")
    args = ap.parse_args()

    root = find_repo_root()
    cfg = load_yaml(root / args.config)
    pipe = cfg.get("pipeline", {}) or {}
    repo_path = Path(args.repo or pipe.get("repo_path", "data/raw/repos/sample_repo"))
    repo_name = str(pipe.get("repo_name", repo_path.name))
    repo_url = str(pipe.get("repo_url", ""))
    max_commits = pipe.get("max_commits")
    szz_method = str(pipe.get("szz_method", "szz_v1"))

    ensure_dirs(root / "data" / "raw", root / "data" / "processed")
    for key in ("raw_records", "bfc_records", "labeled_records"):
        p = pipe.get(key)
        if p:
            ensure_dirs(root / Path(p).parent)

    if not args.skip_mine:
        cmd = [
            "python",
            "scripts/11_mine_github.py",
            "--repo",
            str(repo_path),
            "--repo_name",
            repo_name,
            "--repo_url",
            repo_url,
            "--out",
            str(pipe.get("raw_records", "data/raw/function_records.parquet")),
        ]
        if max_commits:
            cmd.extend(["--max_commits", str(max_commits)])
        subprocess.run(cmd, cwd=root, check=True)

    subprocess.run(
        [
            "python",
            "scripts/12_detect_bfc.py",
            "--in_path",
            str(pipe.get("raw_records", "data/raw/function_records.parquet")),
            "--out_path",
            str(pipe.get("bfc_records", "data/processed/github_with_bfc.parquet")),
        ],
        cwd=root,
        check=True,
    )

    szz_cmd = [
        "python",
        "scripts/20_label_szz.py",
        "--in_path",
        str(pipe.get("bfc_records", "data/processed/github_with_bfc.parquet")),
        "--out_path",
        str(pipe.get("labeled_records", "data/processed/github_labeled.parquet")),
        "--method",
        szz_method,
    ]
    if szz_method == "szz_v1":
        szz_cmd.extend(["--repo", str(repo_path)])
    subprocess.run(szz_cmd, cwd=root, check=True)

    subprocess.run(
        ["python", "scripts/21_build_dataset.py", "--config", args.config],
        cwd=root,
        check=True,
    )
    print("GitHub pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
