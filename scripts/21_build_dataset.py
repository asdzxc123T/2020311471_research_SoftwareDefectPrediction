from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sdp.config.loader import load_yaml
from sdp.data.io import write_table
from sdp.data.sample_data import make_sample_dataset
from sdp.data.splits import TemporalSplitConfig, apply_temporal_split
from sdp.utils.paths import ProjectPaths, ensure_dirs, find_repo_root


def build_from_sample(cfg: dict) -> pd.DataFrame:
    sample_cfg = cfg.get("sample", {}) or {}
    n_repos = int(sample_cfg.get("n_repos", 2))
    n_commits = int(sample_cfg.get("n_commits_per_repo", 80))
    seed = int(cfg.get("seed", 42))
    return make_sample_dataset(n_repos=n_repos, n_commits_per_repo=n_commits, seed=seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path (e.g., configs/exp/sample_end_to_end.yaml)")
    ap.add_argument("--seed", type=int, default=None, help="Override project seed")
    args = ap.parse_args()

    root = find_repo_root()
    paths = ProjectPaths(root=root)
    ensure_dirs(paths.data_dir, paths.artifacts_dir, paths.reports_dir)

    config = load_yaml(root / args.config)
    seed = int(args.seed if args.seed is not None else config.get("project", {}).get("seed", 42))
    data_cfg = config.get("data", {}) or {}

    source = str(data_cfg.get("source", "sample"))
    if source == "sample":
        df = build_from_sample({"sample": data_cfg.get("sample", {}), "seed": seed})
    elif source == "parquet":
        in_path = Path(data_cfg["input_path"])
        df = pd.read_parquet(root / in_path)
    else:
        raise ValueError(f"Unsupported data.source: {source}")

    ts_cfg_raw = data_cfg.get("temporal_split", {}) or {}
    ts_cfg = TemporalSplitConfig(
        train_ratio=float(ts_cfg_raw.get("train_ratio", 0.7)),
        val_ratio=float(ts_cfg_raw.get("val_ratio", 0.15)),
        test_ratio=float(ts_cfg_raw.get("test_ratio", 0.15)),
    )
    df = apply_temporal_split(df, cfg=ts_cfg)

    out_path = Path(data_cfg.get("output_path", "data/processed/dataset.parquet"))
    write_table(df, root / out_path)
    print(f"Wrote dataset: {out_path} (rows={len(df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

