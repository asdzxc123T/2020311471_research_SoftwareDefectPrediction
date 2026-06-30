from __future__ import annotations

import argparse
from pathlib import Path

from sdp.config.loader import load_yaml
from sdp.data.defects4j import build_defects4j_sample_dataset, load_defects4j_if_available
from sdp.data.io import write_table
from sdp.data.splits import TemporalSplitConfig, apply_temporal_split
from sdp.utils.paths import ensure_dirs, find_repo_root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp/defects4j_baseline.yaml")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    root = find_repo_root()
    cfg = load_yaml(root / args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("project", {}).get("seed", 42))
    data_cfg = cfg.get("data", {}) or {}
    ensure_dirs(root / "data" / "processed")

    d4j_root = data_cfg.get("defects4j_root")
    projects = list(data_cfg.get("projects", ["Lang", "Chart", "Math", "Time"]))
    df = load_defects4j_if_available(d4j_root, projects=projects)
    if df is None:
        sample_cfg = data_cfg.get("sample", {}) or {}
        df = build_defects4j_sample_dataset(
            seed=seed,
            n_projects=int(sample_cfg.get("n_projects", 4)),
            n_methods_per_project=int(sample_cfg.get("n_methods_per_project", 120)),
        )

    ts_cfg_raw = data_cfg.get("temporal_split", {}) or {}
    ts_cfg = TemporalSplitConfig(
        train_ratio=float(ts_cfg_raw.get("train_ratio", 0.7)),
        val_ratio=float(ts_cfg_raw.get("val_ratio", 0.15)),
        test_ratio=float(ts_cfg_raw.get("test_ratio", 0.15)),
    )
    df = apply_temporal_split(df, cfg=ts_cfg)

    out_path = Path(data_cfg.get("output_path", "data/processed/defects4j_dataset.parquet"))
    write_table(df, root / out_path)
    print(f"Wrote Defects4J dataset: {out_path} (rows={len(df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
