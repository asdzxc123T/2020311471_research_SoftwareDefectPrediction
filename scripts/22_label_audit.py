from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from sdp.config.loader import load_yaml
from sdp.data.io import read_table
from sdp.utils.paths import ensure_dirs, find_repo_root
from sdp.utils.seed import set_global_seed


def _repo_slug(cfg: dict) -> str:
    pipeline = cfg.get("pipeline", {}) or {}
    repo_name = str(pipeline.get("repo_name") or cfg.get("data", {}).get("source", "dataset"))
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", repo_name).strip("_").lower()
    return slug or "dataset"


def _commit_url(repo_url: str, commit_hash: str) -> str:
    base = (repo_url or "").rstrip("/")
    if base and commit_hash:
        return f"{base}/commit/{commit_hash}"
    return ""


def _stratified_sample(
    df: pd.DataFrame,
    *,
    n: int,
    n_positive: int,
    label_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n_pos = min(n_positive, n, int((df[label_col].astype(int) == 1).sum()))
    n_neg = min(n - n_pos, int((df[label_col].astype(int) == 0).sum()))
    pos_df = df[df[label_col].astype(int) == 1]
    neg_df = df[df[label_col].astype(int) == 0]
    parts = []
    if n_pos > 0 and len(pos_df) > 0:
        parts.append(pos_df.sample(n=n_pos, random_state=int(rng.integers(0, 2**31 - 1))))
    if n_neg > 0 and len(neg_df) > 0:
        parts.append(neg_df.sample(n=n_neg, random_state=int(rng.integers(0, 2**31 - 1))))
    if not parts:
        idx = rng.choice(df.index.to_numpy(), size=min(n, len(df)), replace=False)
        return df.loc[idx].copy()
    combined = pd.concat(parts, ignore_index=False)
    return combined[~combined.index.duplicated(keep="first")]


def _git_verify_positive(repo_path: Path, row: pd.Series) -> int | None:
    """Return 1 if inducing commit is ancestor of a later BFC on same file, 0 if not, None if unknown."""
    if not repo_path.exists():
        return None
    commit = str(row.get("commit_hash") or "")
    file_path = str(row.get("file_path") or "").replace("\\", "/")
    if not commit or not file_path:
        return None
    proc = subprocess.run(
        ["git", "-C", str(repo_path), "log", "--format=%H", "--", file_path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        return None
    commits_on_file = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if commit not in commits_on_file:
        return 0
    idx = commits_on_file.index(commit)
    # Later commits on same file exist => possible fix chain
    if idx > 0:
        return 1
    return None


def _compute_audit_metrics(filled: pd.DataFrame, label_col: str) -> dict:
    pred = filled["is_bug_inducing"].astype(int).to_numpy()
    human = filled["human_verified_is_bug_inducing"].astype(int).to_numpy()
    tp = int(((pred == 1) & (human == 1)).sum())
    fp = int(((pred == 1) & (human == 0)).sum())
    fn = int(((pred == 0) & (human == 1)).sum())
    tn = int(((pred == 0) & (human == 0)).sum())
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "n_filled": int(len(filled)),
        "n_human_positive": int((human == 1).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--n", type=int, default=100, help="Total samples to audit")
    ap.add_argument("--n-positive", type=int, default=30, help="Stratified positive samples")
    ap.add_argument("--seed", type=int, default=None, help="Override seed")
    ap.add_argument("--regenerate", action="store_true", help="Force regenerate audit sheet")
    ap.add_argument(
        "--git-assist",
        action="store_true",
        help="Pre-fill human_verified for unlabeled rows using git file-history heuristic",
    )
    args = ap.parse_args()

    root = find_repo_root()
    cfg = load_yaml(root / args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    label_col = str(cfg.get("eval", {}).get("label_col", "is_bug_inducing"))
    ds_path = Path(
        cfg.get("eval", {}).get("dataset_path", cfg.get("data", {}).get("output_path", "data/processed/dataset.parquet"))
    )
    df = read_table(root / ds_path)
    pipeline = cfg.get("pipeline", {}) or {}
    repo_url = str(pipeline.get("repo_url") or "")
    repo_path = root / str(pipeline.get("repo_path", ""))

    out_dir = root / "reports" / "labeling"
    ensure_dirs(out_dir)
    slug = _repo_slug(cfg)
    audit_path = out_dir / f"label_audit_{slug}.csv"
    report_path = out_dir / f"label_precision_report_{slug}.json"

    if audit_path.exists() and not args.regenerate:
        sample = pd.read_csv(audit_path)
        if "human_verified_is_bug_inducing" not in sample.columns:
            sample["human_verified_is_bug_inducing"] = np.nan
        if "human_notes" not in sample.columns:
            sample["human_notes"] = ""
    else:
        rng = np.random.default_rng(seed)
        sample = _stratified_sample(
            df,
            n=min(args.n, len(df)),
            n_positive=args.n_positive,
            label_col=label_col,
            rng=rng,
        )
        sample = sample.copy()
        sample["human_verified_is_bug_inducing"] = np.nan
        sample["human_notes"] = ""
        if "message" not in sample.columns and "commit_message" in df.columns:
            sample["message"] = sample["commit_message"]
        if "label_source" not in sample.columns:
            sample["label_source"] = df.loc[sample.index, "label_source"] if "label_source" in df.columns else ""
        sample["commit_url"] = [
            _commit_url(repo_url, str(h)) for h in sample.get("commit_hash", pd.Series(dtype=str)).astype(str)
        ]
        sample.to_csv(audit_path, index=False)

    if args.git_assist:
        updated = False
        for i, row in sample.iterrows():
            if pd.notna(row.get("human_verified_is_bug_inducing")):
                continue
            if int(row.get(label_col, 0)) == 0:
                sample.at[i, "human_verified_is_bug_inducing"] = 0
                sample.at[i, "human_notes"] = "git_assist_negative_default"
                updated = True
            else:
                verdict = _git_verify_positive(repo_path, row)
                if verdict is not None:
                    sample.at[i, "human_verified_is_bug_inducing"] = verdict
                    sample.at[i, "human_notes"] = "git_assist_file_history"
                    updated = True
        if updated:
            sample.to_csv(audit_path, index=False)

    filled = sample.dropna(subset=["human_verified_is_bug_inducing"])
    metrics = _compute_audit_metrics(filled, label_col) if len(filled) > 0 and label_col in filled.columns else {}

    report = {
        "dataset_path": str(ds_path),
        "repo_slug": slug,
        "seed": seed,
        "n_sampled": int(len(sample)),
        "n_positive_target": args.n_positive,
        "audit_sheet": str(audit_path.relative_to(root)),
        "precision_if_filled": metrics.get("precision"),
        "recall_if_filled": metrics.get("recall"),
        "audit_counts": metrics,
        "note": "Fill human_verified_is_bug_inducing in audit_sheet then re-run for precision/recall.",
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # Also write aggregate report for backward compatibility
    (out_dir / "label_precision_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
