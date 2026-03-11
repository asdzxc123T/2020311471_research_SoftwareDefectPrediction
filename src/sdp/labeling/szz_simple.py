from __future__ import annotations

import pandas as pd


def label_bug_inducing_commits_simple(df_commits: pd.DataFrame) -> pd.DataFrame:
    """
    실행 가능한 최소 SZZ 대체 구현(뼈대용).
    - BFC(버그 수정 커밋) 직전 커밋을 BIC(버그 유발)로 표시한다.
    - 실제 SZZ(라인 blame 기반) 고도화는 `src/sdp/labeling/`에 확장한다.
    """
    req = {"repo_name", "commit_hash", "commit_time", "is_bug_fixing_commit"}
    missing = req - set(df_commits.columns)
    if missing:
        raise KeyError(f"Missing columns for simple SZZ: {sorted(missing)}")

    out = df_commits.copy()
    out["is_bug_inducing"] = False
    out = out.sort_values(["repo_name", "commit_time", "commit_hash"], kind="mergesort")

    for repo, g in out.groupby("repo_name", sort=False):
        idx = g.index.to_list()
        for i in range(1, len(idx)):
            cur = idx[i]
            prev = idx[i - 1]
            if bool(out.loc[cur, "is_bug_fixing_commit"]):
                out.loc[prev, "is_bug_inducing"] = True
    out["label_source"] = out.get("label_source", "szz_simple")
    out.loc[out["label_source"].isna(), "label_source"] = "szz_simple"
    return out
