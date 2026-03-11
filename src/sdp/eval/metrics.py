from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    auc: float | None
    ap: float | None
    precision: float
    recall: float
    f1: float


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> ClassificationMetrics:
    y_pred = (y_prob >= threshold).astype(int)
    auc = None
    ap = None
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = None
    try:
        ap = float(average_precision_score(y_true, y_prob))
    except Exception:
        ap = None
    return ClassificationMetrics(
        auc=auc,
        ap=ap,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )


def recall_at_topk_loc(df: pd.DataFrame, *, y_col: str, prob_col: str, effort_col: str, topk_loc: float) -> float:
    if not (0 < topk_loc <= 1):
        raise ValueError("topk_loc must be in (0, 1]")
    d = df[[y_col, prob_col, effort_col]].copy()
    d[effort_col] = pd.to_numeric(d[effort_col], errors="coerce").fillna(0).clip(lower=0)
    d = d.sort_values(prob_col, ascending=False, kind="mergesort")
    budget = float(d[effort_col].sum()) * float(topk_loc)
    spent = 0.0
    selected = []
    for i, row in d.iterrows():
        if spent >= budget:
            break
        selected.append(i)
        spent += float(row[effort_col])
    if len(selected) == 0:
        return 0.0
    total_pos = float(d[y_col].sum())
    if total_pos <= 0:
        return 0.0
    found_pos = float(d.loc[selected, y_col].sum())
    return float(found_pos / total_pos)


def popt_like(df: pd.DataFrame, *, y_col: str, prob_col: str, effort_col: str) -> float:
    """
    Popt 계열 지표는 구현/정의 변형이 많아, 여기서는 "effort-aware area under curve"의 단순 변형을 제공한다.
    - effort 축: 누적 effort 비율
    - y 축: 누적 발견 결함 비율
    - 모델 순위 vs 이상적(결함 먼저) vs 무작위(기대값) 비교 대신, 0~1로 정규화된 면적 자체를 반환
    """
    d = df[[y_col, prob_col, effort_col]].copy()
    d[effort_col] = pd.to_numeric(d[effort_col], errors="coerce").fillna(0).clip(lower=0)
    total_eff = float(d[effort_col].sum())
    total_pos = float(d[y_col].sum())
    if total_eff <= 0 or total_pos <= 0:
        return 0.0

    d = d.sort_values(prob_col, ascending=False, kind="mergesort")
    cum_eff = np.cumsum(d[effort_col].to_numpy(dtype=float)) / total_eff
    cum_pos = np.cumsum(d[y_col].to_numpy(dtype=float)) / total_pos
    # trapezoid with origin
    x = np.concatenate([[0.0], cum_eff])
    y = np.concatenate([[0.0], cum_pos])
    area = float(np.trapezoid(y, x))
    return max(0.0, min(1.0, area))
