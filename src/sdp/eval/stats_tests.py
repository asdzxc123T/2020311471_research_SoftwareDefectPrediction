from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    lower: float
    upper: float


def bootstrap_ci(values: np.ndarray, *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> BootstrapCI:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return BootstrapCI(mean=float("nan"), lower=float("nan"), upper=float("nan"))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        s = rng.choice(v, size=v.size, replace=True)
        boots.append(float(np.mean(s)))
    boots = np.array(boots, dtype=float)
    lower = float(np.quantile(boots, alpha / 2))
    upper = float(np.quantile(boots, 1 - alpha / 2))
    return BootstrapCI(mean=float(np.mean(v)), lower=lower, upper=upper)


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    stat = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", mode="auto")
    return float(stat.pvalue)


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    pred_a = np.asarray(pred_a, dtype=int)
    pred_b = np.asarray(pred_b, dtype=int)
    if not (y_true.shape == pred_a.shape == pred_b.shape):
        raise ValueError("Shapes must match")
    correct_a = (pred_a == y_true).astype(int)
    correct_b = (pred_b == y_true).astype(int)
    # Contingency table:
    #          B correct   B wrong
    # A correct    n11        n10
    # A wrong      n01        n00
    n11 = int(((correct_a == 1) & (correct_b == 1)).sum())
    n10 = int(((correct_a == 1) & (correct_b == 0)).sum())
    n01 = int(((correct_a == 0) & (correct_b == 1)).sum())
    n00 = int(((correct_a == 0) & (correct_b == 0)).sum())
    table = [[n11, n10], [n01, n00]]
    res = mcnemar(table, exact=False, correction=True)
    return float(res.pvalue)
