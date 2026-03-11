from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class BaselineTrainConfig:
    task: str
    numeric_features: list[str]
    text_feature: str | None = None
    label_col: str = "is_bug_inducing"
    seed: int = 42


def train_baseline_model(df: pd.DataFrame, cfg: BaselineTrainConfig) -> Pipeline:
    if cfg.label_col not in df.columns:
        raise KeyError(f"Missing label column: {cfg.label_col}")
    X = df.copy()
    y = X.pop(cfg.label_col).astype(int)

    numeric = cfg.numeric_features
    for c in numeric:
        if c not in X.columns:
            raise KeyError(f"Missing numeric feature: {c}")

    transformers: list[tuple[str, object, list[str]]] = []
    transformers.append(("num", Pipeline([("scaler", StandardScaler())]), numeric))

    if cfg.text_feature is not None:
        # Keep a simple categorical encoder for text placeholder; real embedding models come later.
        # This remains fully runnable without heavy NLP deps.
        if cfg.text_feature not in X.columns:
            raise KeyError(f"Missing text feature: {cfg.text_feature}")
        transformers.append(("txt", OneHotEncoder(handle_unknown="ignore"), [cfg.text_feature]))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    if cfg.task == "baseline_rf":
        model = RandomForestClassifier(n_estimators=300, random_state=cfg.seed, n_jobs=-1)
    elif cfg.task == "baseline_lr":
        model = LogisticRegression(max_iter=500, random_state=cfg.seed)
    else:
        raise ValueError(f"Unknown baseline task: {cfg.task}")

    pipe = Pipeline([("pre", pre), ("clf", model)])
    pipe.fit(X, y)
    return pipe


def save_model(model: Pipeline, path: str) -> None:
    dump(model, path)
