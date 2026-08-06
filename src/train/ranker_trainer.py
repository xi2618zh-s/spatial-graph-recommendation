"""M7 ranker training on M6's sampled rows (data/processed/ranking_samples.csv).

Two model types:
  lr    LogisticRegression on median-imputed, standardized features --
        checked for coefficient sign/direction, not raw performance.
  gbdt  sklearn HistGradientBoostingClassifier -- the traditional
        non-linear-interaction baseline. Used instead of LightGBM to avoid
        adding a second large binary dependency on top of torch/faiss; it
        natively handles missing values (no imputation needed), which the
        M6 feature set relies on (see docs/02_samples_features.md).

Feature-group ablation (`feature_columns(df, group)`): progressively adds
recall score -> + user/item statistics -> + spatial distance -> full
feature set (context + candidate metadata), matching the M7 acceptance bar
in PROJECT_HANDOFF_V2.md.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

IDENTIFIER_COLS = {"user_id", "item_id", "label", "negative_source", "split", "query_ts"}


def _is_spatial(col: str) -> bool:
    return col in {"cross_dist_to_center_km", "cross_dist_missing"}


def _is_stat(col: str) -> bool:
    return col.startswith("user_") or col.startswith("item_")


FEATURE_GROUPS = {
    "recall_only": lambda c: c == "cross_recall_score",
    "stats": lambda c: c == "cross_recall_score" or _is_stat(c),
    "spatial": lambda c: c == "cross_recall_score" or _is_stat(c) or _is_spatial(c),
    "full": lambda c: True,
}


def feature_columns(df: pd.DataFrame, group: str = "full") -> list[str]:
    numeric = [c for c in df.columns if c not in IDENTIFIER_COLS and df[c].dtype != object]
    pred = FEATURE_GROUPS[group]
    return [c for c in numeric if pred(c)]


def _xy(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].astype(float).to_numpy()
    y = df["label"].to_numpy()
    return X, y


def train_lr(df: pd.DataFrame, feature_cols: list[str], seed: int = 2020,
            **kwargs) -> Pipeline:
    X, y = _xy(df, feature_cols)
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(random_state=seed, **kwargs)),
    ])
    model.fit(X, y)
    return model


def train_gbdt(df: pd.DataFrame, feature_cols: list[str], seed: int = 2020,
               **kwargs) -> HistGradientBoostingClassifier:
    X, y = _xy(df, feature_cols)
    model = HistGradientBoostingClassifier(random_state=seed, **kwargs)
    model.fit(X, y)
    return model


def score(model, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    X = df[feature_cols].astype(float).to_numpy()
    return model.predict_proba(X)[:, 1]
