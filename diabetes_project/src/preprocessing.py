"""Reusable preprocessing helpers for the diabetes project."""

from __future__ import annotations

import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy import stats


def load_data(path: str) -> pd.DataFrame:
    """Load CSV and return DataFrame."""
    return pd.read_csv(path)


def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    """Return boolean mask of IQR outliers."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Return boolean mask of Z-score outliers."""
    z_scores = pd.Series(
        abs(stats.zscore(series, nan_policy="omit")),
        index=series.index,
    )
    return z_scores > threshold


def build_pipeline(imputer, scaler, sampler=None):
    """Return assembled imblearn Pipeline."""
    steps = [
        ("imputer", imputer),
        ("scaler", scaler),
    ]
    if sampler is not None:
        steps.append(("sampler", sampler))
    return ImbPipeline(steps)
