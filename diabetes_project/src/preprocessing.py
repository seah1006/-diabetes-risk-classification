"""Reusable preprocessing helpers for the diabetes project."""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin


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


class ZScoreClipper(BaseEstimator, TransformerMixin):
    def __init__(self, columns, threshold: float = 3.0):
        self.columns = columns
        self.threshold = threshold

    def fit(self, X, y=None):
        X_df = self._to_dataframe(X)
        self.feature_names_in_ = list(X_df.columns)
        self.means_ = X_df[self.columns].mean()
        self.stds_ = X_df[self.columns].std(ddof=0).replace(0, np.nan)
        self.lower_bounds_ = self.means_ - self.threshold * self.stds_
        self.upper_bounds_ = self.means_ + self.threshold * self.stds_
        return self

    def transform(self, X):
        X_df = self._to_dataframe(X).copy()
        for col in self.columns:
            X_df[col] = X_df[col].clip(self.lower_bounds_[col], self.upper_bounds_[col])
        return X_df

    def _to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=getattr(self, "feature_names_in_", None))
