from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind


def chi2_test(df: pd.DataFrame, binary_features: list, target: str) -> pd.DataFrame:
    rows = []
    for col in binary_features:
        ct = pd.crosstab(df[col], df[target])
        chi2, p, dof, _ = chi2_contingency(ct)
        rows.append({
            "Feature": col,
            "Chi2": round(chi2, 3),
            "p-value": round(p, 6),
            "Significant": p < 0.05,
        })
    return pd.DataFrame(rows).sort_values("Chi2", ascending=False).reset_index(drop=True)


def ttest_summary(df: pd.DataFrame, continuous_features: list, target: str) -> pd.DataFrame:
    diabetic = df[df[target] == 1]
    normal = df[df[target] == 0]
    rows = []
    for col in continuous_features:
        t_stat, p = ttest_ind(diabetic[col], normal[col])
        rows.append({
            "Feature": col,
            "t-stat": round(t_stat, 3),
            "p-value": round(p, 6),
            "Mean (Diabetic)": round(diabetic[col].mean(), 3),
            "Mean (Normal)": round(normal[col].mean(), 3),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("t-stat", key=abs, ascending=False)
        .reset_index(drop=True)
    )


def odds_ratio_df(model, feature_names: list) -> pd.DataFrame:
    return (
        pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": model.coef_[0],
            "Odds Ratio": np.exp(model.coef_[0]),
        })
        .sort_values("Odds Ratio", ascending=False)
        .reset_index(drop=True)
    )
