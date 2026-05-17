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
            "피처": col,
            "카이제곱": round(chi2, 3),
            "p값": round(p, 6),
            "자유도": dof,
            "유의함": p < 0.05,
        })
    return pd.DataFrame(rows).sort_values("카이제곱", ascending=False).reset_index(drop=True)


def ttest_summary(df: pd.DataFrame, continuous_features: list, target: str) -> pd.DataFrame:
    diabetic = df[df[target] == 1]
    normal = df[df[target] == 0]
    rows = []
    for col in continuous_features:
        t_stat, p = ttest_ind(diabetic[col], normal[col], equal_var=False)
        rows.append({
            "피처": col,
            "t통계량": round(t_stat, 3),
            "절대 t통계량": round(abs(t_stat), 3),
            "p값": round(p, 6),
            "당뇨군 평균": round(diabetic[col].mean(), 3),
            "정상군 평균": round(normal[col].mean(), 3),
            "평균 차이": round(diabetic[col].mean() - normal[col].mean(), 3),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("절대 t통계량", ascending=False)
        .reset_index(drop=True)
    )


def odds_ratio_df(model, feature_names: list) -> pd.DataFrame:
    coef_df = pd.DataFrame({
        "피처": feature_names,
        "계수": model.coef_[0],
        "오즈비": np.exp(model.coef_[0]),
    })
    return coef_df.sort_values("오즈비", ascending=False).reset_index(drop=True)
