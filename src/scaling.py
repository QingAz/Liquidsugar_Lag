from typing import Dict, List

import pandas as pd


def fit_zscore_stats(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, float]]:
    """在 train 注入后数据上拟合 z-score 统计量。"""
    stats: Dict[str, Dict[str, float]] = {}
    for col in columns:
        mean = float(df[col].mean())
        std = float(df[col].std(ddof=0))
        if std == 0.0:
            std = 1.0
        stats[col] = {
            "mean": mean,
            "std": std,
        }
    return stats


def apply_zscore(
    df: pd.DataFrame,
    stats: Dict[str, Dict[str, float]],
    columns: List[str],
) -> pd.DataFrame:
    """根据 train 统计量对各 split 应用标准化。"""
    out = df.copy()
    for col in columns:
        mean = stats[col]["mean"]
        std = stats[col]["std"]
        out[col] = (out[col] - mean) / std
    return out
