from typing import Dict, List

import numpy as np
import pandas as pd


def make_windows(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_col: str,
    history_steps: int,
    horizon_steps: int,
    timestamp_col: str,
) -> Dict[str, np.ndarray]:
    """
    从单个 split 的数据中构造滑窗样本。

    说明：
    - 不允许跨 segment 取窗口
    - lag_label 使用窗口末端时刻的 lag_gt
    """
    x_list: List[np.ndarray] = []
    y_list: List[float] = []
    lag_label_list: List[int] = []
    end_timestamp_list: List[str] = []
    segment_id_list: List[int] = []

    for _, seg_df in df.groupby("segment_id", sort=True):
        seg_df = seg_df.reset_index(drop=True)
        n = len(seg_df)

        max_t = n - horizon_steps - 1
        start_t = history_steps - 1

        for t in range(start_t, max_t + 1):
            x_hist = seg_df.loc[t - history_steps + 1:t, feature_columns].to_numpy(dtype=np.float32)
            y_future = float(seg_df.loc[t + horizon_steps, target_col])

            x_list.append(x_hist)
            y_list.append(y_future)
            lag_label_list.append(int(seg_df.loc[t, "lag_gt"]))
            end_timestamp_list.append(str(seg_df.loc[t, timestamp_col]))
            segment_id_list.append(int(seg_df.loc[t, "segment_id"]))

    if not x_list:
        return {
            "X": np.empty((0, history_steps, len(feature_columns)), dtype=np.float32),
            "y": np.empty((0,), dtype=np.float32),
            "lag_label": np.empty((0,), dtype=np.int64),
            "timestamp_end": np.empty((0,), dtype=object),
            "segment_id": np.empty((0,), dtype=np.int64),
        }

    return {
        "X": np.stack(x_list, axis=0),
        "y": np.asarray(y_list, dtype=np.float32),
        "lag_label": np.asarray(lag_label_list, dtype=np.int64),
        "timestamp_end": np.asarray(end_timestamp_list),
        "segment_id": np.asarray(segment_id_list, dtype=np.int64),
    }
