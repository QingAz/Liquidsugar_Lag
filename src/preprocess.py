from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class ColumnGroups:
    """按照工段组织后的列分组。"""
    feed: List[str]
    stage1: List[str]
    stage2: List[str]
    stage3: List[str]
    target: str

    @property
    def feature_columns(self) -> List[str]:
        """模型输入特征列的统一顺序。"""
        return self.feed + self.stage1 + self.stage2 + self.stage3


def load_dataframe(input_csv: str, timestamp_col: str) -> pd.DataFrame:
    """读取原始 CSV，并将时间列解析为 pandas 时间类型。"""
    df = pd.read_csv(input_csv)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    return df


def infer_column_groups(df: pd.DataFrame, config: Dict[str, Any]) -> ColumnGroups:
    """基于列名前缀自动识别 feed / stage1 / stage2 / stage3。"""
    prefix = config["feature_prefix"]
    target_col = config["target_col"]

    feed_cols = [c for c in df.columns if c.startswith(prefix["feed"])]
    stage1_cols = [c for c in df.columns if c.startswith(prefix["stage1"])]
    stage2_cols = [c for c in df.columns if c.startswith(prefix["stage2"])]
    stage3_cols = [c for c in df.columns if c.startswith(prefix["stage3"])]

    return ColumnGroups(
        feed=sorted(feed_cols),
        stage1=sorted(stage1_cols),
        stage2=sorted(stage2_cols),
        stage3=sorted(stage3_cols),
        target=target_col,
    )


def add_segments(
    df: pd.DataFrame,
    timestamp_col: str,
    g_break_minutes: int,
) -> pd.DataFrame:
    """
    依据原始时间 gap 进行分段。

    若相邻两条记录时间差 > g_break_minutes，则进入新 segment。
    """
    out = df.copy()
    dt_minutes = out[timestamp_col].diff().dt.total_seconds().div(60)
    is_new_segment = dt_minutes.gt(g_break_minutes).fillna(False)
    out["segment_id"] = is_new_segment.cumsum().astype(int)
    return out


def _validate_nominal_time_alignment(
    timestamps: pd.Series,
    nominal_step_minutes: int,
) -> None:
    """
    确保时间戳位于名义时间网格上。

    例如 nominal_step_minutes=15 时，要求分钟数能被 15 整除，且秒/微秒为 0。
    """
    minutes_ok = timestamps.dt.minute.mod(nominal_step_minutes).eq(0)
    seconds_ok = timestamps.dt.second.eq(0)
    micros_ok = timestamps.dt.microsecond.eq(0)

    if not (minutes_ok & seconds_ok & micros_ok).all():
        bad_rows = timestamps.loc[~(minutes_ok & seconds_ok & micros_ok)]
        example = bad_rows.iloc[0]
        raise ValueError(
            "检测到未对齐名义时间网格的时间戳，无法按固定 15 分钟 step 生成 lag。"
            f" 示例时间戳：{example}"
        )


def regularize_to_nominal_grid(
    df: pd.DataFrame,
    timestamp_col: str,
    nominal_step_minutes: int,
) -> pd.DataFrame:
    """
    在每个原始 segment 内重建名义 15 分钟时间网格，并对数值列做时间插值。

    这样可确保：
    - 1 step 的物理含义固定为 nominal_step_minutes
    - 后续 2/4/6 step 能稳定对应 30/60/90 min
    - 不跨越 > g_break_minutes 的原始 segment
    """
    out_segments: List[pd.DataFrame] = []
    freq = pd.Timedelta(minutes=nominal_step_minutes)

    for segment_id, seg_df in df.groupby("segment_id", sort=True):
        seg_df = seg_df.sort_values(timestamp_col).drop_duplicates(timestamp_col, keep="last").copy()
        _validate_nominal_time_alignment(seg_df[timestamp_col], nominal_step_minutes)

        original_index = pd.DatetimeIndex(seg_df[timestamp_col])
        full_index = pd.date_range(
            start=original_index.min(),
            end=original_index.max(),
            freq=freq,
        )

        value_columns = [c for c in seg_df.columns if c not in {timestamp_col, "segment_id"}]
        reindexed = seg_df.set_index(timestamp_col)[value_columns].reindex(full_index)
        reindexed = reindexed.astype(float).interpolate(method="time", limit_direction="both")

        reindexed = reindexed.reset_index().rename(columns={"index": timestamp_col})
        reindexed["segment_id"] = int(segment_id)
        reindexed["is_interpolated"] = ~reindexed[timestamp_col].isin(original_index)
        out_segments.append(reindexed)

    out = pd.concat(out_segments, axis=0, ignore_index=True)
    out = out.sort_values(timestamp_col).reset_index(drop=True)
    return out


def _normalize_split_ratios(split_config: Dict[str, float]) -> Tuple[float, float, float]:
    """将 train / val / test 比例归一化到和为 1。"""
    train_ratio = float(split_config["train_ratio"])
    val_ratio = float(split_config["val_ratio"])
    test_ratio = float(split_config["test_ratio"])

    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")
    return train_ratio / total, val_ratio / total, test_ratio / total


def _segment_allocation_unit(
    seg_len: int,
    allocation_unit: str,
    history_steps: Optional[int],
    horizon_steps: Optional[int],
) -> int:
    """计算单个 segment 在切分时使用的预算单位。"""
    seg_len = int(seg_len)

    if allocation_unit == "rows":
        return seg_len

    if allocation_unit == "effective_windows":
        if history_steps is None or horizon_steps is None:
            raise ValueError(
                "allocation_unit='effective_windows' requires history_steps and horizon_steps"
            )
        return max(seg_len - int(history_steps) - int(horizon_steps) + 1, 0)

    raise ValueError(f"Unknown split allocation_unit={allocation_unit}")


def _time_sorted_segment_summary(
    df: pd.DataFrame,
    timestamp_col: str,
    split_config: Dict[str, float],
    history_steps: Optional[int],
    horizon_steps: Optional[int],
) -> pd.DataFrame:
    """按 segment 起始时间排序，并计算每段用于切分的预算单位。"""
    if "segment_id" not in df.columns:
        raise ValueError("segment_id column is required for time-sorted segment split")

    allocation_unit = str(split_config.get("allocation_unit", "effective_windows")).lower()
    summary = (
        df.groupby("segment_id", sort=True)
        .agg(
            seg_start_time=(timestamp_col, "min"),
            seg_end_time=(timestamp_col, "max"),
            seg_len=("segment_id", "size"),
        )
        .reset_index()
        .sort_values(["seg_start_time", "segment_id"])
        .reset_index(drop=True)
    )
    summary["allocation_units"] = summary["seg_len"].apply(
        lambda seg_len: _segment_allocation_unit(
            seg_len=seg_len,
            allocation_unit=allocation_unit,
            history_steps=history_steps,
            horizon_steps=horizon_steps,
        )
    ).astype(int)
    return summary


def _choose_time_sorted_segment_boundaries(
    segment_summary: pd.DataFrame,
    split_config: Dict[str, float],
) -> Tuple[int, int]:
    """
    在时间有序的 segment 序列上寻找两个边界，使各 split 的累计预算单位尽量逼近目标比例。

    返回：
    - train_end_idx: train 最后一个 segment 在 summary 中的下标
    - val_end_idx: val 最后一个 segment 在 summary 中的下标
    """
    n_segments = int(len(segment_summary))
    if n_segments < 3:
        raise ValueError("At least 3 segments are required for train/val/test chronological split")

    weights = segment_summary["allocation_units"].astype(float).tolist()
    total_units = float(sum(weights))
    if total_units <= 0:
        raise ValueError(
            "All segments have zero allocation units; cannot build a split from effective windows"
        )

    train_ratio, val_ratio, test_ratio = _normalize_split_ratios(split_config)
    min_segments_cfg = split_config.get("min_segments", {})
    min_train_segments = int(min_segments_cfg.get("train", 1))
    min_val_segments = int(min_segments_cfg.get("val", 1))
    min_test_segments = int(min_segments_cfg.get("test", 1))
    if min(min_train_segments, min_val_segments, min_test_segments) <= 0:
        raise ValueError("min_segments for train/val/test must all be positive")
    if min_train_segments + min_val_segments + min_test_segments > n_segments:
        raise ValueError(
            "Requested min_segments exceed available segment count for chronological split"
        )
    best_candidate = None
    best_deviation = None

    prefix = [0.0]
    for weight in weights:
        prefix.append(prefix[-1] + weight)

    for train_end_idx in range(0, n_segments - 2):
        train_segment_count = train_end_idx + 1
        train_units = prefix[train_end_idx + 1]
        for val_end_idx in range(train_end_idx + 1, n_segments - 1):
            val_segment_count = val_end_idx - train_end_idx
            test_segment_count = n_segments - val_end_idx - 1
            if (
                train_segment_count < min_train_segments
                or val_segment_count < min_val_segments
                or test_segment_count < min_test_segments
            ):
                continue

            val_units = prefix[val_end_idx + 1] - prefix[train_end_idx + 1]
            test_units = total_units - prefix[val_end_idx + 1]

            if min(train_units, val_units, test_units) <= 0:
                continue

            train_actual = train_units / total_units
            val_actual = val_units / total_units
            test_actual = test_units / total_units
            candidate = (
                abs(train_actual - train_ratio)
                + abs(val_actual - val_ratio)
                + abs(test_actual - test_ratio),
                abs(train_actual - train_ratio),
                abs(val_actual - val_ratio),
                abs(test_actual - test_ratio),
                train_end_idx,
                val_end_idx,
            )
            if best_candidate is None or candidate < best_candidate:
                best_candidate = candidate
                best_deviation = candidate[0]

    if best_candidate is None:
        raise ValueError(
            "Unable to find a chronological segment split with positive allocation units in every split"
        )

    max_ratio_deviation = split_config.get("max_ratio_deviation")
    if max_ratio_deviation is not None:
        threshold = float(max_ratio_deviation)
        if best_deviation is not None and best_deviation > threshold:
            raise ValueError(
                "Best chronological split still exceeds max_ratio_deviation: "
                f"{best_deviation:.4f} > {threshold:.4f}. "
                "Consider relaxing split ratios or min_segments."
            )

    return int(best_candidate[4]), int(best_candidate[5])


def split_by_time_order(
    df: pd.DataFrame,
    split_config: Dict[str, float],
) -> pd.DataFrame:
    """
    按完整时间线顺序切 train / val / test。

    这里严格按时间有序的整条序列切分，而不是按 segment 整段分配，
    以贴合方案 B 对 70/10/20 时间顺序切分的定义。
    """
    out = df.reset_index(drop=True).copy()

    train_ratio = split_config["train_ratio"]
    val_ratio = split_config["val_ratio"]
    test_ratio = split_config["test_ratio"]

    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    total_rows = len(out)
    n_train = int(total_rows * train_ratio)
    n_val = int(total_rows * val_ratio)
    n_test = total_rows - n_train - n_val

    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(
            "按当前 split 比例切分后，至少有一个 split 为空；请调整 split 配置。"
        )

    out["split"] = "test"
    out.loc[out.index < n_train, "split"] = "train"
    out.loc[(out.index >= n_train) & (out.index < n_train + n_val), "split"] = "val"

    return out


def split_by_time_sorted_segments(
    df: pd.DataFrame,
    split_config: Dict[str, float],
    timestamp_col: str,
    history_steps: Optional[int] = None,
    horizon_steps: Optional[int] = None,
) -> pd.DataFrame:
    """
    按 segment 起始时间顺序分配完整 segment，并用累计预算单位逼近目标比例。

    默认预算单位为有效窗口数：
        max(seg_len - history_steps - horizon_steps + 1, 0)
    也可通过 split.allocation_unit='rows' 改为按行数逼近。
    """
    out = df.copy()
    summary = _time_sorted_segment_summary(
        df=out,
        timestamp_col=timestamp_col,
        split_config=split_config,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
    )
    train_end_idx, val_end_idx = _choose_time_sorted_segment_boundaries(
        segment_summary=summary,
        split_config=split_config,
    )

    train_ids = set(summary.loc[:train_end_idx, "segment_id"].astype(int).tolist())
    val_ids = set(summary.loc[train_end_idx + 1:val_end_idx, "segment_id"].astype(int).tolist())

    out["split"] = "test"
    out.loc[out["segment_id"].isin(train_ids), "split"] = "train"
    out.loc[out["segment_id"].isin(val_ids), "split"] = "val"
    return out


def split_by_time_order_and_segment(
    df: pd.DataFrame,
    split_config: Dict[str, float],
) -> pd.DataFrame:
    """向后兼容的别名。"""
    return split_by_time_order(df=df, split_config=split_config)
