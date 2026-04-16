import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .injection import apply_injection_plan, plan_to_dict, sample_injection_plan
from .preprocess import (
    add_segments,
    infer_column_groups,
    load_dataframe,
    regularize_to_nominal_grid,
    split_by_time_sorted_segments,
)
from .scaling import apply_zscore, fit_zscore_stats
from .windows import make_windows


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _infer_stage_pair_name(injection_stage: str) -> str:
    mapping = {
        "stage1": "feed_to_stage1",
        "stage2": "stage1_to_stage2",
        "stage3": "stage2_to_stage3",
    }
    if injection_stage not in mapping:
        raise ValueError(f"未知 injection_stage={injection_stage}")
    return mapping[injection_stage]


def _split_budget_summary(df: pd.DataFrame, history_steps: int, horizon_steps: int) -> Dict[str, Any]:
    """统计各 split 的 segment 数与有效窗口数，便于核对整段切分结果。"""
    segment_view = (
        df.groupby(["split", "segment_id"], sort=True)
        .agg(seg_len=("segment_id", "size"))
        .reset_index()
    )
    segment_view["effective_window_count"] = (
        segment_view["seg_len"] - int(history_steps) - int(horizon_steps) + 1
    ).clip(lower=0).astype(int)

    split_names = ["train", "val", "test"]
    segment_count = (
        segment_view.groupby("split")["segment_id"]
        .nunique()
        .reindex(split_names)
        .fillna(0)
        .astype(int)
        .to_dict()
    )
    effective_window_count = (
        segment_view.groupby("split")["effective_window_count"]
        .sum()
        .reindex(split_names)
        .fillna(0)
        .astype(int)
        .to_dict()
    )
    total_windows = int(sum(effective_window_count.values()))
    effective_window_ratio = {
        split: (effective_window_count[split] / total_windows if total_windows else 0.0)
        for split in split_names
    }
    return {
        "segment_count": segment_count,
        "effective_window_count": effective_window_count,
        "effective_window_ratio_realized": effective_window_ratio,
    }


def build_dataset(config: Dict[str, Any]) -> None:
    """
    主流程：
    1) 读取原始 LiquidSugar
    2) 按 gap 分 segment
    3) 在每个 segment 内规则化到 15 min 名义时间网格
    4) 按 segment 起始时间顺序切完整 segment，并用有效窗口数逼近 train / val / test 比例
    5) 在各 split 内对 stage2 注入局部 lag
    6) 用 train 注入后数据拟合标准化统计量
    7) 构造滑窗样本并保存
    """
    output_dir = Path(config["output_root"]) / config["experiment_name"]
    raw_dir = output_dir / "raw"
    norm_dir = output_dir / "normalized"
    windows_dir = output_dir / "windows"
    meta_dir = output_dir / "metadata"
    gt_dir = output_dir / "ground_truth"

    timestamp_col = config["timestamp_col"]
    seed = int(config["seed"])
    nominal_step_minutes = int(config["nominal_step_minutes"])
    history_steps = int(config["window"]["history_steps"])
    horizon_steps = int(config["window"]["horizon_steps"])

    df = load_dataframe(
        input_csv=config["input_csv"],
        timestamp_col=timestamp_col,
    )

    df = add_segments(
        df=df,
        timestamp_col=timestamp_col,
        g_break_minutes=int(config["g_break_minutes"]),
    )

    df = regularize_to_nominal_grid(
        df=df,
        timestamp_col=timestamp_col,
        nominal_step_minutes=nominal_step_minutes,
    )

    df = split_by_time_sorted_segments(
        df=df,
        split_config=config["split"],
        timestamp_col=timestamp_col,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
    )

    groups = infer_column_groups(df, config)
    injection_stage = config["injection"]["injection_stage"]
    stage_pair_name = _infer_stage_pair_name(injection_stage)
    gt_curve_col = f"g_{stage_pair_name}"
    stage_cols = getattr(groups, injection_stage)

    plan = sample_injection_plan(
        df=df,
        injection_config=config["injection"],
        seed=seed,
    )

    injected_df = apply_injection_plan(
        df=df,
        stage_cols=stage_cols,
        plan=plan,
    )
    injected_df[gt_curve_col] = injected_df["lag_gt"]

    _save_json(plan_to_dict(plan), meta_dir / "injection_plan.json")
    _save_json(
        {
            "feature_columns": groups.feature_columns,
            "target_col": groups.target,
            "injection_stage": injection_stage,
            "stage_pair_name": stage_pair_name,
            "injection_columns": stage_cols,
            "ground_truth_column": gt_curve_col,
        },
        meta_dir / "feature_columns.json",
    )
    split_counts = (
        injected_df["split"]
        .value_counts(sort=False)
        .reindex(["train", "val", "test"])
        .fillna(0)
        .astype(int)
        .to_dict()
    )
    total_rows = int(len(injected_df))
    split_budget = _split_budget_summary(
        df=injected_df,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
    )
    _save_json(
        {
            "nominal_step_minutes": nominal_step_minutes,
            "split_allocation_unit": str(config["split"].get("allocation_unit", "effective_windows")),
            "row_count": total_rows,
            "split_row_count": split_counts,
            "split_ratio_realized": {
                key: (value / total_rows if total_rows else 0.0)
                for key, value in split_counts.items()
            },
            "split_segment_count": split_budget["segment_count"],
            "split_effective_window_count": split_budget["effective_window_count"],
            "split_effective_window_ratio_realized": split_budget["effective_window_ratio_realized"],
            "interpolated_row_count": int(injected_df["is_interpolated"].sum()),
        },
        meta_dir / "split_summary.json",
    )
    _save_csv(
        injected_df[[timestamp_col, "split", "segment_id", "lag_gt", gt_curve_col]].copy(),
        gt_dir / f"lag_curve_{stage_pair_name}.csv",
    )

    for split_name in ["train", "val", "test"]:
        split_df = injected_df.loc[injected_df["split"] == split_name].copy()
        _save_csv(split_df, raw_dir / f"{split_name}.csv")

    train_df = injected_df.loc[injected_df["split"] == "train"].copy()
    scale_columns = groups.feature_columns + [groups.target]

    stats = fit_zscore_stats(
        df=train_df,
        columns=scale_columns,
    )
    _save_json(stats, meta_dir / "scaler_stats.json")

    normalized_split_data: Dict[str, pd.DataFrame] = {}
    for split_name in ["train", "val", "test"]:
        split_df = injected_df.loc[injected_df["split"] == split_name].copy()
        norm_df = apply_zscore(
            df=split_df,
            stats=stats,
            columns=scale_columns,
        )
        normalized_split_data[split_name] = norm_df
        _save_csv(norm_df, norm_dir / f"{split_name}.csv")

    for split_name in ["train", "val", "test"]:
        arrays = make_windows(
            df=normalized_split_data[split_name],
            feature_columns=groups.feature_columns,
            target_col=groups.target,
            history_steps=history_steps,
            horizon_steps=horizon_steps,
            timestamp_col=timestamp_col,
        )

        windows_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(windows_dir / f"{split_name}.npz", **arrays)

    print(f"数据集构造完成，输出目录：{output_dir}")
