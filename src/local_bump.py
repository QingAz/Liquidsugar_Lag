import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .preprocess import (
    add_segments,
    infer_column_groups,
    load_dataframe,
    regularize_to_nominal_grid,
    split_by_time_sorted_segments,
)
from .scaling import apply_zscore, fit_zscore_stats
from .windows import make_windows


@dataclass
class LocalBumpPlan:
    split_name: str
    segment_id: int
    shape: str
    dmax: int
    width: int
    start_idx: int
    end_idx: int


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["split", "segment_id"], sort=True)
        .agg(
            seg_start=("segment_id", lambda x: x.index.min()),
            seg_end=("segment_id", lambda x: x.index.max()),
            seg_len=("segment_id", "size"),
        )
        .reset_index()
    )
    return summary


def _effective_window_count(seg_len: int, history_steps: int, horizon_steps: int) -> int:
    return max(int(seg_len) - int(history_steps) - int(horizon_steps) + 1, 0)


def _with_effective_windows(
    summary: pd.DataFrame,
    history_steps: int,
    horizon_steps: int,
) -> pd.DataFrame:
    out = summary.copy()
    out["effective_window_count"] = out["seg_len"].apply(
        lambda seg_len: _effective_window_count(
            seg_len=seg_len,
            history_steps=history_steps,
            horizon_steps=horizon_steps,
        )
    ).astype(int)
    return out


def _split_budget_summary(df: pd.DataFrame, history_steps: int, horizon_steps: int) -> Dict[str, Any]:
    """统计各 split 的 segment 数与有效窗口数，便于核对整段切分结果。"""
    segment_view = (
        df.groupby(["split", "segment_id"], sort=True)
        .agg(seg_len=("segment_id", "size"))
        .reset_index()
    )
    segment_view["effective_window_count"] = segment_view["seg_len"].apply(
        lambda seg_len: _effective_window_count(
            seg_len=seg_len,
            history_steps=history_steps,
            horizon_steps=horizon_steps,
        )
    ).astype(int)

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


def _eligible_segments(
    summary: pd.DataFrame,
    width: int,
    margin: int,
    min_predictable_offset: int = 0,
) -> pd.DataFrame:
    left_margin = max(int(margin), int(min_predictable_offset))
    min_len = int(width) + left_margin + int(margin)
    return summary.loc[summary["seg_len"] >= min_len].copy()


def _sample_injected_segments(
    summary: pd.DataFrame,
    inject_ratio: float,
    seed: int,
    min_injected_segments: Optional[Dict[str, int]] = None,
) -> Tuple[List[int], Dict[str, int]]:
    rng = np.random.default_rng(seed)
    injected_ids: List[int] = []
    counts: Dict[str, int] = {}
    min_injected_segments = min_injected_segments or {}

    for split_name in ["train", "val", "test"]:
        segs = summary.loc[summary["split"] == split_name, "segment_id"].tolist()
        if not segs:
            counts[split_name] = 0
            continue
        n_inject = int(round(len(segs) * inject_ratio))
        requested_min = int(min_injected_segments.get(split_name, 0))
        effective_min = min(len(segs), requested_min)
        n_inject = max(n_inject, effective_min)
        if inject_ratio > 0.0:
            n_inject = max(1, n_inject)
        if 0.0 < inject_ratio < 1.0 and len(segs) > 1:
            lower_bound = effective_min if effective_min < len(segs) else len(segs)
            upper_bound = len(segs) - 1 if effective_min < len(segs) else len(segs)
            n_inject = max(lower_bound, min(n_inject, upper_bound))
        n_inject = max(0, min(n_inject, len(segs)))
        chosen = rng.choice(segs, size=n_inject, replace=False).tolist() if n_inject > 0 else []
        injected_ids.extend(chosen)
        counts[split_name] = n_inject
    return injected_ids, counts


def _summarize_dmax_assignment(
    injected_summary: pd.DataFrame,
    assignment: Dict[int, int],
    dmax_choices: List[int],
    history_steps: int,
    horizon_steps: int,
    mode: str,
    min_segments_per_dmax: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    min_segments_per_dmax = min_segments_per_dmax or {}
    summary = _with_effective_windows(
        injected_summary.loc[injected_summary["segment_id"].isin(assignment.keys())].copy(),
        history_steps=history_steps,
        horizon_steps=horizon_steps,
    )
    summary["assigned_dmax"] = summary["segment_id"].map(assignment).astype(int)

    overall_segment_counts = {
        str(int(dmax)): int((summary["assigned_dmax"] == int(dmax)).sum()) for dmax in dmax_choices
    }
    overall_window_counts = {
        str(int(dmax)): int(
            summary.loc[summary["assigned_dmax"] == int(dmax), "effective_window_count"].sum()
        )
        for dmax in dmax_choices
    }

    per_split: Dict[str, Any] = {}
    for split_name in ["train", "val", "test"]:
        split_df = summary.loc[summary["split"] == split_name].copy()
        n_segments = int(len(split_df))
        requested_min = int(min_segments_per_dmax.get(split_name, 0))
        effective_min = min(requested_min, n_segments // max(len(dmax_choices), 1))
        dmax_segment_counts = {
            str(int(dmax)): int((split_df["assigned_dmax"] == int(dmax)).sum()) for dmax in dmax_choices
        }
        dmax_window_counts = {
            str(int(dmax)): int(
                split_df.loc[split_df["assigned_dmax"] == int(dmax), "effective_window_count"].sum()
            )
            for dmax in dmax_choices
        }
        total_windows = int(sum(dmax_window_counts.values()))
        dmax_window_ratios = {
            key: (float(value) / float(total_windows) if total_windows > 0 else 0.0)
            for key, value in dmax_window_counts.items()
        }
        per_split[split_name] = {
            "injected_segment_count": n_segments,
            "requested_min_segments_per_dmax": requested_min,
            "effective_min_segments_per_dmax": effective_min,
            "all_dmax_present": bool(
                n_segments >= len(dmax_choices)
                and all(count > 0 for count in dmax_segment_counts.values())
            ),
            "dmax_segment_counts": dmax_segment_counts,
            "dmax_effective_window_counts": dmax_window_counts,
            "dmax_effective_window_ratios": dmax_window_ratios,
        }

    return {
        "mode": mode,
        "overall_dmax_segment_counts": overall_segment_counts,
        "overall_dmax_effective_window_counts": overall_window_counts,
        "per_split": per_split,
    }


def _random_dmax_assignment(
    injected_summary: pd.DataFrame,
    dmax_choices: List[int],
    seed: int,
    history_steps: int,
    horizon_steps: int,
) -> Tuple[Dict[int, int], Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    assignment: Dict[int, int] = {}
    for row in injected_summary.itertuples(index=False):
        assignment[int(row.segment_id)] = int(rng.choice(dmax_choices))
    return assignment, _summarize_dmax_assignment(
        injected_summary=injected_summary,
        assignment=assignment,
        dmax_choices=dmax_choices,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        mode="random",
    )


def _balanced_dmax_assignment(
    injected_summary: pd.DataFrame,
    dmax_choices: List[int],
    history_steps: int,
    horizon_steps: int,
    min_segments_per_dmax: Optional[Dict[str, int]] = None,
) -> Tuple[Dict[int, int], Dict[str, Any]]:
    min_segments_per_dmax = min_segments_per_dmax or {}
    summary = _with_effective_windows(
        injected_summary,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
    )
    assignment: Dict[int, int] = {}

    for split_name in ["train", "val", "test"]:
        split_df = summary.loc[summary["split"] == split_name].copy()
        if split_df.empty:
            continue

        split_df = split_df.sort_values(
            ["effective_window_count", "seg_len", "segment_id"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        requested_min = int(min_segments_per_dmax.get(split_name, 0))
        effective_min = min(requested_min, len(split_df) // max(len(dmax_choices), 1))
        dmax_segment_counts = {int(dmax): 0 for dmax in dmax_choices}
        dmax_window_counts = {int(dmax): 0 for dmax in dmax_choices}

        for row in split_df.itertuples(index=False):
            underfilled = [
                int(dmax) for dmax in dmax_choices if dmax_segment_counts[int(dmax)] < effective_min
            ]
            candidate_dmax = underfilled if underfilled else [int(dmax) for dmax in dmax_choices]
            chosen_dmax = min(
                candidate_dmax,
                key=lambda dmax: (
                    dmax_window_counts[int(dmax)],
                    dmax_segment_counts[int(dmax)],
                    int(dmax),
                ),
            )
            assignment[int(row.segment_id)] = int(chosen_dmax)
            dmax_segment_counts[int(chosen_dmax)] += 1
            dmax_window_counts[int(chosen_dmax)] += int(row.effective_window_count)

    return assignment, _summarize_dmax_assignment(
        injected_summary=injected_summary,
        assignment=assignment,
        dmax_choices=dmax_choices,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        mode="balanced_by_effective_windows",
        min_segments_per_dmax=min_segments_per_dmax,
    )


def _cosine_bump(width: int, dmax: int) -> np.ndarray:
    if width < 2:
        return np.asarray([float(dmax)])
    idx = np.arange(width, dtype=np.float64)
    phase = 2.0 * np.pi * idx / float(width - 1)
    values = dmax * (1.0 - np.cos(phase)) * 0.5
    return values


def _build_bump_profile(shape: str, width: int, dmax: int) -> np.ndarray:
    shape = shape.lower()
    if shape == "block":
        return np.full(width, float(dmax))
    if shape == "smooth":
        return _cosine_bump(width, dmax)
    raise ValueError(f"Unknown bump shape: {shape}")


def _sample_bump_window(
    rng: np.random.Generator,
    seg_start: int,
    seg_end: int,
    width: int,
    margin: int,
    min_predictable_offset: int = 0,
    raw_timestamp_mask: Optional[np.ndarray] = None,
    positive_lags: Optional[np.ndarray] = None,
) -> Tuple[int, int]:
    valid_start_min = max(seg_start + margin, seg_start + int(min_predictable_offset))
    valid_start_max = seg_end - margin - width + 1
    if valid_start_min > valid_start_max:
        raise ValueError("Segment too short for requested width/margin.")

    candidate_starts = np.arange(valid_start_min, valid_start_max + 1, dtype=int)
    if raw_timestamp_mask is not None and positive_lags is not None:
        positive_offsets = np.flatnonzero(np.asarray(positive_lags, dtype=int) > 0)
        if positive_offsets.size > 0:
            feasible: List[int] = []
            for start_idx in candidate_starts.tolist():
                positive_indices = start_idx + positive_offsets
                if raw_timestamp_mask[positive_indices].all():
                    feasible.append(int(start_idx))
            if not feasible:
                raise ValueError("No feasible bump window keeps all positive-lag timestamps on raw rows.")
            candidate_starts = np.asarray(feasible, dtype=int)

    start_idx = int(rng.choice(candidate_starts))
    end_idx = start_idx + width - 1
    return start_idx, end_idx


def _apply_local_bump(
    df: pd.DataFrame,
    plan: List[LocalBumpPlan],
    shift_cols: List[str],
) -> pd.DataFrame:
    out = df.copy()
    out["lag_gt"] = 0
    out["lag_binary_gt"] = 0
    out["inject_flag"] = 0
    out["bump_dmax_gt"] = 0
    out["segment_dmax_gt"] = 0

    base = df
    for block in plan:
        bump = _build_bump_profile(block.shape, block.width, block.dmax)
        bump_int = np.rint(bump).astype(int)
        indices = np.arange(block.start_idx, block.end_idx + 1)
        lags = bump_int[: len(indices)]

        out.loc[out["segment_id"] == block.segment_id, "inject_flag"] = 1
        out.loc[out["segment_id"] == block.segment_id, "segment_dmax_gt"] = int(block.dmax)
        out.loc[indices, "lag_gt"] = lags
        out.loc[indices, "bump_dmax_gt"] = int(block.dmax)

        for col in shift_cols:
            src_indices = indices - lags
            out.loc[indices, col] = base.loc[src_indices, col].to_numpy()

    out["lag_binary_gt"] = (out["lag_gt"] > 0).astype(int)
    return out


def build_local_bump_dataset(config: Dict[str, Any]) -> None:
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

    df = load_dataframe(input_csv=config["input_csv"], timestamp_col=timestamp_col)
    df = add_segments(df=df, timestamp_col=timestamp_col, g_break_minutes=int(config["g_break_minutes"]))
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

    bump_cfg = config["local_bump"]
    shape = str(bump_cfg["shape"]).lower()
    dmax_choices = bump_cfg.get("dmax_choices")
    if dmax_choices is not None:
        dmax_choices = [int(x) for x in dmax_choices]
        if not dmax_choices:
            raise ValueError("dmax_choices must be a non-empty list")
        dmax = max(dmax_choices)
    else:
        dmax = int(bump_cfg["dmax"])
    width = int(bump_cfg["width"])
    inject_ratio = float(bump_cfg.get("inject_ratio", 0.5))
    margin = int(bump_cfg.get("margin", dmax + 4))
    enforce_predictable_region = bool(bump_cfg.get("enforce_predictable_region", False))
    predictable_margin = (
        int(history_steps) + int(horizon_steps) - 1 if enforce_predictable_region else 0
    )
    require_raw_positive_timestamps = bool(bump_cfg.get("require_raw_positive_timestamps", False))
    dmax_assignment_cfg = bump_cfg.get("dmax_assignment", {})
    dmax_assignment_mode = str(dmax_assignment_cfg.get("mode", "random")).lower()
    min_segments_per_dmax = {
        str(split_name): int(value)
        for split_name, value in dmax_assignment_cfg.get("min_segments_per_dmax", {}).items()
    }

    summary = _segment_summary(df)
    eligible = _eligible_segments(
        summary,
        width=width,
        margin=margin,
        min_predictable_offset=predictable_margin,
    )
    min_injected_segments: Dict[str, int] = {}
    if dmax_choices is not None and dmax_assignment_mode == "balanced_by_effective_windows":
        min_injected_segments = {
            str(split_name): int(len(dmax_choices) * int(value))
            for split_name, value in min_segments_per_dmax.items()
        }
    injected_ids, injected_counts = _sample_injected_segments(
        eligible,
        inject_ratio,
        seed=seed,
        min_injected_segments=min_injected_segments,
    )

    rng = np.random.default_rng(seed + 1)
    injected_summary = eligible.loc[eligible["segment_id"].isin(injected_ids)].copy()
    if dmax_choices is None:
        dmax_assignment: Dict[int, int] = {}
        dmax_assignment_summary: Dict[str, Any] = {
            "mode": "fixed",
            "overall_dmax_segment_counts": {str(int(dmax)): int(len(injected_summary))},
            "overall_dmax_effective_window_counts": {
                str(int(dmax)): int(
                    _with_effective_windows(
                        injected_summary,
                        history_steps=history_steps,
                        horizon_steps=horizon_steps,
                    )["effective_window_count"].sum()
                )
            },
            "per_split": {},
        }
    elif dmax_assignment_mode == "balanced_by_effective_windows":
        dmax_assignment, dmax_assignment_summary = _balanced_dmax_assignment(
            injected_summary=injected_summary,
            dmax_choices=dmax_choices,
            history_steps=history_steps,
            horizon_steps=horizon_steps,
            min_segments_per_dmax=min_segments_per_dmax,
        )
    elif dmax_assignment_mode == "random":
        dmax_assignment, dmax_assignment_summary = _random_dmax_assignment(
            injected_summary=injected_summary,
            dmax_choices=dmax_choices,
            seed=seed + 2,
            history_steps=history_steps,
            horizon_steps=horizon_steps,
        )
    else:
        raise ValueError(f"Unknown dmax_assignment mode: {dmax_assignment_mode}")

    plan: List[LocalBumpPlan] = []
    raw_timestamp_mask = (~df["is_interpolated"].astype(bool)).to_numpy()
    for row in eligible.itertuples(index=False):
        if int(row.segment_id) not in injected_ids:
            continue
        if dmax_choices is None:
            chosen_dmax = dmax
        else:
            chosen_dmax = int(dmax_assignment[int(row.segment_id)])
        bump_int = np.rint(_build_bump_profile(shape, width, chosen_dmax)).astype(int)
        start_idx, end_idx = _sample_bump_window(
            rng=rng,
            seg_start=int(row.seg_start),
            seg_end=int(row.seg_end),
            width=width,
            margin=margin,
            min_predictable_offset=predictable_margin,
            raw_timestamp_mask=raw_timestamp_mask if require_raw_positive_timestamps else None,
            positive_lags=bump_int if require_raw_positive_timestamps else None,
        )
        plan.append(
            LocalBumpPlan(
                split_name=str(row.split),
                segment_id=int(row.segment_id),
                shape=shape,
                dmax=chosen_dmax,
                width=width,
                start_idx=start_idx,
                end_idx=end_idx,
            )
        )

    if not plan:
        raise ValueError("No injected segments selected; reduce margin or increase inject_ratio.")

    # stage1 -> stage2 injection: shift stage2, stage3, and target only.
    shift_cols = groups.stage2 + groups.stage3 + [groups.target]
    injected_df = _apply_local_bump(df, plan=plan, shift_cols=shift_cols)

    gt_curve_col = "g_stage1_to_stage2"
    injected_df[gt_curve_col] = injected_df["lag_gt"]

    _save_json([asdict(p) for p in plan], meta_dir / "local_bump_plan.json")
    _save_json(
        {
            "shape": shape,
            "dmax": dmax,
            "dmax_choices": dmax_choices,
            "width": width,
            "margin": margin,
            "enforce_predictable_region": enforce_predictable_region,
            "predictable_margin": predictable_margin,
            "require_raw_positive_timestamps": require_raw_positive_timestamps,
            "inject_ratio": inject_ratio,
            "eligible_segment_counts": {
                split_name: int((eligible["split"] == split_name).sum()) for split_name in ["train", "val", "test"]
            },
            "split_allocation_unit": str(config["split"].get("allocation_unit", "effective_windows")),
            "min_injected_segment_counts": min_injected_segments,
            "injected_segment_counts": injected_counts,
            "injected_dmax_counts": dmax_assignment_summary["overall_dmax_segment_counts"],
            "injected_dmax_effective_window_counts": dmax_assignment_summary[
                "overall_dmax_effective_window_counts"
            ],
            "dmax_assignment": dmax_assignment_summary,
            "split_budget": _split_budget_summary(
                df=injected_df,
                history_steps=history_steps,
                horizon_steps=horizon_steps,
            ),
        },
        meta_dir / "local_bump_summary.json",
    )

    _save_csv(
        injected_df[[timestamp_col, "split", "segment_id", "lag_gt", "lag_binary_gt", gt_curve_col]].copy(),
        gt_dir / "lag_curve_stage1_to_stage2.csv",
    )

    for split_name in ["train", "val", "test"]:
        split_df = injected_df.loc[injected_df["split"] == split_name].copy()
        _save_csv(split_df, raw_dir / f"{split_name}.csv")

    train_df = injected_df.loc[injected_df["split"] == "train"].copy()
    scale_columns = groups.feature_columns + [groups.target]
    stats = fit_zscore_stats(df=train_df, columns=scale_columns)
    _save_json(stats, meta_dir / "scaler_stats.json")

    normalized_split_data: Dict[str, pd.DataFrame] = {}
    for split_name in ["train", "val", "test"]:
        split_df = injected_df.loc[injected_df["split"] == split_name].copy()
        norm_df = apply_zscore(df=split_df, stats=stats, columns=scale_columns)
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

    print(f"Local-bump dataset built: {output_dir}")
