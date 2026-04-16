#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

MPLCONFIGDIR = Path(tempfile.mkdtemp(prefix="matplotlib-", dir="/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.preprocess import (
    add_segments,
    load_dataframe,
    regularize_to_nominal_grid,
    split_by_time_sorted_segments,
)


def _rebuild_no_lag_dataframe(config: Dict[str, object]) -> pd.DataFrame:
    timestamp_col = str(config["timestamp_col"])
    df = load_dataframe(input_csv=str(config["input_csv"]), timestamp_col=timestamp_col)
    df = add_segments(
        df=df,
        timestamp_col=timestamp_col,
        g_break_minutes=int(config["g_break_minutes"]),
    )
    df = regularize_to_nominal_grid(
        df=df,
        timestamp_col=timestamp_col,
        nominal_step_minutes=int(config["nominal_step_minutes"]),
    )
    df = split_by_time_sorted_segments(
        df=df,
        split_config=dict(config["split"]),
        timestamp_col=timestamp_col,
        history_steps=int(config["window"]["history_steps"]),
        horizon_steps=int(config["window"]["horizon_steps"]),
    )
    return df.reset_index(drop=True)


def _load_injected_dataframe(output_dir: Path, timestamp_col: str) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for split_name in ["train", "val", "test"]:
        csv_path = output_dir / "raw" / f"{split_name}.csv"
        split_df = pd.read_csv(csv_path, parse_dates=[timestamp_col])
        parts.append(split_df)
    df = pd.concat(parts, ignore_index=True)
    return df.reset_index(drop=True)


def _load_plan(output_dir: Path) -> List[Dict[str, object]]:
    plan_path = output_dir / "metadata" / "local_bump_plan.json"
    return json.loads(plan_path.read_text(encoding="utf-8"))


def _pick_representative_blocks(
    plan: List[Dict[str, object]],
    split_name: str,
) -> List[Dict[str, object]]:
    chosen: Dict[int, Dict[str, object]] = {}
    candidates = sorted(
        [item for item in plan if str(item["split_name"]) == split_name],
        key=lambda item: (int(item["dmax"]), int(item["segment_id"]), int(item["start_idx"])),
    )
    for item in candidates:
        dmax = int(item["dmax"])
        if dmax not in chosen:
            chosen[dmax] = item
    return [chosen[key] for key in sorted(chosen)]


def _segment_bounds(df: pd.DataFrame) -> Dict[int, Tuple[int, int]]:
    bounds: Dict[int, Tuple[int, int]] = {}
    for segment_id, segment_df in df.groupby("segment_id", sort=True):
        bounds[int(segment_id)] = (int(segment_df.index.min()), int(segment_df.index.max()))
    return bounds


def _window_for_block(
    bounds: Dict[int, Tuple[int, int]],
    block: Dict[str, object],
    context_steps: int,
) -> Tuple[int, int]:
    seg_start, seg_end = bounds[int(block["segment_id"])]
    start_idx = int(block["start_idx"])
    end_idx = int(block["end_idx"])
    window_start = max(seg_start, start_idx - int(context_steps))
    window_end = min(seg_end, end_idx + int(context_steps))
    return window_start, window_end


def _mean_abs_diff(series_a: pd.Series, series_b: pd.Series) -> float:
    return float(np.mean(np.abs(series_a.to_numpy() - series_b.to_numpy())))


def _auto_select_signal(
    base_df: pd.DataFrame,
    injected_df: pd.DataFrame,
    blocks: List[Dict[str, object]],
    context_steps: int,
) -> str:
    stage2_cols = [col for col in base_df.columns if col.startswith("stage2_")]
    bounds = _segment_bounds(base_df)

    best_col = stage2_cols[0]
    best_score = float("-inf")
    for col in stage2_cols:
        scores: List[float] = []
        for block in blocks:
            start_idx, end_idx = _window_for_block(bounds, block, context_steps=context_steps)
            scores.append(
                _mean_abs_diff(
                    base_df.loc[start_idx:end_idx, col],
                    injected_df.loc[start_idx:end_idx, col],
                )
            )
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_col = col
    return best_col


def _validate_alignment(
    base_df: pd.DataFrame,
    injected_df: pd.DataFrame,
    timestamp_col: str,
) -> None:
    if len(base_df) != len(injected_df):
        raise ValueError("No-lag baseline and injected dataframe have different row counts.")

    base_timestamps = pd.to_datetime(base_df[timestamp_col])
    injected_timestamps = pd.to_datetime(injected_df[timestamp_col])
    if not base_timestamps.equals(injected_timestamps):
        raise ValueError("Timestamp alignment mismatch between baseline and injected dataframe.")

    if not base_df["segment_id"].equals(injected_df["segment_id"]):
        raise ValueError("segment_id alignment mismatch between baseline and injected dataframe.")


def _make_summary_rows(
    base_df: pd.DataFrame,
    injected_df: pd.DataFrame,
    blocks: List[Dict[str, object]],
    signal_col: str,
    timestamp_col: str,
    context_steps: int,
) -> List[Dict[str, object]]:
    bounds = _segment_bounds(base_df)
    rows: List[Dict[str, object]] = []

    for block in blocks:
        window_start, window_end = _window_for_block(bounds, block, context_steps=context_steps)
        start_idx = int(block["start_idx"])
        end_idx = int(block["end_idx"])
        base_signal = base_df.loc[window_start:window_end, signal_col]
        lagged_signal = injected_df.loc[window_start:window_end, signal_col]
        lag_series = injected_df.loc[window_start:window_end, "lag_gt"]

        rows.append(
            {
                "split_name": str(block["split_name"]),
                "segment_id": int(block["segment_id"]),
                "dmax": int(block["dmax"]),
                "signal_col": signal_col,
                "window_start_idx": window_start,
                "window_end_idx": window_end,
                "bump_start_idx": start_idx,
                "bump_end_idx": end_idx,
                "window_start_time": str(pd.to_datetime(base_df.loc[window_start, timestamp_col])),
                "window_end_time": str(pd.to_datetime(base_df.loc[window_end, timestamp_col])),
                "bump_start_time": str(pd.to_datetime(base_df.loc[start_idx, timestamp_col])),
                "bump_end_time": str(pd.to_datetime(base_df.loc[end_idx, timestamp_col])),
                "signal_mad_in_window": _mean_abs_diff(base_signal, lagged_signal),
                "signal_mad_in_bump_only": _mean_abs_diff(
                    base_df.loc[start_idx:end_idx, signal_col],
                    injected_df.loc[start_idx:end_idx, signal_col],
                ),
                "yield_flow_mad_in_bump_only": _mean_abs_diff(
                    base_df.loc[start_idx:end_idx, "yield_flow"],
                    injected_df.loc[start_idx:end_idx, "yield_flow"],
                ),
                "lag_pattern": injected_df.loc[start_idx:end_idx, "lag_gt"].astype(int).tolist(),
                "nonzero_lag_steps_in_window": int((lag_series > 0).sum()),
            }
        )
    return rows


def _plot_comparison(
    base_df: pd.DataFrame,
    injected_df: pd.DataFrame,
    blocks: List[Dict[str, object]],
    signal_col: str,
    timestamp_col: str,
    context_steps: int,
    output_svg: Path,
    output_png: Path,
) -> List[Dict[str, object]]:
    plt.style.use("seaborn-v0_8-whitegrid")
    bounds = _segment_bounds(base_df)
    summary_rows = _make_summary_rows(
        base_df=base_df,
        injected_df=injected_df,
        blocks=blocks,
        signal_col=signal_col,
        timestamp_col=timestamp_col,
        context_steps=context_steps,
    )

    fig = plt.figure(figsize=(16, 12))
    outer = fig.add_gridspec(
        nrows=len(blocks),
        ncols=1,
        top=0.92,
        bottom=0.07,
        left=0.06,
        right=0.98,
        hspace=0.12,
    )

    legend_handles = None
    legend_labels = None

    for row_idx, (block, summary) in enumerate(zip(blocks, summary_rows)):
        subgrid = outer[row_idx].subgridspec(nrows=2, ncols=1, height_ratios=[4.2, 1.2], hspace=0.03)
        ax_signal = fig.add_subplot(subgrid[0])
        ax_lag = fig.add_subplot(subgrid[1], sharex=ax_signal)

        window_start, window_end = _window_for_block(bounds, block, context_steps=context_steps)
        start_idx = int(block["start_idx"])
        end_idx = int(block["end_idx"])
        time_window = pd.to_datetime(base_df.loc[window_start:window_end, timestamp_col])
        bump_start_time = pd.to_datetime(base_df.loc[start_idx, timestamp_col])
        bump_end_time = pd.to_datetime(base_df.loc[end_idx, timestamp_col])

        base_signal = base_df.loc[window_start:window_end, signal_col]
        lagged_signal = injected_df.loc[window_start:window_end, signal_col]
        lag_series = injected_df.loc[window_start:window_end, "lag_gt"]

        ax_signal.axvspan(bump_start_time, bump_end_time, color="#fde68a", alpha=0.30, linewidth=0)
        ax_signal.fill_between(
            time_window,
            base_signal.to_numpy(),
            lagged_signal.to_numpy(),
            color="#fca5a5",
            alpha=0.20,
        )
        ax_signal.plot(
            time_window,
            base_signal,
            color="#2563eb",
            linewidth=2.2,
            label="Original no-lag",
        )
        ax_signal.plot(
            time_window,
            lagged_signal,
            color="#dc2626",
            linewidth=2.2,
            label="Latest bump-lag",
        )
        ax_signal.axvline(bump_start_time, color="#b45309", linestyle="--", linewidth=1.2, alpha=0.9)
        ax_signal.axvline(bump_end_time, color="#b45309", linestyle="--", linewidth=1.2, alpha=0.9)
        ax_signal.set_ylabel(f"{signal_col}\n(raw)")
        ax_signal.set_title(
            f"dmax={int(block['dmax'])} | test segment {int(block['segment_id'])} | "
            f"signal MAD (bump-only)={summary['signal_mad_in_bump_only']:.2f}",
            loc="left",
            fontsize=12,
            pad=8,
        )
        ax_signal.text(
            0.01,
            0.98,
            f"bump: {summary['bump_start_time'][5:16]} -> {summary['bump_end_time'][5:16]}  |  "
            f"lag pattern: {summary['lag_pattern']}",
            transform=ax_signal.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            color="#475569",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#ffffff", "edgecolor": "#e2e8f0", "alpha": 0.9},
        )

        ax_lag.axhline(0, color="#cbd5e1", linewidth=1.0)
        ax_lag.axvspan(bump_start_time, bump_end_time, color="#fde68a", alpha=0.30, linewidth=0)
        ax_lag.fill_between(
            time_window,
            0,
            lag_series.to_numpy(),
            step="mid",
            color="#f59e0b",
            alpha=0.25,
        )
        ax_lag.step(
            time_window,
            lag_series.to_numpy(),
            where="mid",
            color="#92400e",
            linewidth=2.0,
            label="lag_gt",
        )
        ax_lag.set_ylabel("lag_gt")
        ax_lag.set_ylim(-0.3, max(1.0, float(lag_series.max()) + 0.8))

        if row_idx < len(blocks) - 1:
            ax_signal.tick_params(labelbottom=False)
            ax_lag.tick_params(labelbottom=False)
        else:
            ax_lag.set_xlabel("Timestamp")
            ax_lag.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

        if legend_handles is None:
            legend_handles, legend_labels = ax_signal.get_legend_handles_labels()

    fig.suptitle(
        "Original No-Lag vs Latest Local Bump Lag",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.01,
        0.965,
        f"Dataset: liquidsugar_local_bump_stage12_smooth_mixed_balanced | "
        f"Representative test segments for dmax=2/4/6 | Auto-selected signal: {signal_col}",
        ha="left",
        va="top",
        fontsize=11,
        color="#475569",
    )
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper right",
            bbox_to_anchor=(0.985, 0.976),
            frameon=False,
            ncol=2,
        )

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg, bbox_inches="tight")
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return summary_rows


def main() -> None:
    parser = ArgumentParser(description="Plot original no-lag vs local-bump lag comparison.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "liquidsugar_local_bump_mixed_balanced.yaml"),
        help="Path to the local bump config used to rebuild the no-lag baseline.",
    )
    parser.add_argument(
        "--lagged-output-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "liquidsugar_local_bump_stage12_smooth_mixed_balanced"),
        help="Existing local-bump output directory with raw/*.csv and metadata/local_bump_plan.json.",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="test",
        help="Which split to sample representative dmax segments from.",
    )
    parser.add_argument(
        "--context-steps",
        type=int,
        default=16,
        help="How many extra timesteps to keep before and after each bump window.",
    )
    parser.add_argument(
        "--signal-col",
        type=str,
        default="auto",
        help="Stage-2 signal to plot. Use 'auto' to select the most visibly shifted column.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="original_vs_bump_lag_test_segments",
        help="Base filename written under <lagged-output-dir>/plots/.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    output_dir = Path(args.lagged_output_dir)
    timestamp_col = str(config["timestamp_col"])

    base_df = _rebuild_no_lag_dataframe(config)
    injected_df = _load_injected_dataframe(output_dir=output_dir, timestamp_col=timestamp_col)
    _validate_alignment(base_df=base_df, injected_df=injected_df, timestamp_col=timestamp_col)

    blocks = _pick_representative_blocks(_load_plan(output_dir), split_name=args.split_name)
    if not blocks:
        raise ValueError(f"No local bump blocks found for split={args.split_name!r}.")

    signal_col = args.signal_col
    if signal_col == "auto":
        signal_col = _auto_select_signal(
            base_df=base_df,
            injected_df=injected_df,
            blocks=blocks,
            context_steps=int(args.context_steps),
        )

    plots_dir = output_dir / "plots"
    output_svg = plots_dir / f"{args.output_name}.svg"
    output_png = plots_dir / f"{args.output_name}.png"
    output_json = plots_dir / f"{args.output_name}.summary.json"

    summary_rows = _plot_comparison(
        base_df=base_df,
        injected_df=injected_df,
        blocks=blocks,
        signal_col=signal_col,
        timestamp_col=timestamp_col,
        context_steps=int(args.context_steps),
        output_svg=output_svg,
        output_png=output_png,
    )

    summary = {
        "config_path": str(Path(args.config).resolve()),
        "lagged_output_dir": str(output_dir.resolve()),
        "split_name": args.split_name,
        "context_steps": int(args.context_steps),
        "signal_col": signal_col,
        "signal_selection_mode": "auto_max_avg_mad" if args.signal_col == "auto" else "manual",
        "representative_segments": summary_rows,
        "artifacts": {
            "svg": str(output_svg.resolve()),
            "png": str(output_png.resolve()),
        },
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
