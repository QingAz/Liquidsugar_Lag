#!/usr/bin/env python3

import argparse
import csv
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union
from xml.sax.saxutils import escape

import yaml


TIMESTAMP_FMT = "%Y-%m-%d %H:%M"
SPLITS = ("train", "val", "test")
LAG_COLORS = {
    0: "#475569",
    2: "#0f766e",
    4: "#b45309",
    6: "#b91c1c",
}
SPLIT_COLORS = {
    "train": "#dbeafe",
    "val": "#dcfce7",
    "test": "#fee2e2",
}


class InjectionBlock(NamedTuple):
    split_name: str
    segment_id: int
    lag_step: int
    start_idx: int
    end_idx: int
    src_start_idx: int
    src_end_idx: int
    block_length: int


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a stage2 lagged LiquidSugar preview without touching the original CSV."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/liquidsugar_lag.yaml"),
        help="YAML config path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs_stdlib/<experiment_name>_preview.",
    )
    return parser.parse_args()


def load_rows(input_csv: Path, timestamp_col: str) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_csv}")

        fieldnames = list(reader.fieldnames)
        numeric_columns = [col for col in fieldnames if col != timestamp_col]
        rows: List[Dict[str, Any]] = []

        for raw in reader:
            row: Dict[str, Any] = {}
            row[timestamp_col] = datetime.strptime(raw[timestamp_col], TIMESTAMP_FMT)
            for col in numeric_columns:
                row[col] = float(raw[col])
            rows.append(row)

    rows.sort(key=lambda item: item[timestamp_col])
    return rows, fieldnames, numeric_columns


def add_segments(
    rows: Sequence[Dict[str, Any]],
    timestamp_col: str,
    g_break_minutes: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    prev_ts = None  # type: Optional[datetime]
    segment_id = 0

    for row in rows:
        current = dict(row)
        if prev_ts is not None:
            gap_minutes = (current[timestamp_col] - prev_ts).total_seconds() / 60.0
            if gap_minutes > g_break_minutes:
                segment_id += 1
        current["segment_id"] = segment_id
        out.append(current)
        prev_ts = current[timestamp_col]

    return out


def validate_nominal_time_alignment(
    rows: Sequence[Dict[str, Any]],
    timestamp_col: str,
    nominal_step_minutes: int,
) -> None:
    for row in rows:
        ts = row[timestamp_col]
        if ts.minute % nominal_step_minutes != 0 or ts.second != 0 or ts.microsecond != 0:
            raise ValueError(
                f"Found a timestamp outside the nominal grid: {ts.strftime(TIMESTAMP_FMT)}"
            )


def regularize_to_nominal_grid(
    rows: Sequence[Dict[str, Any]],
    timestamp_col: str,
    numeric_columns: Sequence[str],
    nominal_step_minutes: int,
) -> List[Dict[str, Any]]:
    validate_nominal_time_alignment(rows, timestamp_col, nominal_step_minutes)
    freq = timedelta(minutes=nominal_step_minutes)

    segment_map: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        segment_map.setdefault(int(row["segment_id"]), []).append(row)

    out: List[Dict[str, Any]] = []
    for segment_id in sorted(segment_map):
        segment_rows = segment_map[segment_id]
        deduped: Dict[datetime, Dict[str, Any]] = {}
        for row in segment_rows:
            deduped[row[timestamp_col]] = row

        known_times = sorted(deduped)
        if not known_times:
            continue

        for index, current_ts in enumerate(known_times):
            current_row = deduped[current_ts]
            base_row = {col: current_row[col] for col in numeric_columns}
            base_row[timestamp_col] = current_ts
            base_row["segment_id"] = segment_id
            base_row["is_interpolated"] = False
            out.append(base_row)

            if index == len(known_times) - 1:
                continue

            next_ts = known_times[index + 1]
            next_row = deduped[next_ts]
            gap_steps = int((next_ts - current_ts) // freq)
            if gap_steps <= 1:
                continue

            for step in range(1, gap_steps):
                interp_ts = current_ts + step * freq
                ratio = step / gap_steps
                interp_row: Dict[str, Any] = {
                    timestamp_col: interp_ts,
                    "segment_id": segment_id,
                    "is_interpolated": True,
                }
                for col in numeric_columns:
                    start_value = current_row[col]
                    end_value = next_row[col]
                    interp_row[col] = start_value + (end_value - start_value) * ratio
                out.append(interp_row)

    out.sort(key=lambda item: item[timestamp_col])
    return out


def split_by_time_order(
    rows: Sequence[Dict[str, Any]],
    split_config: Dict[str, float],
) -> List[Dict[str, Any]]:
    out = [dict(row) for row in rows]

    train_ratio = float(split_config["train_ratio"])
    val_ratio = float(split_config["val_ratio"])
    test_ratio = float(split_config["test_ratio"])
    ratio_total = train_ratio + val_ratio + test_ratio
    if ratio_total <= 0:
        raise ValueError("Split ratios must sum to a positive number.")

    train_ratio /= ratio_total
    val_ratio /= ratio_total
    test_ratio /= ratio_total

    total_rows = len(out)
    n_train = int(total_rows * train_ratio)
    n_val = int(total_rows * val_ratio)
    n_test = total_rows - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("The current split would leave at least one split empty.")

    for index, row in enumerate(out):
        if index < n_train:
            row["split"] = "train"
        elif index < n_train + n_val:
            row["split"] = "val"
        else:
            row["split"] = "test"

    return out


def natural_sort_key(text: str) -> Tuple[str, int]:
    index = len(text) - 1
    while index >= 0 and text[index].isdigit():
        index -= 1
    suffix = text[index + 1:]
    prefix = text[: index + 1]
    if suffix:
        return (prefix, int(suffix))
    return (text, -1)


def infer_stage_columns(fieldnames: Sequence[str], prefix: str) -> List[str]:
    return sorted([col for col in fieldnames if col.startswith(prefix)], key=natural_sort_key)


def select_representative_columns(columns: Sequence[str], target_count: int) -> List[str]:
    if not columns:
        return []
    if len(columns) <= target_count:
        return list(columns)
    if target_count <= 1:
        return [columns[0]]

    selected: List[str] = []
    for order in range(target_count):
        index = int(round(order * (len(columns) - 1) / float(target_count - 1)))
        column = columns[index]
        if column not in selected:
            selected.append(column)
    return selected


def segment_summary(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Union[int, str]]]:
    summary: List[Dict[str, Union[int, str]]] = []
    index = 0

    while index < len(rows):
        split_name = str(rows[index]["split"])
        segment_id = int(rows[index]["segment_id"])
        start_idx = index
        while (
            index + 1 < len(rows)
            and str(rows[index + 1]["split"]) == split_name
            and int(rows[index + 1]["segment_id"]) == segment_id
        ):
            index += 1

        summary.append(
            {
                "split": split_name,
                "segment_id": segment_id,
                "seg_start": start_idx,
                "seg_end": index,
                "seg_len": index - start_idx + 1,
            }
        )
        index += 1

    return summary


def sample_block_length(rng: random.Random, injection_config: Dict[str, Any]) -> int:
    mode = injection_config.get("block_length_mode", "fixed")
    if mode == "fixed":
        return int(injection_config["block_length_default"])
    if mode == "random_range":
        block_min, block_max = injection_config["block_length_range"]
        return int(rng.randint(int(block_min), int(block_max)))
    raise ValueError(f"Unsupported block_length_mode={mode}")


def lookup_requested_blocks(
    blocks_per_split: Dict[str, Dict[Any, Any]],
    split_name: str,
    lag_step: int,
) -> int:
    lag_dict = blocks_per_split[split_name]
    if lag_step in lag_dict:
        return int(lag_dict[lag_step])
    if str(lag_step) in lag_dict:
        return int(lag_dict[str(lag_step)])
    raise KeyError(f"Missing block count for split={split_name}, lag={lag_step}")


def select_eligible_segments(
    split_summary: Sequence[Dict[str, Union[int, str]]],
    split_name: str,
    lag_steps: Sequence[int],
    blocks_per_split: Dict[str, Dict[Any, Any]],
    preferred_min_segment_length: int,
    fallback_min_segment_length: int,
    max_blocks_per_segment: int,
    used_segment_ids: Sequence[int],
) -> List[Dict[str, Union[int, str]]]:
    filtered = [
        row for row in split_summary if int(row["segment_id"]) not in set(used_segment_ids)
    ]

    required_blocks = sum(
        lookup_requested_blocks(blocks_per_split, split_name, int(lag_step))
        for lag_step in lag_steps
    )

    preferred = [row for row in filtered if int(row["seg_len"]) >= preferred_min_segment_length]
    fallback = [row for row in filtered if int(row["seg_len"]) >= fallback_min_segment_length]

    preferred_capacity = len(preferred) * max_blocks_per_segment
    fallback_capacity = len(fallback) * max_blocks_per_segment

    if preferred_capacity >= required_blocks:
        return preferred
    if fallback_capacity >= required_blocks:
        return fallback

    raise ValueError(
        f"{split_name} does not have enough eligible segments. "
        f"Need {required_blocks} blocks, but >= {fallback_min_segment_length} only allows {fallback_capacity}."
    )


def enforce_non_overlap(plan: Sequence[InjectionBlock], buffer_steps: int) -> List[InjectionBlock]:
    final_plan: List[InjectionBlock] = []
    by_segment: Dict[int, List[InjectionBlock]] = {}
    for block in plan:
        by_segment.setdefault(block.segment_id, []).append(block)

    for segment_id in sorted(by_segment):
        accepted: List[InjectionBlock] = []
        for block in sorted(by_segment[segment_id], key=lambda item: item.start_idx):
            if accepted:
                prev = accepted[-1]
                if block.start_idx <= prev.end_idx + buffer_steps:
                    raise ValueError(
                        f"Buffer violation inside segment {segment_id}: "
                        f"previous block {prev.start_idx}-{prev.end_idx}, "
                        f"current block {block.start_idx}-{block.end_idx}."
                    )
            accepted.append(block)
        final_plan.extend(accepted)

    return final_plan


def sample_injection_plan(
    rows: Sequence[Dict[str, Any]],
    injection_config: Dict[str, Any],
    seed: int,
) -> List[InjectionBlock]:
    rng = random.Random(seed)
    summary = segment_summary(rows)

    lag_steps = [int(step) for step in injection_config["lag_steps"]]
    buffer_steps = int(injection_config["buffer_steps"])
    preferred_min_segment_length = int(injection_config["preferred_min_segment_length"])
    fallback_min_segment_length = int(injection_config["fallback_min_segment_length"])
    max_blocks_per_segment = int(injection_config["max_blocks_per_segment"])
    blocks_per_split = injection_config["blocks_per_split"]

    used_segment_ids: Dict[int, int] = {}
    plan: List[InjectionBlock] = []

    for split_name in SPLITS:
        split_rows = [row for row in summary if str(row["split"]) == split_name]
        eligible = select_eligible_segments(
            split_summary=split_rows,
            split_name=split_name,
            lag_steps=lag_steps,
            blocks_per_split=blocks_per_split,
            preferred_min_segment_length=preferred_min_segment_length,
            fallback_min_segment_length=fallback_min_segment_length,
            max_blocks_per_segment=max_blocks_per_segment,
            used_segment_ids=list(used_segment_ids.keys()),
        )

        for lag_step in lag_steps:
            n_blocks = lookup_requested_blocks(blocks_per_split, split_name, lag_step)
            for _ in range(n_blocks):
                candidates: List[Dict[str, int]] = []
                for row in eligible:
                    segment_id = int(row["segment_id"])
                    if used_segment_ids.get(segment_id, 0) >= max_blocks_per_segment:
                        continue

                    block_length = sample_block_length(rng, injection_config)
                    valid_start_min = int(row["seg_start"]) + lag_step
                    valid_start_max = int(row["seg_end"]) - block_length + 1
                    if valid_start_min > valid_start_max:
                        continue

                    candidates.append(
                        {
                            "segment_id": segment_id,
                            "seg_start": int(row["seg_start"]),
                            "seg_end": int(row["seg_end"]),
                            "block_length": block_length,
                            "valid_start_min": valid_start_min,
                            "valid_start_max": valid_start_max,
                        }
                    )

                if not candidates:
                    raise ValueError(
                        f"Unable to sample a legal block for split={split_name}, lag={lag_step}."
                    )

                chosen = rng.choice(candidates)
                start_idx = rng.randint(chosen["valid_start_min"], chosen["valid_start_max"])
                end_idx = start_idx + chosen["block_length"] - 1

                plan.append(
                    InjectionBlock(
                        split_name=split_name,
                        segment_id=chosen["segment_id"],
                        lag_step=lag_step,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        src_start_idx=start_idx - lag_step,
                        src_end_idx=end_idx - lag_step,
                        block_length=chosen["block_length"],
                    )
                )
                used_segment_ids[chosen["segment_id"]] = used_segment_ids.get(chosen["segment_id"], 0) + 1

    return enforce_non_overlap(plan, buffer_steps)


def apply_injection_plan(
    rows: Sequence[Dict[str, Any]],
    stage_cols: Sequence[str],
    plan: Sequence[InjectionBlock],
) -> List[Dict[str, Any]]:
    out = [dict(row) for row in rows]
    for row in out:
        row["lag_gt"] = 0
        row["g_stage1_to_stage2"] = 0

    for block in plan:
        for offset, row_index in enumerate(range(block.start_idx, block.end_idx + 1)):
            src_index = block.src_start_idx + offset
            for col in stage_cols:
                out[row_index][col] = rows[src_index][col]
            out[row_index]["lag_gt"] = block.lag_step
            out[row_index]["g_stage1_to_stage2"] = block.lag_step

    return out


def format_timestamp(ts: datetime) -> str:
    return ts.strftime(TIMESTAMP_FMT)


def save_csv(rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str], output_path: Path, timestamp_col: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            csv_row: Dict[str, Any] = {}
            for field in fieldnames:
                value = row[field]
                if isinstance(value, datetime):
                    csv_row[field] = format_timestamp(value)
                elif isinstance(value, bool):
                    csv_row[field] = int(value)
                else:
                    csv_row[field] = value
            writer.writerow(csv_row)


def save_json(obj: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def plan_to_dict(plan: Sequence[InjectionBlock], rows: Sequence[Dict[str, Any]], timestamp_col: str) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for block in plan:
        item = dict(block._asdict())
        item["start_timestamp"] = format_timestamp(rows[block.start_idx][timestamp_col])
        item["end_timestamp"] = format_timestamp(rows[block.end_idx][timestamp_col])
        item["src_start_timestamp"] = format_timestamp(rows[block.src_start_idx][timestamp_col])
        item["src_end_timestamp"] = format_timestamp(rows[block.src_end_idx][timestamp_col])
        serialized.append(item)
    return serialized


def split_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {split: 0 for split in SPLITS}
    interpolated = 0
    for row in rows:
        counts[str(row["split"])] += 1
        if row["is_interpolated"]:
            interpolated += 1

    total = len(rows)
    return {
        "row_count": total,
        "split_row_count": counts,
        "split_ratio_realized": {
            split: (counts[split] / total if total else 0.0)
            for split in SPLITS
        },
        "interpolated_row_count": interpolated,
    }


def contiguous_split_ranges(rows: Sequence[Dict[str, Any]]) -> List[Tuple[str, int, int]]:
    ranges: List[Tuple[str, int, int]] = []
    if not rows:
        return ranges

    start = 0
    current_split = str(rows[0]["split"])
    for index in range(1, len(rows)):
        next_split = str(rows[index]["split"])
        if next_split != current_split:
            ranges.append((current_split, start, index - 1))
            start = index
            current_split = next_split
    ranges.append((current_split, start, len(rows) - 1))
    return ranges


def segment_lookup(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, int], Tuple[int, int]]:
    lookup: Dict[Tuple[str, int], Tuple[int, int]] = {}
    for item in segment_summary(rows):
        key = (str(item["split"]), int(item["segment_id"]))
        lookup[key] = (int(item["seg_start"]), int(item["seg_end"]))
    return lookup


def svg_polyline(points: Iterable[Tuple[float, float]], stroke: str, stroke_width: float, fill: str = "none") -> str:
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polyline points="{coords}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{stroke_width:.2f}" stroke-linejoin="round" stroke-linecap="round" />'
    )


def svg_step_path(xs: Sequence[float], ys: Sequence[float]) -> str:
    if not xs:
        return ""
    commands = [f"M {xs[0]:.2f} {ys[0]:.2f}"]
    for index in range(1, len(xs)):
        commands.append(f"L {xs[index]:.2f} {ys[index - 1]:.2f}")
        commands.append(f"L {xs[index]:.2f} {ys[index]:.2f}")
    return " ".join(commands)


def write_lag_timeline_svg(rows: Sequence[Dict[str, Any]], output_path: Path, timestamp_col: str) -> None:
    width, height = 1500, 420
    margin_left, margin_top, margin_right, margin_bottom = 80, 50, 30, 55
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_lag = 6
    total_rows = max(len(rows) - 1, 1)

    def x_of(index: int) -> float:
        return margin_left + (index / total_rows) * plot_width

    def y_of(lag_value: int) -> float:
        return margin_top + plot_height - (lag_value / max_lag) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        '<text x="80" y="28" font-size="22" font-family="Arial, sans-serif" fill="#111827">Stage2 Lag Timeline</text>',
        '<text x="80" y="46" font-size="12" font-family="Arial, sans-serif" fill="#4b5563">Discrete ground-truth lag after stage2 injection</text>',
    ]

    for split_name, start_idx, end_idx in contiguous_split_ranges(rows):
        x0 = x_of(start_idx)
        x1 = x_of(end_idx)
        parts.append(
            f'<rect x="{x0:.2f}" y="{margin_top:.2f}" width="{max(x1 - x0, 2):.2f}" '
            f'height="{plot_height:.2f}" fill="{SPLIT_COLORS[split_name]}" opacity="0.45" />'
        )
        parts.append(
            f'<text x="{(x0 + x1) / 2:.2f}" y="{margin_top + 18:.2f}" text-anchor="middle" '
            f'font-size="12" font-family="Arial, sans-serif" fill="#1f2937">{escape(split_name)}</text>'
        )

    for lag_value in [0, 2, 4, 6]:
        y = y_of(lag_value)
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" '
            f'stroke="#cbd5e1" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="12" font-family="Arial, sans-serif" fill="#475569">{lag_value}</text>'
        )

    xs = [x_of(index) for index in range(len(rows))]
    ys = [y_of(int(row["lag_gt"])) for row in rows]
    path_d = svg_step_path(xs, ys)
    parts.append(
        f'<path d="{path_d}" fill="none" stroke="{LAG_COLORS[6]}" stroke-width="2.2" '
        'stroke-linejoin="round" stroke-linecap="round" />'
    )

    for block_color, label, offset in [
        (SPLIT_COLORS["train"], "train", 0),
        (SPLIT_COLORS["val"], "val", 84),
        (SPLIT_COLORS["test"], "test", 148),
    ]:
        x = width - 280 + offset
        parts.append(f'<rect x="{x}" y="20" width="16" height="10" fill="{block_color}" opacity="0.8" />')
        parts.append(
            f'<text x="{x + 22}" y="29" font-size="12" font-family="Arial, sans-serif" fill="#334155">{label}</text>'
        )

    parts.append(
        f'<line x1="{width - 280}" y1="42" x2="{width - 264}" y2="42" stroke="{LAG_COLORS[6]}" stroke-width="2.2" />'
    )
    parts.append(
        f'<text x="{width - 252}" y="46" font-size="12" font-family="Arial, sans-serif" fill="#334155">lag_gt</text>'
    )

    parts.append(
        f'<text x="{margin_left}" y="{height - 18}" font-size="12" font-family="Arial, sans-serif" fill="#475569">'
        f'{escape(format_timestamp(rows[0][timestamp_col]))}</text>'
    )
    parts.append(
        f'<text x="{width - margin_right}" y="{height - 18}" text-anchor="end" '
        f'font-size="12" font-family="Arial, sans-serif" fill="#475569">'
        f'{escape(format_timestamp(rows[-1][timestamp_col]))}</text>'
    )

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_stage_examples_svg(
    original_rows: Sequence[Dict[str, Any]],
    lagged_rows: Sequence[Dict[str, Any]],
    plan: Sequence[InjectionBlock],
    stage_column: str,
    timestamp_col: str,
    output_path: Path,
) -> None:
    first_block_per_split: List[InjectionBlock] = []
    for split_name in SPLITS:
        split_blocks = sorted(
            [block for block in plan if block.split_name == split_name],
            key=lambda item: item.start_idx,
        )
        if split_blocks:
            first_block_per_split.append(split_blocks[0])

    if not first_block_per_split:
        return

    lookup = segment_lookup(lagged_rows)
    panel_width = 1400
    panel_height = 230
    width = panel_width
    height = 40 + len(first_block_per_split) * panel_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        '<text x="60" y="26" font-size="22" font-family="Arial, sans-serif" fill="#111827">Stage2 Example Windows</text>',
        f'<text x="60" y="44" font-size="12" font-family="Arial, sans-serif" fill="#4b5563">Original vs lagged {escape(stage_column)}</text>',
    ]

    for panel_index, block in enumerate(first_block_per_split):
        top = 60 + panel_index * panel_height
        left = 60
        right = 30
        plot_width = width - left - right
        plot_height = 135
        seg_start, seg_end = lookup[(block.split_name, block.segment_id)]
        window_start = max(seg_start, block.start_idx - 40)
        window_end = min(seg_end, block.end_idx + 40)
        indices = list(range(window_start, window_end + 1))

        series_original = [float(original_rows[index][stage_column]) for index in indices]
        series_lagged = [float(lagged_rows[index][stage_column]) for index in indices]
        value_min = min(series_original + series_lagged)
        value_max = max(series_original + series_lagged)
        if math.isclose(value_min, value_max):
            value_min -= 1.0
            value_max += 1.0
        padding = (value_max - value_min) * 0.08
        value_min -= padding
        value_max += padding

        def x_of(index: int) -> float:
            if len(indices) == 1:
                return left + plot_width / 2.0
            return left + ((index - window_start) / (window_end - window_start)) * plot_width

        def y_of(value: float) -> float:
            ratio = (value - value_min) / (value_max - value_min)
            return top + plot_height - ratio * plot_height

        block_x0 = x_of(block.start_idx)
        block_x1 = x_of(block.end_idx)
        parts.append(
            f'<rect x="{block_x0:.2f}" y="{top:.2f}" width="{max(block_x1 - block_x0, 2):.2f}" '
            f'height="{plot_height:.2f}" fill="{LAG_COLORS[block.lag_step]}" opacity="0.12" />'
        )

        for value in [value_min, (value_min + value_max) / 2.0, value_max]:
            y = y_of(value)
            parts.append(
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" '
                f'stroke="#e2e8f0" stroke-width="1" />'
            )

        original_points = [(x_of(index), y_of(float(original_rows[index][stage_column]))) for index in indices]
        lagged_points = [(x_of(index), y_of(float(lagged_rows[index][stage_column]))) for index in indices]
        parts.append(svg_polyline(original_points, "#64748b", 1.8))
        parts.append(svg_polyline(lagged_points, LAG_COLORS[block.lag_step], 2.2))

        title = (
            f"{block.split_name} | segment {block.segment_id} | lag={block.lag_step} steps | "
            f"{format_timestamp(lagged_rows[block.start_idx][timestamp_col])} -> "
            f"{format_timestamp(lagged_rows[block.end_idx][timestamp_col])}"
        )
        parts.append(
            f'<text x="{left}" y="{top - 10}" font-size="13" font-family="Arial, sans-serif" fill="#111827">{escape(title)}</text>'
        )

        parts.append(
            f'<text x="{left}" y="{top + plot_height + 18}" font-size="11" font-family="Arial, sans-serif" fill="#475569">'
            f'{escape(format_timestamp(lagged_rows[window_start][timestamp_col]))}</text>'
        )
        parts.append(
            f'<text x="{left + plot_width}" y="{top + plot_height + 18}" text-anchor="end" '
            f'font-size="11" font-family="Arial, sans-serif" fill="#475569">'
            f'{escape(format_timestamp(lagged_rows[window_end][timestamp_col]))}</text>'
        )
        parts.append(
            f'<text x="{left}" y="{top + plot_height + 36}" font-size="11" font-family="Arial, sans-serif" fill="#64748b">gray = original</text>'
        )
        parts.append(
            f'<text x="{left + 120}" y="{top + plot_height + 36}" font-size="11" font-family="Arial, sans-serif" fill="{LAG_COLORS[block.lag_step]}">'
            f'color = lagged</text>'
        )

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def block_file_stem(block: InjectionBlock) -> str:
    return (
        f"{block.split_name}_seg{block.segment_id}_"
        f"lag{block.lag_step}_start{block.start_idx}_end{block.end_idx}"
    )


def block_window_indices(
    lookup: Dict[Tuple[str, int], Tuple[int, int]],
    block: InjectionBlock,
    pad_steps: int,
) -> Tuple[int, int]:
    seg_start, seg_end = lookup[(block.split_name, block.segment_id)]
    window_start = max(seg_start, block.src_start_idx - pad_steps)
    window_end = min(seg_end, block.end_idx + pad_steps)
    return window_start, window_end


def compute_value_bounds(values_a: Sequence[float], values_b: Sequence[float]) -> Tuple[float, float]:
    value_min = min(list(values_a) + list(values_b))
    value_max = max(list(values_a) + list(values_b))
    if math.isclose(value_min, value_max):
        value_min -= 1.0
        value_max += 1.0
    padding = (value_max - value_min) * 0.08
    return value_min - padding, value_max + padding


def write_block_detail_svg(
    original_rows: Sequence[Dict[str, Any]],
    lagged_rows: Sequence[Dict[str, Any]],
    block: InjectionBlock,
    stage_columns: Sequence[str],
    timestamp_col: str,
    output_path: Path,
) -> None:
    lookup = segment_lookup(lagged_rows)
    window_start, window_end = block_window_indices(lookup, block, pad_steps=24)
    indices = list(range(window_start, window_end + 1))

    width = 1500
    left = 78
    right = 34
    plot_width = width - left - right
    lag_panel_height = 86
    value_panel_height = 118
    panel_gap = 32
    top_margin = 96
    total_panels = 1 + len(stage_columns)
    height = top_margin + total_panels * value_panel_height + panel_gap * (total_panels - 1) + 44

    def x_of(index: int) -> float:
        if window_end == window_start:
            return left + plot_width / 2.0
        return left + ((index - window_start) / float(window_end - window_start)) * plot_width

    def shaded_regions(panel_top: float, panel_height: float) -> List[str]:
        source_x0 = x_of(block.src_start_idx)
        source_x1 = x_of(block.src_end_idx)
        block_x0 = x_of(block.start_idx)
        block_x1 = x_of(block.end_idx)
        return [
            f'<rect x="{source_x0:.2f}" y="{panel_top:.2f}" width="{max(source_x1 - source_x0, 2):.2f}" '
            f'height="{panel_height:.2f}" fill="#cbd5e1" opacity="0.40" />',
            f'<rect x="{block_x0:.2f}" y="{panel_top:.2f}" width="{max(block_x1 - block_x0, 2):.2f}" '
            f'height="{panel_height:.2f}" fill="{LAG_COLORS[block.lag_step]}" opacity="0.14" />',
            f'<line x1="{block_x0:.2f}" y1="{panel_top:.2f}" x2="{block_x0:.2f}" y2="{panel_top + panel_height:.2f}" '
            f'stroke="{LAG_COLORS[block.lag_step]}" stroke-width="1.4" stroke-dasharray="5 4" />',
            f'<line x1="{block_x1:.2f}" y1="{panel_top:.2f}" x2="{block_x1:.2f}" y2="{panel_top + panel_height:.2f}" '
            f'stroke="{LAG_COLORS[block.lag_step]}" stroke-width="1.4" stroke-dasharray="5 4" />',
        ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        '<text x="78" y="28" font-size="22" font-family="Arial, sans-serif" fill="#111827">Stage2 Lag Block Detail</text>',
        (
            f'<text x="78" y="48" font-size="12" font-family="Arial, sans-serif" fill="#475569">'
            f'{escape(block.split_name)} | segment {block.segment_id} | lag={block.lag_step} steps '
            f'({block.lag_step * 15} min) | block={block.block_length} steps '
            f'({block.block_length * 15 // 60}h {block.block_length * 15 % 60:02d}m)</text>'
        ),
        (
            f'<text x="78" y="66" font-size="12" font-family="Arial, sans-serif" fill="#475569">'
            f'injected: {escape(format_timestamp(lagged_rows[block.start_idx][timestamp_col]))} -> '
            f'{escape(format_timestamp(lagged_rows[block.end_idx][timestamp_col]))} | '
            f'source: {escape(format_timestamp(lagged_rows[block.src_start_idx][timestamp_col]))} -> '
            f'{escape(format_timestamp(lagged_rows[block.src_end_idx][timestamp_col]))}</text>'
        ),
        '<text x="78" y="84" font-size="11" font-family="Arial, sans-serif" fill="#64748b">gray shade = source slice, color shade = injected slice, gray line = original, color line = lagged</text>',
    ]

    lag_top = top_margin
    lag_bottom = lag_top + lag_panel_height

    for panel in shaded_regions(lag_top, lag_panel_height):
        parts.append(panel)

    def lag_y(value: int) -> float:
        return lag_bottom - (value / 6.0) * lag_panel_height

    for lag_value in [0, 2, 4, 6]:
        y = lag_y(lag_value)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e2e8f0" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="Arial, sans-serif" fill="#475569">{lag_value}</text>'
        )
    lag_xs = [x_of(index) for index in indices]
    lag_ys = [lag_y(int(lagged_rows[index]["lag_gt"])) for index in indices]
    parts.append(
        f'<path d="{svg_step_path(lag_xs, lag_ys)}" fill="none" stroke="{LAG_COLORS[block.lag_step]}" stroke-width="2.4" '
        'stroke-linejoin="round" stroke-linecap="round" />'
    )
    parts.append(
        f'<text x="{left}" y="{lag_top - 10}" font-size="13" font-family="Arial, sans-serif" fill="#111827">lag_gt around this block</text>'
    )

    for column_index, column in enumerate(stage_columns):
        panel_top = lag_bottom + panel_gap + column_index * (value_panel_height + panel_gap)
        panel_bottom = panel_top + value_panel_height
        for panel in shaded_regions(panel_top, value_panel_height):
            parts.append(panel)

        original_values = [float(original_rows[index][column]) for index in indices]
        lagged_values = [float(lagged_rows[index][column]) for index in indices]
        value_min, value_max = compute_value_bounds(original_values, lagged_values)

        def value_y(value: float) -> float:
            ratio = (value - value_min) / (value_max - value_min)
            return panel_bottom - ratio * value_panel_height

        for value in [value_min, (value_min + value_max) / 2.0, value_max]:
            y = value_y(value)
            parts.append(
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e2e8f0" stroke-width="1" />'
            )

        original_points = [(x_of(index), value_y(float(original_rows[index][column]))) for index in indices]
        lagged_points = [(x_of(index), value_y(float(lagged_rows[index][column]))) for index in indices]
        parts.append(svg_polyline(original_points, "#64748b", 1.7))
        parts.append(svg_polyline(lagged_points, LAG_COLORS[block.lag_step], 2.3))
        parts.append(
            f'<text x="{left}" y="{panel_top - 10}" font-size="13" font-family="Arial, sans-serif" fill="#111827">{escape(column)}</text>'
        )

    source_x0 = x_of(block.src_start_idx)
    block_x0 = x_of(block.start_idx)
    parts.append(f'<rect x="{width - 360}" y="20" width="14" height="10" fill="#cbd5e1" opacity="0.8" />')
    parts.append(
        f'<text x="{width - 340}" y="29" font-size="11" font-family="Arial, sans-serif" fill="#334155">source slice</text>'
    )
    parts.append(
        f'<rect x="{width - 250}" y="20" width="14" height="10" fill="{LAG_COLORS[block.lag_step]}" opacity="0.35" />'
    )
    parts.append(
        f'<text x="{width - 230}" y="29" font-size="11" font-family="Arial, sans-serif" fill="#334155">injected block</text>'
    )
    parts.append(
        f'<line x1="{width - 110}" y1="25" x2="{width - 94}" y2="25" stroke="{LAG_COLORS[block.lag_step]}" stroke-width="2.3" />'
    )
    parts.append(
        f'<text x="{width - 86}" y="29" font-size="11" font-family="Arial, sans-serif" fill="#334155">lagged curve</text>'
    )

    parts.append(
        f'<text x="{left}" y="{height - 14}" font-size="11" font-family="Arial, sans-serif" fill="#475569">{escape(format_timestamp(lagged_rows[window_start][timestamp_col]))}</text>'
    )
    parts.append(
        f'<text x="{left + plot_width}" y="{height - 14}" text-anchor="end" font-size="11" font-family="Arial, sans-serif" fill="#475569">{escape(format_timestamp(lagged_rows[window_end][timestamp_col]))}</text>'
    )
    parts.append(
        f'<text x="{source_x0:.2f}" y="{lag_top + 16:.2f}" font-size="10" font-family="Arial, sans-serif" fill="#334155">source</text>'
    )
    parts.append(
        f'<text x="{block_x0:.2f}" y="{lag_top + 30:.2f}" font-size="10" font-family="Arial, sans-serif" fill="{LAG_COLORS[block.lag_step]}">inject</text>'
    )
    parts.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_block_gallery_html(
    rows: Sequence[Dict[str, Any]],
    plan: Sequence[InjectionBlock],
    stage_columns: Sequence[str],
    timestamp_col: str,
    output_path: Path,
) -> None:
    items: List[str] = []
    for block in sorted(plan, key=lambda item: (SPLITS.index(item.split_name), item.start_idx)):
        file_name = block_file_stem(block) + ".svg"
        items.append(
            """
            <article class="card">
              <div class="meta">
                <h2>{title}</h2>
                <p><strong>Injected:</strong> {start_ts} to {end_ts}</p>
                <p><strong>Source:</strong> {src_start_ts} to {src_end_ts}</p>
                <p><strong>Delay:</strong> {lag_steps} steps ({lag_minutes} min)</p>
                <p><strong>Block length:</strong> {block_length} steps</p>
              </div>
              <a class="image-wrap" href="blocks/{file_name}" target="_blank" rel="noopener noreferrer">
                <img src="blocks/{file_name}" alt="{title}" loading="lazy" />
              </a>
            </article>
            """.format(
                title=escape(
                    "{split_name} | segment {segment_id} | lag={lag_step}".format(
                        split_name=block.split_name,
                        segment_id=block.segment_id,
                        lag_step=block.lag_step,
                    )
                ),
                start_ts=escape(format_timestamp(rows[block.start_idx][timestamp_col])),
                end_ts=escape(format_timestamp(rows[block.end_idx][timestamp_col])),
                src_start_ts=escape(format_timestamp(rows[block.src_start_idx][timestamp_col])),
                src_end_ts=escape(format_timestamp(rows[block.src_end_idx][timestamp_col])),
                lag_steps=block.lag_step,
                lag_minutes=block.lag_step * 15,
                block_length=block.block_length,
                file_name=escape(file_name),
            ).strip()
        )

    summary_bits = []
    for split_name in SPLITS:
        blocks = [block for block in plan if block.split_name == split_name]
        summary_bits.append(
            "{split_name}: {count} blocks".format(split_name=split_name, count=len(blocks))
        )

    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stage2 Lag Block Gallery</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --text: #0f172a;
      --muted: #475569;
      --card: #ffffff;
      --line: #e2e8f0;
    }}
    body {{
      margin: 0;
      padding: 28px;
      font-family: Arial, sans-serif;
      background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
      color: var(--text);
    }}
    .shell {{
      max-width: 1540px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 10px 0;
      font-size: 30px;
    }}
    .intro {{
      color: var(--muted);
      margin-bottom: 18px;
      line-height: 1.45;
    }}
    .links {{
      margin-bottom: 20px;
      font-size: 14px;
    }}
    .links a {{
      color: #0f766e;
      margin-right: 18px;
      text-decoration: none;
      font-weight: 600;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 20px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 14px 32px rgba(15, 23, 42, 0.06);
    }}
    .meta h2 {{
      margin: 0 0 8px 0;
      font-size: 20px;
    }}
    .meta p {{
      margin: 4px 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .image-wrap {{
      display: block;
      margin-top: 14px;
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid var(--line);
      background: #fff;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      background: #fff;
    }}
  </style>
</head>
<body>
  <div class="shell">
    <h1>Stage2 Lag Block Gallery</h1>
    <p class="intro">
      Detailed block-by-block zoom around every injected lag region.
      Representative stage2 columns shown: {columns}.
      Summary: {summary}.
    </p>
    <div class="links">
      <a href="lag_timeline.svg">Open full lag timeline</a>
      <a href="stage2_var_example.svg">Open quick example sheet</a>
    </div>
    <section class="grid">
      {items}
    </section>
  </div>
</body>
</html>
""".format(
        columns=", ".join(escape(column) for column in stage_columns),
        summary=" | ".join(summary_bits),
        items="\n".join(items),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def build_outputs(config: Dict[str, Any], output_dir: Path) -> None:
    timestamp_col = str(config["timestamp_col"])
    input_csv = Path(config["input_csv"])
    nominal_step_minutes = int(config["nominal_step_minutes"])
    seed = int(config["seed"])
    stage2_prefix = str(config["feature_prefix"]["stage2"])

    rows, fieldnames, numeric_columns = load_rows(input_csv=input_csv, timestamp_col=timestamp_col)
    segmented_rows = add_segments(
        rows=rows,
        timestamp_col=timestamp_col,
        g_break_minutes=int(config["g_break_minutes"]),
    )
    regularized_rows = regularize_to_nominal_grid(
        rows=segmented_rows,
        timestamp_col=timestamp_col,
        numeric_columns=numeric_columns,
        nominal_step_minutes=nominal_step_minutes,
    )
    split_rows = split_by_time_order(
        rows=regularized_rows,
        split_config=config["split"],
    )

    stage2_cols = infer_stage_columns(fieldnames, stage2_prefix)
    if not stage2_cols:
        raise ValueError("No stage2 columns found in the input CSV.")
    detail_stage2_cols = select_representative_columns(stage2_cols, target_count=4)

    plan = sample_injection_plan(
        rows=split_rows,
        injection_config=config["injection"],
        seed=seed,
    )
    lagged_rows = apply_injection_plan(split_rows, stage2_cols, plan)

    output_fieldnames = list(fieldnames) + [
        "segment_id",
        "is_interpolated",
        "split",
        "lag_gt",
        "g_stage1_to_stage2",
    ]

    save_csv(
        rows=lagged_rows,
        fieldnames=output_fieldnames,
        output_path=output_dir / "raw" / "liquidsugar_stage2_lagged.csv",
        timestamp_col=timestamp_col,
    )
    save_csv(
        rows=[
            {
                timestamp_col: row[timestamp_col],
                "split": row["split"],
                "segment_id": row["segment_id"],
                "lag_gt": row["lag_gt"],
                "g_stage1_to_stage2": row["g_stage1_to_stage2"],
            }
            for row in lagged_rows
        ],
        fieldnames=[
            timestamp_col,
            "split",
            "segment_id",
            "lag_gt",
            "g_stage1_to_stage2",
        ],
        output_path=output_dir / "ground_truth" / "lag_curve_stage1_to_stage2.csv",
        timestamp_col=timestamp_col,
    )
    save_json(
        plan_to_dict(plan, lagged_rows, timestamp_col),
        output_dir / "metadata" / "injection_plan.json",
    )
    save_json(
        {
            "input_csv": str(input_csv),
            "original_csv_unchanged": True,
            "timestamp_col": timestamp_col,
            "target_col": config["target_col"],
            "stage2_columns": stage2_cols,
            "detail_stage2_columns": detail_stage2_cols,
            "nominal_step_minutes": nominal_step_minutes,
            "split_summary": split_summary(lagged_rows),
        },
        output_dir / "metadata" / "run_summary.json",
    )

    write_lag_timeline_svg(
        rows=lagged_rows,
        output_path=output_dir / "plots" / "lag_timeline.svg",
        timestamp_col=timestamp_col,
    )
    write_stage_examples_svg(
        original_rows=split_rows,
        lagged_rows=lagged_rows,
        plan=plan,
        stage_column=stage2_cols[0],
        timestamp_col=timestamp_col,
        output_path=output_dir / "plots" / "stage2_var_example.svg",
    )
    for block in plan:
        write_block_detail_svg(
            original_rows=split_rows,
            lagged_rows=lagged_rows,
            block=block,
            stage_columns=detail_stage2_cols,
            timestamp_col=timestamp_col,
            output_path=output_dir / "plots" / "blocks" / (block_file_stem(block) + ".svg"),
        )
    write_block_gallery_html(
        rows=lagged_rows,
        plan=plan,
        stage_columns=detail_stage2_cols,
        timestamp_col=timestamp_col,
        output_path=output_dir / "plots" / "block_gallery.html",
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.output_dir is None:
        output_dir = Path("outputs_stdlib") / f"{config['experiment_name']}_preview"
    else:
        output_dir = args.output_dir

    build_outputs(config=config, output_dir=output_dir)
    print(f"Generated lagged preview under: {output_dir}")


if __name__ == "__main__":
    main()
