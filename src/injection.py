from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass
class InjectionBlock:
    """
    一个局部 lag 注入块。

    说明：
    - split_name: train / val / test
    - segment_id: 所属连续 segment
    - lag_step: 注入的 lag 大小
    - start_idx / end_idx: 在整个 DataFrame 中的全局行号
    - src_start_idx / src_end_idx: 该 block 实际引用的历史区间
    """
    split_name: str
    segment_id: int
    lag_step: int
    start_idx: int
    end_idx: int
    src_start_idx: int
    src_end_idx: int
    block_length: int


def _segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """提取每个 segment 的起止位置与长度。"""
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


def _sample_block_length(
    rng: np.random.Generator,
    injection_config: Dict[str, Any],
) -> int:
    """根据配置决定 block 长度是固定值还是随机值。"""
    mode = injection_config.get("block_length_mode", "fixed")

    if mode == "fixed":
        return int(injection_config["block_length_default"])

    if mode == "random_range":
        block_min, block_max = injection_config["block_length_range"]
        return int(rng.integers(block_min, block_max + 1))

    raise ValueError(f"不支持的 block_length_mode={mode}")


def _lookup_requested_blocks(
    blocks_per_split: Dict[str, Dict[Any, Any]],
    split_name: str,
    lag_step: int,
) -> int:
    """兼容 YAML 中 lag key 可能为 int 或 str 的写法。"""
    lag_dict = blocks_per_split[split_name]
    if lag_step in lag_dict:
        return int(lag_dict[lag_step])
    if str(lag_step) in lag_dict:
        return int(lag_dict[str(lag_step)])
    raise KeyError(f"{split_name} 中缺少 lag_step={lag_step} 的 block 数配置")


def _select_eligible_segments(
    split_summary: pd.DataFrame,
    split_name: str,
    lag_steps: List[int],
    blocks_per_split: Dict[str, Dict[Any, Any]],
    preferred_min_segment_length: int,
    fallback_min_segment_length: int,
    max_blocks_per_segment: int,
    used_segment_ids: List[int],
) -> pd.DataFrame:
    """
    按“优先 >=160，不够再放宽到 >=128”的规则选择可注入 segment。
    """
    if used_segment_ids:
        split_summary = split_summary.loc[
            ~split_summary["segment_id"].isin(used_segment_ids)
        ].copy()

    required_blocks = sum(
        _lookup_requested_blocks(blocks_per_split, split_name, lag_step)
        for lag_step in lag_steps
    )

    preferred = split_summary.loc[
        split_summary["seg_len"] >= preferred_min_segment_length
    ].copy()
    fallback = split_summary.loc[
        split_summary["seg_len"] >= fallback_min_segment_length
    ].copy()

    preferred_capacity = len(preferred) * max_blocks_per_segment
    fallback_capacity = len(fallback) * max_blocks_per_segment

    if preferred_capacity >= required_blocks:
        return preferred
    if fallback_capacity >= required_blocks:
        return fallback

    raise ValueError(
        f"{split_name} 可注入 segment 不足："
        f"需要 {required_blocks} 个 block，但 >= {fallback_min_segment_length} step 的 "
        f"segment 最多只能提供 {fallback_capacity} 个位置。"
    )


def sample_injection_plan(
    df: pd.DataFrame,
    injection_config: Dict[str, Any],
    seed: int,
) -> List[InjectionBlock]:
    """
    为 train / val / test 分别采样注入计划。

    当前版本默认：
    - 同一 segment 最多 1 个 block
    - block 内 lag 为常量
    - 整体上不同 block 的 lag 可以不同
    """
    rng = np.random.default_rng(seed)
    summary = _segment_summary(df)

    lag_steps = injection_config["lag_steps"]
    buffer_steps = injection_config["buffer_steps"]
    preferred_min_segment_length = int(
        injection_config.get(
            "preferred_min_segment_length",
            injection_config.get("min_segment_length", 160),
        )
    )
    fallback_min_segment_length = int(
        injection_config.get(
            "fallback_min_segment_length",
            injection_config.get("min_segment_length", preferred_min_segment_length),
        )
    )
    max_blocks_per_segment = injection_config["max_blocks_per_segment"]
    blocks_per_split = injection_config["blocks_per_split"]

    seg_used_count: Dict[int, int] = {}
    plan: List[InjectionBlock] = []

    for split_name in ["train", "val", "test"]:
        split_summary = summary.loc[summary["split"] == split_name].copy()
        split_summary = _select_eligible_segments(
            split_summary=split_summary,
            split_name=split_name,
            lag_steps=lag_steps,
            blocks_per_split=blocks_per_split,
            preferred_min_segment_length=preferred_min_segment_length,
            fallback_min_segment_length=fallback_min_segment_length,
            max_blocks_per_segment=max_blocks_per_segment,
            used_segment_ids=list(seg_used_count.keys()),
        )

        for lag_step in lag_steps:
            n_blocks = _lookup_requested_blocks(blocks_per_split, split_name, int(lag_step))

            for _ in range(n_blocks):
                candidate_rows: List[Dict[str, int]] = []

                for row in split_summary.itertuples(index=False):
                    seg_key = int(row.segment_id)
                    used_count = seg_used_count.get(seg_key, 0)

                    if used_count >= max_blocks_per_segment:
                        continue

                    block_length = _sample_block_length(rng, injection_config)

                    valid_start_min = int(row.seg_start) + int(lag_step)
                    valid_start_max = int(row.seg_end) - block_length + 1

                    if valid_start_min > valid_start_max:
                        continue

                    candidate_rows.append(
                        {
                            "segment_id": int(row.segment_id),
                            "seg_start": int(row.seg_start),
                            "seg_end": int(row.seg_end),
                            "block_length": block_length,
                            "valid_start_min": valid_start_min,
                            "valid_start_max": valid_start_max,
                        }
                    )

                if not candidate_rows:
                    raise ValueError(
                        f"{split_name} 无法为 lag={lag_step} 采样到合法 block。"
                        " 请检查 split 比例、segment 长度阈值或 block 数配置。"
                    )

                chosen = candidate_rows[int(rng.integers(0, len(candidate_rows)))]
                start_idx = int(rng.integers(chosen["valid_start_min"], chosen["valid_start_max"] + 1))
                end_idx = start_idx + chosen["block_length"] - 1

                block = InjectionBlock(
                    split_name=split_name,
                    segment_id=chosen["segment_id"],
                    lag_step=int(lag_step),
                    start_idx=start_idx,
                    end_idx=end_idx,
                    src_start_idx=start_idx - int(lag_step),
                    src_end_idx=end_idx - int(lag_step),
                    block_length=chosen["block_length"],
                )

                plan.append(block)
                seg_used_count[chosen["segment_id"]] = seg_used_count.get(
                    chosen["segment_id"], 0
                ) + 1

    plan = _enforce_non_overlap(plan, buffer_steps)
    return plan


def _enforce_non_overlap(plan: List[InjectionBlock], buffer_steps: int) -> List[InjectionBlock]:
    """
    在同一 split、同一 segment 内确保 block 之间不重叠，且至少留 buffer_steps 的正常缓冲区。

    不同 segment 之间本来就被原始时间 gap 分开，因此无需额外施加行级缓冲约束。
    """
    final_plan: List[InjectionBlock] = []

    segment_ids = sorted({b.segment_id for b in plan})

    for segment_id in segment_ids:
        blocks = [b for b in plan if b.segment_id == segment_id]
        blocks = sorted(blocks, key=lambda x: x.start_idx)

        accepted: List[InjectionBlock] = []
        for block in blocks:
            if not accepted:
                accepted.append(block)
                continue

            prev = accepted[-1]
            if block.start_idx <= prev.end_idx + buffer_steps:
                raise ValueError(
                    f"segment={segment_id} 中的 lag block 发生重叠或缓冲不足："
                    f"prev=({prev.start_idx},{prev.end_idx},{prev.split_name}), "
                    f"curr=({block.start_idx},{block.end_idx},{block.split_name})"
                )

            accepted.append(block)

        final_plan.extend(accepted)

    return final_plan


def apply_injection_plan(
    df: pd.DataFrame,
    stage_cols: List[str],
    plan: List[InjectionBlock],
) -> pd.DataFrame:
    """
    对指定工段列执行 lag 注入，并同步写入 ground-truth lag。

    公式：
        x_new(t) = x(t - d),  t in [a, b]
        x_new(t) = x(t),      otherwise
    """
    out = df.copy()
    out["lag_gt"] = 0

    for block in plan:
        for col in stage_cols:
            out.loc[block.start_idx:block.end_idx, col] = (
                df.loc[block.src_start_idx:block.src_end_idx, col].to_numpy()
            )

        out.loc[block.start_idx:block.end_idx, "lag_gt"] = block.lag_step

    return out


def plan_to_dict(plan: List[InjectionBlock]) -> List[Dict[str, Any]]:
    """将注入计划转为可写入 JSON 的字典列表。"""
    return [asdict(p) for p in plan]
