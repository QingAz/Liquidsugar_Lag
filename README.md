# LiquidSugar Scheme B Lag Framework

这个项目用于按你们当前定下来的 **方案 B 标准主实验**，生成带有局部块状 lag 的 LiquidSugar 数据集，并输出可直接用于 DIMF 的窗口样本。

默认主实验设定：

- 时间列：`TimeStamp`
- 目标列：`yield_flow`
- 名义时间步：`15 min`
- 原始 gap 分段阈值：`120 min`
- 切分方式：完整时间线 `70 / 10 / 20`
- 注入对象：`stage2_*`
- lag 类型：`2 / 4 / 6` step
- lag 物理含义：`30 / 60 / 90 min`
- block 默认长度：`32 step`
- block 可选范围：`24 ~ 48 step`
- 合法 segment 规则：优先 `>=160`，不足时放宽到 `>=128`
- train / val / test 都注入 lag，但注入位置不同

## 1. 框架执行的真实 pipeline

1. 读取原始 `LiquidSugar.csv`，按 `TimeStamp` 升序排序。
2. 用 `g_break_minutes = 120` 在原始数据上切分 segment。
3. 在每个原始 segment 内重建 `15 min` 名义时间网格，并对数值列做时间插值。
4. 在完整时间线上按顺序切分 `train / val / test = 70 / 10 / 20`。
5. 只对 `stage2_*` 变量注入局部块状硬平移 lag。
6. 用注入后的 `train` 估计标准化统计量，并应用到 `val / test`。
7. 构造滑动窗口，并用窗口末端时刻的真实 lag 作为 `lag_label`。

对应的注入形式是：

\[
x^{(2)}_{\text{new}}(t)=x^{(2)}(t-d), \quad d\in\{2,4,6\}
\]

ground-truth lag 曲线满足：

\[
g(t)\in\{0,2,4,6\}
\]

## 2. 输出内容

运行完成后，会在 `output_root/experiment_name/` 下生成：

- `raw/train.csv` / `raw/val.csv` / `raw/test.csv`
  注入 lag 后、尚未标准化的数据；其中包含 `lag_gt`、显式 stage-pair ground-truth 列和 `is_interpolated`
- `normalized/train.csv` / `normalized/val.csv` / `normalized/test.csv`
  使用注入后 `train` 的统计量标准化后的数据
- `ground_truth/lag_curve_stage1_to_stage2.csv`
  显式保存的 `g_{1→2}(t)` 曲线
- `metadata/injection_plan.json`
  每个 lag block 的注入计划
- `metadata/scaler_stats.json`
  标准化统计量
- `metadata/feature_columns.json`
  特征列、目标列、注入 stage、ground-truth 列名
- `metadata/split_summary.json`
  实际切分行数、比例、插值行数
- `windows/train.npz` / `val.npz` / `test.npz`
  可直接用于训练的窗口样本

## 3. 主实验默认配置

默认配置文件：

```bash
configs/liquidsugar_lag.yaml
```

默认主实验是：

- `lag_steps: [2, 4, 6]`
- `block_length_mode: fixed`
- `block_length_default: 32`
- `block_length_range: [24, 48]`
- `preferred_min_segment_length: 160`
- `fallback_min_segment_length: 128`
- `buffer_steps: 16`
- `max_blocks_per_segment: 1`
- `train`: 每种 lag 2 段
- `val`: 每种 lag 1 段
- `test`: 每种 lag 2 段

如果之后想做 sensitivity，可以把：

```yaml
block_length_mode: random_range
```

这样 block 长度就会在 `24 ~ 48` 内随机采样。

## 4. 运行方式

建议在 Quest 上准备独立 Python 环境后运行。

安装依赖：

```bash
pip install -r requirements.txt
```

运行：

```bash
python3 run_build_dataset.py --config configs/liquidsugar_lag.yaml
```

如果你们在 Quest 上已经有自己的 `venv` 或 `conda` 环境，只需要把配置文件里的：

- `input_csv`
- `output_root`
- `experiment_name`

改成 Quest 上的实际路径即可。

## 5. 窗口标签定义

窗口样本定义为：

\[
X_t=[x_{t-L+1},\dots,x_t], \quad y_t=yield\_flow_{t+H}
\]

并额外输出：

- `lag_label`：窗口末端时刻的真实 lag
- `timestamp_end`：窗口末端时间戳
- `segment_id`：对应原始 segment

## 6. 一句话总结

这套框架现在对应的是：

> **先按原始 gap 切 segment，再在 segment 内规则化到 15 分钟时间网格，然后按完整时间线顺序切分 70/10/20，最后在各 split 的合法连续 segment 内，对 stage2 全部变量注入 2/4/6 step 的局部块状 lag，并显式保存 `g_{1→2}(t)` 与窗口标签。**
