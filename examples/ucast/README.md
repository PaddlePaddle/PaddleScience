# U-Cast

本目录提供 U-Cast 的 PaddleScience 迁移实现，包括：

- 与原始 Dhariwal/ADM U-Net 对齐的 `ppsci.arch.UCast`；
- PyTorch Lightning EMA 权重到 Paddle `.pdparams` 的转换工具；
- WeatherBench-2 ERA5 数据读取、归一化、周期边界卷积；
- MC-Dropout 集成、自回归预报、fair CRPS 和 RMSE；
- ERA5 单步训练数据集、WMAE 第一阶段和 fair-CRPS 第二阶段；
- 原生 `ppsci.solver.Solver.train()` / `Solver.eval()`、Muon+AdamW、梯度累积、warmup+cosine 和全局梯度裁剪；
- 与原版一致的逐 batch EMA、多步自回归验证和完整训练状态恢复；
- NetCDF 预报结果导出。

当前入口覆盖发布权重推理、第一阶段确定性训练和第二阶段概率微调。
训练使用 `SupervisedConstraint` + `FunctionalLoss`，验证使用 `SupervisedValidator` +
`FunctionalMetric`；通用 Solver、训练 epoch 函数和验证函数均保持不变。
U-Cast 的调度、逐 microbatch EMA 和精确中途恢复通过官方回调接入。

## 环境

在 PaddleScience 环境中安装推理所需的可选依赖：

```bash
conda activate paddlescience
python -m pip install "xarray>=2024.1" "zarr<3" gcsfs netCDF4 safetensors dask
```

转换的第一阶段需要原 U-Cast 的 PyTorch 环境以及 `safetensors`：

```bash
conda activate ucast
python -m pip install safetensors
```

## 转换发布权重

由于 PyTorch 与 PaddlePaddle 不必安装在同一个环境，转换分成两个独立阶段。
以下命令从 PaddleScience 仓库根目录执行：

```bash
mkdir -p examples/ucast/checkpoints

conda run -n ucast python examples/ucast/convert_checkpoint.py export-pytorch \
  /path/to/ucast.ckpt \
  examples/ucast/checkpoints/ucast.safetensors

conda run -n paddlescience python examples/ucast/convert_checkpoint.py import-paddle \
  examples/ucast/checkpoints/ucast.safetensors \
  examples/ucast/checkpoints/ucast.pdparams
```

默认转换 EMA 权重，这与原项目的发布推理脚本一致。如需转换非 EMA 训练权重，
可在第一条命令末尾加 `--raw-weights`。

## 推理

配置文件默认从公共 WeatherBench-2 GCS 读取 ERA5，并假定原项目的统计文件位于
相邻的 `u-cast/data/stats`。进入示例目录后运行：

```bash
cd examples/ucast
CUDA_VISIBLE_DEVICES=0 conda run -n paddlescience python ucast.py
```

常用 Hydra 覆盖参数：

```bash
# 一个初始时刻、5 个成员、未来 15 天，并保存 NetCDF
python ucast.py \
  INFER.ic_start_dates='[2020-01-01]' \
  INFER.ensemble_size=5 \
  INFER.prediction_horizon=30 \
  INFER.output_path=./outputs_ucast/forecast.nc

# 本地 ERA5 和统计文件
python ucast.py \
  DATA.data_dir=/path/to/era5 \
  DATA.stats_dir=/path/to/stats
```

`prediction_horizon` 的单位是 12 小时。输入通道为 `2 × 83 + 4 + 2 = 172`：
两个历史场、四个时间强迫场和两个静态场；输出为 83 个归一化残差通道。

## 两阶段训练

训练已迁移为 Muon+AdamW：隐藏矩阵（包括输入卷积）使用 Muon，输出卷积与一维
参数使用 AdamW，归一化参数及偏置不做 AdamW weight decay。第一阶段使用 WMAE，
第二阶段使用 fair CRPS，训练 2 个 MC-Dropout 成员、验证 5 个成员。默认第一阶段
100 epochs、第二阶段 12 epochs，有效 batch size 48，1500 optimizer-step warmup。

RTX 3090 / 24GB 推荐使用以下配置：重计算、每卡 batch size 1、梯度累积 48 次；
验证每次前向处理一个 `(member, initial_condition)`，保留全部成员及各自的自回归轨迹。
以下命令在 `examples/ucast` 中执行：

```bash
# 第一阶段；数据目录应包含配置指定的 ERA5 .zarr 数据集
CUDA_VISIBLE_DEVICES=6 conda run --no-capture-output -n paddlescience \
  python ucast.py --config-name ucast_3090 \
  hydra.run.dir=./outputs_ucast/stage1 seed=11 \
  DATA.data_dir=/path/to/era5

# 第二阶段：加载第一阶段模型，重新初始化优化器、调度器和 EMA
CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n paddlescience \
  python ucast.py --config-name ucast_prob_3090 \
  hydra.run.dir=./outputs_ucast/stage2 seed=11 \
  DATA.data_dir=/path/to/era5 \
  TRAIN.pretrained_model_path=./outputs_ucast/stage1/checkpoints/best_raw.pdparams
```

显存更大的设备可以使用 `ucast.yaml mode=train` / `ucast_prob.yaml`，按需覆盖
`TRAIN.batch_size`、`TRAIN.eval_batch_size`、`TRAIN.update_freq`、
`MODEL.use_recompute` 和 `TRAIN.eval_forward_batch_size`。
单卡有效 batch size 为 `batch_size × update_freq`。当前入口支持单 GPU。

EMA 与原版 LitEma 一致，每个训练 batch 后更新一次，衰减率为
`min(decay, (1 + num_updates) / (10 + num_updates))`。验证使用 EMA 权重，启用
MC-Dropout（包括确定性阶段的单成员验证），默认自回归 4 步；跨全部 batch 汇总
逐变量面积加权的物理量 RMSE / L1，概率阶段另外计算 fair CRPS / SSR，再对
预报步和变量取平均。第一阶段按 `val/avg/rmse`、第二阶段按 `val/avg/crps`
保存最佳模型。Muon 的学习率、动量和 BF16 Newton-Schulz 均对照原实现迁移。

### 恢复训练

`TRAIN.pretrained_model_path` 只加载模型，用于 warm-start；
`TRAIN.checkpoint_path`（或兼容别名 `TRAIN.resume_path`）恢复同一训练任务。
恢复包括 Muon/AdamW 状态、调度位置、AMP scaler、EMA、随机数和 epoch 内的数据位置。

```bash
python ucast.py --config-name ucast_3090 \
  hydra.run.dir=./outputs_ucast/stage1 seed=11 \
  DATA.data_dir=/path/to/era5 \
  TRAIN.checkpoint_path=./outputs_ucast/stage1/checkpoints/last_raw
```

恢复时保持原模型、优化器、总 epochs、训练数据及 batch/累积配置一致。
恢复会还原完整训练状态；GPU AMP/BF16 运算仍可能出现浮点归约差异，不保证逐 bit
复现。回归测试的 FP32 容差为 `2e-6`，GPU 混合精度容差为 `1e-3`；详细实测见
[MIGRATION_REPORT.md](MIGRATION_REPORT.md)。
保持总 epochs 不变，可用 `TRAIN.stop_after_epoch=1` 先跑一个 epoch，恢复时设回 `null`；
也可设置 `TRAIN.max_steps` 在指定优化器步数后保存并停止，或设置
`TRAIN.save_freq_steps` 在优化器更新边界保存。

- `last_raw` / `best_raw`：PaddleScience 原生 `.pdparams`、`.pdopt`、`.pdstates`，
  AMP 对应 `.pdscaler`。`.pdstates` 扩展保存 U-Cast 的 EMA、随机数和采样器位置。
- `last.pdparams` / `best.pdparams`：可直接用于推理的 EMA 权重。
- `last.pdckpt` / `best.pdckpt`：指向对应原生文件组的轻量清单，恢复时保留整个文件组。
- Solver 自动保存的 `latest.*` 是通用任务快照；精确恢复请使用 `last_raw` / `best_raw`。
- 旧 PR 的 `.pdtrain` 训练状态与新优化器布局不同，可用其 `.pdparams` 做 warm-start；
  不能将它当作新格式的精确恢复断点。

### 评估、深度集成和数据缓存

```bash
# 使用官方 Solver.eval()，评估单个确定性模型
python ucast.py mode=eval \
  INFER.pretrained_model_path=./outputs_ucast/stage1/checkpoints/best.pdparams

# 多 checkpoint 深度集成；ensemble_size 是总成员数，按模型均分余数
python ucast.py \
  'INFER.pretrained_model_paths=[/path/model1.pdparams,/path/model2.pdparams]' \
  INFER.ensemble_size=7 INFER.ensemble_chunk_size=1 \
  INFER.output_path=./outputs_ucast/forecast.nc \
  INFER.score_output_path=./outputs_ucast/scores.json
```

`DATA.cache_dir=/path/to/cache` 启用 PR 原有的原子 ERA5 只读缓存。
NetCDF 预报采用显式 CF 小时编码；读取时使用
`xr.open_dataset(path, decode_timedelta=True)`。


## 实现说明

- 张量空间维度为 `(longitude, latitude) = (240, 121)`。
- 经度轴采用 circular padding，纬度轴采用 zero padding。
- 每一步先把网络输出从 residual-normalized 空间转换回 normalized 空间，再加到
  最近一帧状态上。
- 推理时仅 Dropout 层处于训练状态，其余层保持评估状态。
- 原生模型的集成输出形状为 `(member, batch, channel, lon, lat)`；
  Solver 任务将 batch 放在首维，便于官方 DataParallel/验证器收集。
  任务内部只扩展一次成员，保留 PR 的混合精度注意力和重计算实现。
- 数据读取会先裁剪到所需变量和时间，再执行 `.load()`，避免原独立脚本将无关变量
  一并拉取到内存。

## 来源与许可注意事项

U-Cast 仓库整体声明 Apache-2.0，但其
`src/models/networks_edm.py` 文件保留了 NVIDIA 的 CC BY-NC-SA 4.0 头。这里的网络
结构是面向 PaddlePaddle 的重新实现；若准备提交到 PaddleScience 上游或用于商业
发布，仍应先由项目维护者确认该部分的许可兼容性。
