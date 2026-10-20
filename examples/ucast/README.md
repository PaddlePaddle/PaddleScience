# U-Cast

本目录提供 U-Cast 的 PaddleScience 迁移实现，包括：

- 与原始 Dhariwal/ADM U-Net 对齐的 `ppsci.arch.UCast`；
- PyTorch Lightning EMA 权重到 Paddle `.pdparams` 的转换工具；
- WeatherBench-2 ERA5 数据读取、归一化、周期边界卷积；
- MC-Dropout 集成、自回归预报、fair CRPS 和 RMSE；
- ERA5 单步训练数据集、WMAE 第一阶段和 fair-CRPS 第二阶段；
- PaddleScience Solver、Muon+AdamW、梯度累积、warmup+cosine 和全局梯度裁剪；
- 与原版一致的逐 batch EMA、多步自回归验证和完整训练状态恢复；
- NetCDF 预报结果导出。

当前入口覆盖发布权重推理、第一阶段确定性训练和第二阶段概率微调。

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
  TRAIN.pretrained_model_path=./outputs_ucast/stage1/checkpoints/latest
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
`TRAIN.checkpoint_path` 恢复同一训练任务，同时加载 Muon 动量、AdamW 状态、
学习率位置、AMP scaler、EMA 权重及更新次数，以及 Python / NumPy / Paddle
CPU / CUDA 随机数状态。路径可以带 `.pdparams` 后缀。

```bash
CUDA_VISIBLE_DEVICES=6 conda run --no-capture-output -n paddlescience \
  python ucast.py --config-name ucast_3090 \
  hydra.run.dir=./outputs_ucast/stage1 seed=11 \
  DATA.data_dir=/path/to/era5 \
  TRAIN.checkpoint_path=./outputs_ucast/stage1/checkpoints/latest
```

恢复时保持原模型、优化器、总 epochs、训练数据及 batch/累积配置一致。
先试跑一个 epoch 时可保持总 epochs 不变并设置 `TRAIN.stop_after_epoch=1`，
恢复时将其设回 `null`。checkpoint 保存于 epoch 边界；每组包含 `.pdparams`、
`.pdopt`、`.pdstates`、`.pdtrain`，AMP 对应 `.pdscaler`，EMA 对应 `_ema.pdparams`。
`best_model_ema.pdparams` 可以直接用于推理。

## 实现说明

- 张量空间维度为 `(longitude, latitude) = (240, 121)`。
- 经度轴采用 circular padding，纬度轴采用 zero padding。
- 每一步先把网络输出从 residual-normalized 空间转换回 normalized 空间，再加到
  最近一帧状态上。
- 推理时仅 Dropout 层处于训练状态，其余层保持评估状态。
- 概率训练把 ensemble 维放在输出首维，形状为 `(member, batch, channel, lon, lat)`。
- 数据读取会先裁剪到所需变量和时间，再执行 `.load()`，避免原独立脚本将无关变量
  一并拉取到内存。

## 来源与许可注意事项

U-Cast 仓库整体声明 Apache-2.0，但其
`src/models/networks_edm.py` 文件保留了 NVIDIA 的 CC BY-NC-SA 4.0 头。这里的网络
结构是面向 PaddlePaddle 的重新实现；若准备提交到 PaddleScience 上游或用于商业
发布，仍应先由项目维护者确认该部分的许可兼容性。
