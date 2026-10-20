# Aurora

## 1. 背景与实现范围

[Microsoft Aurora](https://github.com/microsoft/aurora) 是面向地球系统预测的基础模型。
本案例将其移植到原生 Paddle，并通过 `AuroraArch` 接入 PaddleScience 的字典接口和 `Solver`。
参考源码固定为 `85dd3880683f0ac19aa8fe4386c5dcacfec99892`，保留原 MIT 许可证。

模型使用变量 Patch Embedding 和 Perceiver 编码气象场，通过 3D Swin U-Net
建模空间及垂直层之间的关系，再用 Perceiver 解码到指定气压层和变量。
自回归预测把前一步预报作为后一步历史输入。

完整权重验收以 **AuroraPretrained** 为主，参数量为 **1,256,300,176**。
Small 用于调试和权重对齐；12h、HighRes、AirPollution、Wave、1.5 及集合变体
仅完成轻量结构检查，未验收各自完整预训练权重。

## 2. 安装与离线运行

已验证环境为 Python 3.10、PaddlePaddle GPU 3.4.0、CUDA 12.9、RTX 3090。
先按飞桨官方说明安装 PaddlePaddle，再从 PaddleScience 仓库根目录执行：

```sh
python -m pip install -e .
python -m pip install -r examples/aurora/requirements.txt
python examples/aurora/prepare_demo.py
python examples/aurora/finetune.py --config-name tiny
python examples/aurora/finetune.py --config-name tiny mode=eval \
  MODEL.checkpoint=outputs/aurora-tiny/model.pdparams
```

上述离线流程使用合成数据和缩小架构，用于检查训练、评估与权重保存重载。
Tiny 权重不能加载到正式完整模型，也不能用于衡量真实预报能力。
原生 `.pdparams` 推理和训练不依赖 PyTorch；权重导出在独立 Torch 环境执行。

## 3. 数据与权重准备

本例下载公开 WeatherBench2 ERA5 的 0.25°、13 气压层子集。
源数据时间截至 2023-01-10 18:00 UTC。起报输入包含两个历史时刻；
`--steps` 指定未来真值帧数，每步间隔 6 小时。

```sh
python examples/aurora/download_era5.py --start 2023-01-01 --steps 2 \
  --directory data/aurora
python examples/aurora/prepare_era5.py --directory data/aurora
python examples/aurora/download_checkpoint.py --directory checkpoints
```

下载脚本记录 SHA256、源对象 ETag 及输出清单，支持校验后复用缓存。
完整权重约 4.68 GiB，另需预留压缩数据、原始 NetCDF、转换数据及权重的磁盘空间。
将 `TORCH_PYTHON` 替换成独立 Torch 环境的 Python 可执行文件：

```sh
TORCH_PYTHON examples/aurora/export_torch_checkpoint.py \
  checkpoints/aurora-0.25-pretrained.ckpt checkpoints/aurora-torch.npz
python examples/aurora/convert_checkpoint.py --input checkpoints/aurora-torch.npz \
  --output checkpoints/aurora-pretrained.pdparams
```

地表变量为 `2t, 10u, 10v, msl`，大气变量为 `z, u, v, t, q`，
静态变量为 `lsm, z, slt`。地表输入形状为 `(B,T,H,W)`，大气为 `(B,T,C,H,W)`，
静态场为 `(H,W)`；输入历史长度为 2，真值长度为 1。
气压层单位为 hPa，经度在 `[0,360)` 内，纬度递减；元数据时间表示最后一帧历史的 UTC 时间。
模型裁剪不能组成完整 Patch 的纬度行，常见 `721 -> 720`；指标计算严格对齐时间和空间坐标。

## 4. 推理、训练与评估

完整模型全球推理：

```sh
CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 python examples/aurora/run_era5.py \
  --framework paddle --device gpu:0 --steps 2 \
  --checkpoint checkpoints/aurora-pretrained.pdparams \
  --input data/aurora/input.nc --output outputs/aurora-era5/paddle
```

FP32 推理对独立注意力窗口和解码器网格位置分批，默认解码批量为 2048。
已测峰值分配显存为 17.83 GiB；训练显存需求更大，尚未验证全球全分辨率训练。

基于真实 ERA5 一对输入/真值的监督微调：

```sh
CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 python examples/aurora/finetune.py \
  device=gpu:0 MODEL.checkpoint=checkpoints/aurora-pretrained.pdparams
python examples/aurora/finetune.py mode=eval \
  MODEL.checkpoint=outputs/aurora-finetune/model.pdparams
```

`conf/aurora.yaml` 连接 `NamedArrayDataset → SupervisedConstraint → Solver`，
使用 Adam 及 `SupervisedValidator`。损失为各变量误差除以官方变量/气压层尺度后的 MSE，
再对九个变量取平均。默认空间采样步长为 45，保留全部 13 气压层。
该示例重复训练并评估同一对样本，用于验证完整工作流，不代表独立验证集精度或训练收敛。
原生 `model.pdparams` 可直接用于推理，Solver 还会独立保存训练状态。

Torch 与 Paddle 预测分别在独立环境生成后，用 `evaluate_era5.py` 计算纬度余弦加权 RMSE。
每个变量、层和时效分别比较，不能以平均指标掩盖单项失败。
完整命令见仓库 `examples/aurora/README.md`。

## 5. 验收结果与限制

| 检查 | 最大观测误差 | 项目阈值 |
| --- | --- | --- |
| 同输入前向：标准化最大绝对误差 | 4.21e-5 | 1e-4 |
| 同输入前向：变量相对 L2 | 7.88e-7 | 1e-5 |
| 全部梯度整体相对 L2 | 2.31e-5 | 1e-4 |
| 更新后参数整体相对 L2 | 9.35e-10 | 1e-5 |
| 4,554 组天气 RMSE 的最大相对差异 | 0.000943% | 每组 1% |

这些是本项目约定的迁移阈值，不是 PaddleScience 通用门槛。
训练对齐覆盖完整预训练参数、17×32 空间采样、13 层和连续 3 次 SGD 更新。
预报覆盖三个季节日期、最长 240 小时，另含原有两步案例。
独立长时滚动逐点容差 `rtol=atol=2e-4` 在部分时效仍有超差；
天气指标通过不代表全部格点逐点通过。实验性 FP32 融合注意力优化不包含在默认实现中。
未计算缺少气候态的 ACC，也未声称论文完整训练或统计预报技巧复现。
机器可读记录位于 `examples/aurora/validation/acceptance.json`。

```sh
python -m pytest test/arch/test_aurora.py -q
CUDA_VISIBLE_DEVICES=0 AURORA_TEST_DEVICE=gpu:0 NVIDIA_TF32_OVERRIDE=0 \
  python -m pytest test/arch/test_aurora.py -q
```
