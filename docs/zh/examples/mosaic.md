# Mosaic 全球概率天气预报

[Mosaic](https://github.com/maxxxzdn/mosaic) 使用 HEALPix U-Net 和局部、压缩、
选块注意力进行全球概率天气预报。本实现注册为 `ppsci.arch.Mosaic`，
提供 GQA、RoPE、网格插值、条件噪声及 LoRA 微调。

## 安装和数据

先安装 PaddleScience 和适合本机 CUDA 的 Paddle，再执行：

```bash
python -m pip install -r examples/mosaic/requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
```

ERA5 模型使用两帧历史和 82 个变量，网格为 240×121，单步预报为 24 小时。
Paddle 权重目录包含 `config.json` 和 `model.pdparams`，后者同时保存几何缓冲区。
静态场和归一化统计来自
[官方资源库](https://huggingface.co/AMLab-UvA/mosaic/tree/main/data)。
数据脚本从公开 WeatherBench2 ERA5 中按日期读取样本：

```bash
python examples/mosaic/prepare_data.py --samples 3 \
  --init-time 2020-01-02T00:00 --output data/mosaic/era5_samples.npz
```

## 推理

将 Paddle 权重放入 `data/mosaic/model/`，静态场和归一化统计放入
`data/mosaic/`，在仓库根目录运行：

```bash
python examples/mosaic/infer.py --model data/mosaic/model \
  --static data/mosaic/static_vars.npz --stats data/mosaic/norm_stats_era5.npz \
  --data data/mosaic/era5_samples.npz --device gpu:0 --output output/mosaic
```

输出包含物理单位及归一化的预报、经纬度和逐变量纬度加权 RMSE。
增加 `--rollout-steps 3` 可运行 72 小时滚动预报。

权重目录、数据下载、LoRA 训练检查及参考结果见
[案例 README](https://github.com/PaddlePaddle/PaddleScience/blob/develop/examples/mosaic/README.md)。
ERA5 基座有 214,184,448 个参数，rank-8 LoRA 增加 1,837,056 个可训练参数。
训练入口用于固定样本的参数更新检查，支持加权 MSE、MAE 和 fair-CRPS。

论文：*(Sparse) Attention to the Details: Preserving Spectral Fidelity in
ML-based Weather Forecasting Models*, ICML 2026。原始代码采用 Apache-2.0 许可。
