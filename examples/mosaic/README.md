# Mosaic

[Mosaic](https://github.com/maxxxzdn/mosaic) 使用 HEALPix U-Net 和局部、压缩、
选块注意力进行全球概率天气预报。本实现注册为 `ppsci.arch.Mosaic`，
包含 GQA、RoPE、网格插值和条件噪声，支持 Paddle 格式权重推理及 LoRA 微调。

## 安装

先安装 PaddleScience 和适合本机 CUDA 的 Paddle，再在仓库根目录执行：

```bash
python -m pip install -r examples/mosaic/requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
```

## 数据和权重

ERA5 模型使用两帧历史、82 个变量，输入网格为 240×121，单步预报时效为 24 小时。
Paddle 权重目录应包含 `config.json` 和 `model.pdparams`；
后者同时保存模型参数及几何缓冲区。下文以 `data/mosaic/model/` 为例。

从[官方资源库](https://huggingface.co/AMLab-UvA/mosaic/tree/main/data)下载静态场和归一化统计：

```bash
mkdir -p data/mosaic
curl -L https://huggingface.co/AMLab-UvA/mosaic/resolve/main/data/static_vars.npz \
  -o data/mosaic/static_vars.npz
curl -L https://huggingface.co/AMLab-UvA/mosaic/resolve/main/data/norm_stats_era5.npz \
  -o data/mosaic/norm_stats_era5.npz

python examples/mosaic/prepare_data.py --samples 3 \
  --init-time 2020-01-02T00:00 --output data/mosaic/era5_samples.npz
```

数据脚本按日期从公开 WeatherBench2 ERA5 中读取历史场和预报目标，
输出物理单位的数据、经纬度和时间条件，无需下载完整数据集。

## 推理

```bash
python examples/mosaic/infer.py --model data/mosaic/model \
  --static data/mosaic/static_vars.npz --stats data/mosaic/norm_stats_era5.npz \
  --data data/mosaic/era5_samples.npz --device gpu:0 \
  --output output/mosaic
```

输出 `prediction.npz` 包含归一化结果、反归一化预报和经纬度；
`inference.json` 记录逐变量的纬度加权 RMSE。增加 `--rollout-steps 3`
可运行 72 小时滚动预报，期间只将模型输出作为后续历史场。

## LoRA 微调

```bash
python examples/mosaic/train_smoke.py --model data/mosaic/model \
  --static data/mosaic/static_vars.npz --stats data/mosaic/norm_stats_era5.npz \
  --data data/mosaic/era5_samples.npz --output output/mosaic_lora \
  --device gpu:0 --sample-index 1 --steps 3 \
  --lora-rank 8 --lora-alpha 16 --learning-rate 0.001
```

此入口在固定样本上检查梯度、参数更新和适配权重保存；支持 MSE、MAE 和
fair-CRPS（`--loss crps --members 2`）。加载适配权重时，向推理命令增加
`--adapter output/mosaic_lora`。适配权重需与同一基座配套使用。

## 参考结果

ERA5 基座有 214,184,448 个参数。rank=8、alpha=16 的 LoRA 适配有
1,837,056 个可训练参数。下表是相同输入和随机种子下，与原版参考在有限日期上的
24 小时预报对比；指标为各日期、各变量中最大的 RMSE 相对增幅。

| 初始日期 | 最大 RMSE 相对增幅 |
| --- | ---: |
| 2020-01-02 / 03 / 04 | 0.5462% |
| 2020-04-02 | 1.5526% |
| 2020-07-02 | 1.0420% |
| 2020-10-02 | 0.0179% |

## 测试

```bash
PYTHONPATH=. python -m unittest discover -s test/arch -p 'test_mosaic*.py' -v
```

论文：*(Sparse) Attention to the Details: Preserving Spectral Fidelity in
ML-based Weather Forecasting Models*, ICML 2026。原始代码采用 Apache-2.0 许可。
