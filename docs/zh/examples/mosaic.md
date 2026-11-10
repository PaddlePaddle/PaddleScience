# Mosaic 全球概率天气预报

[Mosaic](https://github.com/maxxxzdn/mosaic) 使用 HEALPix U-Net 和稀疏注意力
建模全球概率天气预报。本实现提供网格插值、局部/压缩/选块注意力、GQA、RoPE
和条件噪声，并注册为 `ppsci.arch.Mosaic`。

## 安装和数据

先安装 PaddleScience 和适合本机 CUDA 的 Paddle，再安装示例依赖：

```bash
python -m pip install -r examples/mosaic/requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
```

官方 ERA5 权重及静态场来自 [AMLab-UvA/mosaic](https://huggingface.co/AMLab-UvA/mosaic)，
真实样本由 `prepare_data.py` 从公开 WeatherBench2 数据中按时间片提取。
ERA5 输入包含两帧历史、82个变量，网格为240×121，单步预报为24小时。
模型输出为归一化预报，评估前需使用官方统计反归一化。

权重下载、校验、转换和保留原版几何缓冲区的完整步骤见
[复现说明](https://github.com/PaddlePaddle/PaddleScience/blob/develop/examples/mosaic/VALIDATION.md)。

## 推理和训练检查

在仓库根目录执行，将路径替换为转换后的权重和提取的数据：

```bash
python examples/mosaic/infer.py --model /path/to/paddle-era5-source-geometry \
  --static /path/to/static_vars.npz --stats /path/to/norm_stats_era5.npz \
  --data /path/to/era5_three_samples.npz --device gpu:0 \
  --output /path/to/predictions
```

支持 LoRA、变量/纬度加权 MSE/MAE/fair-CRPS 和集合训练诊断。
`train_smoke.py` 是固定样本可训练性检查，不是正式多样本训练器；
完整分辨率的 LoRA 路线已验证，全参数反向在24GB设备上尚未通过。
详细命令见[案例 README](https://github.com/PaddlePaddle/PaddleScience/blob/develop/examples/mosaic/README.md)。

## 验证范围

官方权重包含214,184,448个基座参数，131组参数全部转换。
三个冬季日期的24小时预报对原版参考最差逐变量 RMSE 退化为0.5462%；
春、夏、秋各一个日期为1.5526%、1.0420%、0.0179%。
原版参考使用 Triton 稀疏注意力及 Torch SDPA，未覆盖完整 FlashAttention 路径。

RTX3090 的完整模型 CINN 测试吞吐提升约23%，三个冬季输入的最差天气 RMSE
退化为0.001085%；严格逐点绝对误差1e-4检查仍失败。
1%天气退化线和逐点阈值均为内部诊断选择，不是已确认的甲方验收要求。
不据此宣称全年论文复现、集合概率校准或逐点一致性。

单元测试可在CPU执行：

```bash
PYTHONPATH=. python -m unittest discover -s test/arch -p 'test_mosaic*.py' -v
```
