# Aurora

## 1. 模型与实现范围

[Microsoft Aurora](https://github.com/microsoft/aurora) 是地球系统预测基础模型。
本例使用原生 Paddle 实现变量 Patch Embedding、Perceiver 编解码、3D Swin U-Net
及自回归预报，通过 `AuroraArch` 接入 PaddleScience 字典接口和 `Solver`。
参考源码固定为 `85dd3880683f0ac19aa8fe4386c5dcacfec99892`，保留上游 MIT 许可证。

完整权重验证覆盖 **AuroraPretrained，1,256,300,176 参数**。
其他变体保留结构实现，但未验证各自完整预训练权重。
以下命令均在 PaddleScience 仓库根目录执行。

## 2. 安装与离线示例

已验证环境为 Python 3.10、PaddlePaddle GPU 3.4.0、CUDA 12.9、RTX 3090。
按飞桨官方说明安装 PaddlePaddle 后执行：

```sh
python -m pip install -e .
python -m pip install -r examples/aurora/requirements.txt
python examples/aurora/prepare_demo.py
python examples/aurora/finetune.py --config-name tiny
python examples/aurora/finetune.py --config-name tiny mode=eval \
  MODEL.checkpoint=outputs/aurora-tiny/model.pdparams
python examples/aurora/predict.py --tiny --output outputs/aurora-demo
```

Tiny 使用缩小架构和合成数据，检查前向、反向及权重重载。
其结果不用于衡量气象预报能力，Tiny 权重也不能加载到完整模型。

## 3. 权重与数据

使用原生 `aurora-pretrained.pdparams`，大小为 5,025,272,922 字节，约 4.68 GiB。
SHA256 为 `eb586812d66fea058522700165b2858540bb8b15144390f9255fae434cd44d81`。
权重发布于[星河社区模型页](https://aistudio.baidu.com/modelsdetail/55897)。
下载[原生 Paddle 权重](https://git.aistudio.baidu.com/api/v1/repos/yubb/aurora-paddle-pretrained/media/aurora-pretrained.pdparams)，
保存为 `checkpoints/aurora-pretrained.pdparams` 后运行下列命令。
模型直接加载原生 Paddle 参数。

本例使用公开 WeatherBench2 ERA5 的 0.25°、13 气压层子集。
源数据截至 2023-01-10 18:00 UTC。`--start` 按 UTC 解释，下载两帧历史与
`--steps` 帧未来真值，相邻帧间隔 6 小时。

```sh
python examples/aurora/download_era5.py --start 2023-01-01 --steps 2 \
  --directory data/aurora
python examples/aurora/prepare_era5.py --directory data/aurora
```

下载器校验缓存，记录 SHA256 和源对象 ETag。需为源数据块及 NetCDF 文件预留空间。
默认使用代理环境配置，`--direct` 使用直连，`--request-bytes 4194304` 启用分段请求。

地表变量为 `2t, 10u, 10v, msl`，大气变量为 `z, u, v, t, q`，
静态变量为 `lsm, z, slt`。气压层单位为 hPa，共
`50,100,150,200,250,300,400,500,600,700,850,925,1000`。
地表形状为 `(B,T,H,W)`，大气为 `(B,T,C,H,W)`，静态场为 `(H,W)`；
输入 `T=2`，真值 `T=1`。纬度递减，经度在 `[0,360)` 内，
元数据时间为最后一帧历史的 UTC 时间。模型裁剪不完整纬度 Patch，常见 `721 -> 720`；
评估严格对齐时间与空间坐标。

## 4. 推理与效果展示

```sh
CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 python examples/aurora/predict.py \
  --device gpu:0 --steps 2 --checkpoint checkpoints/aurora-pretrained.pdparams \
  --input data/aurora/input.nc --output outputs/aurora-era5
python examples/aurora/evaluate_era5.py --data data/aurora \
  --prediction outputs/aurora-era5 --steps 2 --output outputs/aurora-era5/metrics
python examples/aurora/plot_era5.py --data data/aurora \
  --prediction outputs/aurora-era5 --steps 2 --output outputs/aurora-era5/figures
```

FP32 注意力窗口和解码器位置独立分批处理，解码批量默认为 2048。
完整全球推理此前测得峰值分配显存为 17.83 GiB；全球全分辨率训练尚未验证。

下表为 2023-01-01 06:00 UTC 起报、720×1440 预报网格、13 气压层的结果。
指标采用纬度余弦加权 RMSE，保留物理单位；持续性基线将最后一帧输入保持不变。

| 变量 | 单位 | +6h RMSE | +6h 持续性 | +12h RMSE | +12h 持续性 |
| --- | --- | --- | --- | --- | --- |
| 2 m 温度 | K | 0.491621 | 2.82432 | 0.612136 | 3.25386 |
| 10 m 纬向风 | m/s | 0.523374 | 1.95561 | 0.636562 | 2.77033 |
| 10 m 经向风 | m/s | 0.533163 | 2.31386 | 0.655754 | 3.38663 |
| 海平面气压 | Pa | 31.9165 | 264.613 | 38.4941 | 358.009 |
| 850 hPa 温度 | K | 0.432024 | 1.49239 | 0.485647 | 2.07166 |
| 500 hPa 位势 | m²/s² | 23.9960 | 222.807 | 29.2050 | 338.332 |

[![2 m 温度：ERA5、Paddle 预报与误差](https://paddle-org.bj.bcebos.com/paddlescience/docs/aurora/temperature.png)](https://paddle-org.bj.bcebos.com/paddlescience/docs/aurora/temperature.png)

[![500 hPa 位势高度：ERA5、Paddle 预报与误差](https://paddle-org.bj.bcebos.com/paddlescience/docs/aurora/z500.png)](https://paddle-org.bj.bcebos.com/paddlescience/docs/aurora/z500.png)

两行分别为 +6h、+12h，三列分别为 ERA5、Paddle 和 Paddle−ERA5。
温度图使用摄氏度；位势图除以 9.80665 m/s² 后显示位势高度，单位为米，表格仍使用位势原单位。
气象场色标取真值的 1%–99% 分位，误差色标取绝对误差的 99% 分位，极值可能饱和。
上述结果仅代表单一起报案例，不代表统计基准或论文完整预报技巧复现；未计算缺少气候态的 ACC。

## 5. 监督微调

```sh
CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 python examples/aurora/finetune.py \
  device=gpu:0 MODEL.checkpoint=checkpoints/aurora-pretrained.pdparams
python examples/aurora/finetune.py mode=eval \
  MODEL.checkpoint=outputs/aurora-finetune/model.pdparams
```

`conf/aurora.yaml` 通过 `ppsci.arch.build_model` 构建 `AuroraArch`，连接
`NamedArrayDataset → SupervisedConstraint → Adam → Solver`，并使用 `SupervisedValidator`。
损失为各变量误差除以官方变量/气压层尺度后的 MSE，再对九个变量取平均。
默认 `DATA.spatial_stride=45`，保留全部气压层，在有限网格上演示训练。
输入与真值的时间、气压层和坐标必须一致。

示例训练并评估同一对样本，仅验证工作流，不代表独立验证集精度或训练收敛。
随机初始化需设置 `TRAIN.allow_random_init=true`，评估必须提供权重。
`model.pdparams` 保存原生模型参数，Solver 另存训练状态。

## 6. CINN 编译验证

入口脚本在导入 Paddle 前启用 CINN，并对 Swin3D 骨干调用
`paddle.jit.to_static(backend="CINN", full_graph=False)`。
编解码器、元数据和 NetCDF 处理仍使用动态图，不支持的算子可由原生 Paddle 内核执行。
这是子图编译，不代表整模型全图编译。

```sh
CUDA_VISIBLE_DEVICES=0 bash examples/aurora/run_cinn.sh --tiny \
  --input data/aurora-demo/input.nc --target data/aurora-demo/truth_001.nc \
  --output outputs/aurora-cinn-tiny
CUDA_VISIBLE_DEVICES=0 bash examples/aurora/run_cinn.sh \
  --checkpoint checkpoints/aurora-pretrained.pdparams --input data/aurora/input.nc \
  --train-steps 0 --output outputs/aurora-cinn-full
```

可设置 `PYTHON` 指定当前环境的解释器。完整模型默认采样 17×32 网格，保留全部 13 层，
只检查推理；Tiny 另执行两次 SGD 更新并检查梯度、损失有限。
输出 `cinn.json` 和 `execution.paddle_trace.json`。通过条件包含 profiler 中实际执行的
`cinn_jit` 算子，以及逐变量相对动态图的标准化最大绝对误差 ≤1e-4、相对 L2 ≤1e-5。

| 配置 | 训练更新 | 最大标准化绝对误差 | 最大相对 L2 | 记录的 CINN 算子执行次数 |
| --- | --- | --- | --- | --- |
| Tiny，17×32 合成网格，4 层 | 2 | 2.84e-6 | 9.05e-7 | 16 |
| 完整预训练模型，17×32 ERA5 网格，13 层 | 0 | 1.26e-5 | 2.87e-6 | 6 |

次数仅对应被记录的执行阶段，不代表生成内核数量或编译覆盖率。
两项均在上述环境通过；尚未验收整模型全图编译、全球全分辨率 CINN 训练或性能加速。
