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

## 6. CINN 训练与性能验证

`run_cinn.sh` 提供 `--mode inference`、`--mode train` 和 `--mode benchmark`。
默认进入直接训练：先设置训练模式，再执行前向、反向与 SGD 更新，不需要先做推理。
`--train-steps` 接受任意正整数，包括单步训练。

动态图和 CINN 分别在独立进程中运行，使用相同随机种子、原生权重、输入、FP32 精度和
SGD 配置。两种后端均在导入 Paddle 前设置 `FLAGS_new_executor_serial_run=true`，
固定 PIR 执行器的调度方式，报告中记录实际开关值。CINN 进程另设置编译开关，
对 Swin 骨干的各个编码/解码 stage 调用
`paddle.jit.to_static(backend="CINN", full_graph=True)`。动态图进程关闭 CINN 和组合算子。
特征展开、时间条件、模型编解码器及 Batch 处理仍使用动态图。每个 stage 完整捕图后，
由 CINN 编译支持的子图；不代表整模型全图编译。注意力掩码的分区和经度连通关系
只依赖网格配置，使用 NumPy 生成常量，避免在每次执行时重复切片赋值和填充。
动态图缓存掩码张量，静态程序各自持有常量，避免复用其他程序的中间值。

```sh
CUDA_VISIBLE_DEVICES=0 bash examples/aurora/run_cinn.sh --mode train --tiny \
  --input data/aurora-demo/input.nc --target data/aurora-demo/truth_001.nc \
  --train-steps 2 --output outputs/aurora-cinn-tiny
CUDA_VISIBLE_DEVICES=0 bash examples/aurora/run_cinn.sh --mode train \
  --checkpoint checkpoints/aurora-pretrained.pdparams --input data/aurora/input.nc \
  --target data/aurora/truth_001.nc --train-steps 2 --output outputs/aurora-cinn-train
CUDA_VISIBLE_DEVICES=0 bash examples/aurora/run_cinn.sh --mode inference \
  --checkpoint checkpoints/aurora-pretrained.pdparams --input data/aurora/input.nc \
  --output outputs/aurora-cinn-inference
```

可设置 `PYTHON` 指定解释器。完整模型默认 `--spatial-stride 45`，得到 17×32 网格，保留
全部 13 个气压层；可根据显存调小采样步长。`--backend eager` 或 `--backend cinn` 可独立运行
某一后端；省略时自动运行并比较两者。

输出根目录的 `cinn.json` 汇总两种后端的结果。子目录保存运行日志、预测和 `result.json`，
CINN 子目录另存 `execution.paddle_trace.json`。推理和训练均检查实际 `cinn_jit` 执行，
逐变量标准化最大绝对误差 ≤1e-4、相对 L2 ≤1e-5。训练另检查全部已有梯度有限、参数确有更新，
各步 loss 满足 `rtol=1e-5, atol=1e-6`；每个参数最多抽样 64 个位置比较首步梯度和最终参数，
抽样向量相对 L2 ≤1e-4。抽样检查不代表全量梯度逐点对齐。

性能测试使用已驻留显存的相同输入，分别测量推理和包含前向、loss、反向、SGD、清梯度的训练步：

```sh
CUDA_VISIBLE_DEVICES=0 bash examples/aurora/run_cinn.sh --mode benchmark \
  --benchmark-task train --checkpoint checkpoints/aurora-pretrained.pdparams \
  --input data/aurora/input.nc --target data/aurora/truth_001.nc \
  --spatial-stride 20 --warmup 10 --iterations 50 --repeat 3 \
  --output outputs/aurora-benchmark-train
CUDA_VISIBLE_DEVICES=0 bash examples/aurora/run_cinn.sh --mode benchmark \
  --benchmark-task inference --checkpoint checkpoints/aurora-pretrained.pdparams \
  --input data/aurora/input.nc --spatial-stride 20 \
  --warmup 10 --iterations 50 --repeat 3 --output outputs/aurora-benchmark-inference
```

性能模式的训练预热中，最初 `--train-steps` 步（不超过预热步数）同时检查 loss 和梯度；
计时结束后另做一次不计时的诊断更新。训练的首步耗时包含这些检查。
首步耗时单独记录，包含按需捕获、编译和执行，可能命中编译缓存；首次编译可能需要数分钟。
稳态计时在预热后进行，
每步通过 GPU 同步界定边界，排除编译、检查、profiler、NumPy 拷贝和文件读写。
每个后端使用独立进程，预热后连续测量 `--repeat` 轮；训练在轮次间继续更新参数，
两种后端使用相同更新步数。报告包含逐步耗时、均值、中位数、P95、
步吞吐和本进程峰值分配显存；加速比为各轮中位数的中位数之比 `eager / CINN`，大于 1 表示提速。
性能测试通过表示数值检查通过，不要求加速比大于 1。请在无其他计算负载的 GPU 上复测。

采样网格的结果不代表全球全分辨率训练性能或整模型全图编译能力。

### 采样网格测量结果

以下为 RTX 3090 独占卡、PaddlePaddle 3.4.0 / CUDA 12.9、FP32、batch size 1 的测量，
两种后端均设置 `FLAGS_new_executor_serial_run=true`。
使用完整的 1,256,300,176 参数原生权重和 ERA5 全部 13 个气压层，
`--spatial-stride 20` 得到 37×72 输入，模型裁剪后为 36×72。
每个后端预热 10 步，连续测量 3 轮、每轮 50 步；训练使用 SGD，学习率 1e-6。

| 任务 | 动态图稳态耗时 (ms/步) | CINN 稳态耗时 (ms/步) | 加速比 | 动态图 / CINN 峰值分配显存 (GiB) |
| --- | ---: | ---: | ---: | ---: |
| 训练 | 186.54 | 131.96 | 1.414× | 10.38 / 11.43 |
| 推理 | 76.30 | 48.03 | 1.588× | 4.74 / 4.81 |

训练累计执行 161 次参数更新，首两次更新前无推理调用。
最大标准化绝对误差为 1.99e-5、最大相对 L2 为 2.59e-6，
首步抽样梯度相对 L2 为 1.26e-5，最终抽样参数相对 L2 为 7.50e-11。
推理最大标准化绝对误差为 1.52e-5、最大相对 L2 为 3.52e-6。
训练与推理的诊断 trace 分别记录 1,113 和 466 次 CINN 算子执行。
Tiny 直接训练的两步 loss 与动态图完全一致，抽样梯度相对 L2 为 5.93e-8，
最大标准化绝对误差为 2.84e-6、最大相对 L2 为 9.32e-7。

CINN 首次训练步为 347.36 秒，包含捕获、编译、执行与梯度检查；
动态图为 5.23 秒。CINN 首次推理步为 199.55 秒，动态图为 0.57 秒。
稳态提速伴随更高的显存使用，短任务总耗时可能更长。
这些结果仅对应上述环境和采样网格，不能外推为全球全分辨率的加速保证。

此前多线程执行器配置曾在 Paddle profiler 收尾时出现
`HostEventRecorder::GatherEvents` 崩溃或 `vector::reserve` 异常。
本入口固定使用串行 PIR 调度；Tiny 训练、完整训练和完整推理的 trace 导出及数值检查均通过。
切换执行器配置后应重新验证稳定性与性能，进程异常退出不能视为验收通过。
