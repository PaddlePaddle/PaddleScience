# Mosaic

Paddle implementation of [maxxxzdn/mosaic](https://github.com/maxxxzdn/mosaic),
*(Sparse) Attention to the Details: Preserving Spectral Fidelity in ML-based
Weather Forecasting Models*, ICML 2026. Original code: Apache-2.0.

当前实现包含 HEALPix U-Net、网格插值、局部/压缩/top-k 选块注意力、GQA、RoPE、条件噪声与 FFN，并注册为 `ppsci.arch.Mosaic`。原 Torch 自定义内核由可微 Paddle 算子实现，选块计算采用分块 gather，不构造全局 token attention 矩阵。

本贡献范围为模型迁移、官方权重推理、LoRA 训练诊断和评估工具。
当前可供代码评审；不是原论文全年评估或生产训练流程的完整复现。
最新结果、复现环境及限制见 [验证摘要](VALIDATION.md)；下文保留历史诊断以便追溯。
1% 天气 RMSE 退化线和 `atol=1e-4` 逐点检查均是内部诊断条件，
不是甲方确认的验收要求；超线结果如实保留，不作为本次准备提交的阻断项。

## 环境

安装 PaddleScience 基础依赖及本示例依赖，并在仓库根目录设置：

```bash
python-paddle -m pip install -r requirements.txt -r examples/mosaic/requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
```

`python-paddle`、`python-torch` 是文档占位符，执行时替换为各自环境的 Python。
先按飞桨官方安装说明安装适配 CUDA 的 Paddle；本次验证使用的 nightly 版本
见验证摘要，尚未验证所有稳定版。不要把两种框架装进同一个验证环境。

PyTorch 参考与 Paddle 使用分离的环境。不要为了复现而降级其他任务的共享环境。

## 可运行的最小结构对齐

先克隆原始参考仓库，并检出 commit `30c5baef1dc76c5810caf84c449e46c7925b9aec`。以下命令在 PaddleScience 根目录执行，`python-torch`、`python-paddle` 分别代表对应环境的 Python：

```bash
python-torch examples/mosaic/validate_alignment.py \
  --framework torch --source /path/to/mosaic \
  --output /path/to/alignment

python-paddle examples/mosaic/validate_alignment.py \
  --framework paddle --output /path/to/alignment
```

Torch 端安装 `torch`、`einops`、`healpy`、`scikit-learn`；此结构对齐使用独立的数学 attention，不需要安装 FlashAttention。Paddle GPU 验证增加 `--device gpu:0`。

参考程序产生固定合成输入、随机权重、显式噪声与三步 SGD 结果。Paddle 端先比较几何 buffer，再按模块名称转置所有 Linear 权重，比较预测、loss 和第一步全部参数梯度。报告写入 `alignment.json`，超出默认 `1e-4` 诊断阈值时退出失败。这个阈值仅用于本脚本，不代表甲方已确认的气象验收口径。

覆盖其他分支时，对 Torch 命令增加 `--no-compression` 或 `--sparse-count 0 --noise-dim 0`，使用独立输出目录，随后运行对应 Paddle 命令。

## 当前验证范围

CPU 小配置三路 attention、no-compression、无选块/无噪声路径都已通过三步前反向对齐。第一轮三路 attention 的前向差约 `1.8e-7`，参数梯度差约 `7.5e-8`。GPU 同配置前向差约 `1.94e-7`、梯度差约 `8.94e-8`。

上述小配置数据来自合成输入和随机小模型，不代表完整模型验收。

官方 ERA5 checkpoint 已完成 SHA256 校验（`187df31a0caef3e934e4eb8a11506c9fea518ff0e02ce6d1f804fc6c8a78a940`），131/131 组参数转换成功，总参数 214,184,448。在真实 ERA5 2020-01-01/02 两帧输入上，完整模型已于 RTX 3090 跑通 2020-01-03 的一步预测，输出 `[1,1,240,121,82]` 均为有限值。

历史初始实现的逐点诊断（不代表当前候选）：相同输入、权重和显式噪声下，对 Torch 数学稀疏参考的最大归一化输出差 `0.0362911`、平均差 `4.84447e-5`；对原始 Triton 稀疏内核（局部/压缩路径使用 Torch SDPA）的最大差 `5.34369`、平均差 `0.0271917`。原 Triton 内部会转 BF16，逐点一致性与下文的天气任务指标分别记录，不能用小模型结果代替完整模型指标。

后续逐层诊断确认：气象输入完全相同，静态场归一化 buffer 最大差 `0.00106382`，插值方向 buffer 最大差约 `0.000622757`。已将 Paddle 静态场统计改为 FP64 累积，并增加完整 `240×121` 网格回归测试。以下结果来自同一真实输入、官方权重和显式噪声，对照 Torch 数学参考：

| 运行方式 | 最大归一化输出差 | 平均差 | RMSE |
| --- | ---: | ---: | ---: |
| 初始实现，重新生成几何 | 0.0362911 | 4.84447e-5 | 1.54551e-4 |
| FP64 静态场统计，重新生成几何 | 0.0155888 | 3.14834e-5 | 7.77365e-5 |
| 转换时保留原版全部 65 个几何 buffer，独立推理 | 0.00986958 | 1.65025e-5 | 1.00459e-4 |
| 相同几何并强制使用参考路由（仅诊断） | 0.00167274 | 8.26601e-6 | 2.22763e-5 |

保留原版几何后第一层线性投影完全一致，仍有一组 top-k 选块集合不同。其余若干索引差异只是集合内顺序不同。固定路由结果不能作为独立推理验收，所有结果仍仅限单样本。不能断言剩余误差完全由路由造成。

上面的逐点精度为历史诊断，不等于下面的天气任务指标。当前已补有限样本滚动预测和完整模型 CINN 探测，但正式评估、原版内核逐点对齐与 CI 仍未完成。

### LoRA 微调验证（甲方允许的训练路线）

`train_smoke.py` 提供全参数或 LoRA 的真实输入训练检查。LoRA 只适配 Transformer attention 的 Q/V 和 FFN 两个线性层，冻结其余全部权重；默认使用可选的 block、局部/选块注意力和插值重计算降低训练显存。重计算只在训练且开启梯度时生效，不改变原生 checkpoint 的参数名或默认推理路径。

```bash
python-paddle examples/mosaic/train_smoke.py \
  --model /path/to/paddle-era5-source-geometry \
  --static /path/to/static_vars.npz --stats /path/to/norm_stats_era5.npz \
  --data /path/to/era5_three_samples.npz --output /path/to/lora_run \
  --device gpu:0 --sample-index 1 --steps 3 \
  --lora-rank 8 --lora-alpha 16 --learning-rate 0.001

python-paddle examples/mosaic/infer.py \
  --model /path/to/paddle-era5-source-geometry --adapter /path/to/lora_run \
  --static /path/to/static_vars.npz --stats /path/to/norm_stats_era5.npz \
  --data /path/to/era5_three_samples.npz --sample-index 2 \
  --device gpu:0 --output /path/to/lora_prediction
```

完整 214,184,448 参数基座，rank=8/alpha=16 新增 **1,837,056** 个可训练参数（112 个张量，约为基座的 0.858%）；56 个目标层。FP32、RTX3090、完整 240×121 输入，在两个样本上分别从同一基座独立完成三步 SGD。样本0 loss 从 0.102370024 降至 0.102362894；样本1从 0.098202184 降至 0.098195232。所有适配参数梯度存在且有限，适配权重保存/加载的输出逐点相同。样本1还检查了冻结权重前后 SHA256 完全一致。峰值 allocated 显存约 19.1 GiB，单步约 7.5–7.9 秒（含梯度检查，不是标准性能 benchmark）。

报告 `training.json` 区分总参数和可训练参数，记录每一步 loss、全部适配梯度范数、冻结权重哈希和 checkpoint roundtrip；`adapter.pdparams` / `adapter_config.json` 只保存低秩适配参数及目标层清单，必须与原基座配套使用。不要把 LoRA adapter 当作完整模型 checkpoint。`--lora-rank 0` 是全参数路线；此前完整全参数 FP32 反向在 24GB 卡上遇到 OOM，**尚未通过**。

此检查使用原版支持的变量/纬度加权 MSE，固定一个样本和噪声，验证可训练性，不是原论文默认的 ensemble CRPS 训练、训练收敛或正式泛化评估。第一步 A 的梯度可因 B 零初始化而为零，这是标准 LoRA 初始化行为。新增回归覆盖零初始化输出不变、重计算前反向一致、冻结权重不变与 adapter 保存加载。

样本1的 adapter 已通过独立推理进程加载，在未参与该次更新的样本2上完成 24h 预报；对原 Triton sparse / Torch SDPA 基线，82变量最差 RMSE 退化 0.54587%，仍在内部 1% 筛查内。这里只是相邻日期的微调后检查，不是独立跨季节泛化测试，也不能据此宣称 LoRA 提升了预报精度。

### 集合 fair-CRPS 训练与验证入口

训练脚本支持 `--loss mse|mae|crps`、`--members N`。fair-CRPS 采用原版 `losses.py` 的无偏集合估计，排除自身配对，至少需要2个成员；在归一化空间施加相同变量/纬度权重。通过逐对累积避免分配完整的成员数平方空间张量。它与物理单位RMSE不同，也不能用单日期小集合证明概率校准效果。

```bash
python-paddle examples/mosaic/train_smoke.py \
  --model /path/to/paddle-era5-source-geometry \
  --static /path/to/static_vars.npz --stats /path/to/norm_stats_era5.npz \
  --data /path/to/era5_three_samples.npz --output /path/to/new_crps_run \
  --device gpu:0 --sample-index 1 --steps 10 \
  --lora-rank 8 --learning-rate 0.001 --members 2 --loss crps

python-paddle examples/mosaic/evaluate_ensemble.py \
  --model /path/to/paddle-era5-source-geometry --adapter /path/to/new_crps_run \
  --static /path/to/static_vars.npz --stats /path/to/norm_stats_era5.npz \
  --data /path/to/era5_three_samples.npz --sample-index 2 \
  --members 2 --device gpu:0 --output /path/to/new_ensemble_evaluation
```

各成员使用显式独立噪声，逐成员运行单成员模型；训练时外层重计算释放成员图，内部保留已有的block/chunk重计算。入口目前只支持每batch一份历史上下文（N=1），不把多个时间窗混成集合。`--no-recompute` 关闭此省显存功能；大模型可能OOM。训练输出目录必须新建，防止混入旧adapter；随机种子在LoRA初始化前设置，保证初始化可复现。

`evaluate_ensemble.py` 在同样噪声下分别运行基座和adapter，保存完整成员预测、归一化加权fair-CRPS、成员MSE、集合平均预报的物理RMSE与spread。拒绝训练报告中相同数据路径/样本索引的检查；不同文件的重叠日期仍须调用者核对。单测覆盖fair-CRPS与独立全配对公式的数值/梯度，以及逐成员重计算与普通模式的全部参数梯度一致性。固定样本更新仍属训练检查，不是论文训练流程或收敛复现。

2026-10-22 两成员FP32、rank8 LoRA、真实ERA5样本1、固定噪声的十步SGD检查通过：fair-CRPS从0.081527159降至0.081524864；全部112个适配参数张量梯度存在且有限，冻结基座SHA256前后相同，adapter保存加载后所有成员输出逐点相同。峰值allocated21,036,877,568bytes（约19.6GiB），每步含检查约19.1–19.4s。独立样本2集合评估中，adapter相对基座的82变量集合平均RMSE最差退化0.00069105%，加权归一化fair-CRPS变化-8.27e-7。这里只是相邻日期、2成员、十步固定样本检查，不证明统计显著收益或泛化收敛。

### 多日期/跨季节批量评估

对不同日期分别运行 `prepare_data.py --samples 1 --init-time ...`，例如 `2020-04-02T00:00`、`2020-07-02T00:00` 和 `2020-10-02T00:00`，保留其 JSON 数据清单。以下入口每份数据只取样本0，在指定GPU上串行启动隔离的 Torch/Paddle 进程，不将跨季节数据错误地作为连续滚动序列：

```bash
python-paddle examples/mosaic/evaluate_seasons.py \
  --data /path/to/april.npz /path/to/july.npz /path/to/october.npz \
  --model /path/to/paddle-era5-source-geometry \
  --source /path/to/original-mosaic --checkpoint /path/to/era5_best.pt \
  --static /path/to/static_vars.npz --stats /path/to/norm_stats_era5.npz \
  --adapter /path/to/lora_run \
  --paddle-python /path/to/paddle/bin/python \
  --torch-python /path/to/torch/bin/python --gpu 0 \
  --output /path/to/new_seasonal_run
```

`--adapter` 可省略。输出目录必须是新目录，以免误用旧预测。每个日期分别保存原版、Paddle 基座、可选 LoRA 输出及日志；使用相同显式噪声种子2026、FP32、24h一步。逐日期报告保留完整82变量指标；`suite.json` 汇总时先平均各日期 MSE 再开方，同时要求聚合和每个“日期×变量”均满足内部阈值，超限非零退出。单测专门覆盖聚合通过但单日期超限的情况。

评估日期应与LoRA训练日期分离，脚本不自动推断训练数据划分，调用者需检查。原版参考仍为原 Triton sparse + Torch SDPA，不是 FlashAttention 完整基线。少量季节样本不能代替全年评估、集合概率评分或ACC；没有气候态时不报告ACC。

2026-10-22 完成上述三个初始化日期，沿用冬季训练的rank8 adapter，均未参与本轮评估日期更新。以下为82变量中相对原 Triton sparse / SDPA 的最差天气 RMSE 退化：

| 初始化日期 | Paddle 基座 | LoRA | 超过内部1%的变量 |
| --- | ---: | ---: | --- |
| 2020-04-02 | 1.55258% | 1.55067% | geopotential_100 |
| 2020-07-02 | 1.04204% | 1.04089% | vertical_velocity_400 |
| 2020-10-02 | 0.01790% | 0.01580% | 无 |

三日期聚合82变量最差退化为基座0.22827%、LoRA0.22661%；但两者均有2/246个“日期×变量”超限，因此整轮报告是 **`screening_failed`，非零退出**，不能以聚合通过代替逐样本通过。三个日期的基座/LoRA所有变量均优于持续性预报。LoRA相对Paddle基座的最差额外RMSE退化0.01894%，没有修复原始稀疏内核路径差异。

对两个超线样本补跑Torch数学稀疏参考后，Paddle基座逐变量天气RMSE最差退化分别为春季0.00034075%、夏季0.00057557%；最大归一化输出差分别0.00180244/0.00235176。数学参考非常接近，但原始Triton路径下仍超内部线，不能将两类基线混为逐点对齐，也不能断言BF16转换是全部根因。该差异保留为已知限制，不作为本次迁移提交的内部阻断项。

### 天气任务指标筛查

`evaluate_weather.py` 在物理量空间使用 FP64 计算逐变量余弦纬度加权 RMSE，对比原版预测及持续性预报（沿用最后一帧）。内部暂定所有变量 RMSE 相对原版退化不超过 1%，同时报告每个样本的最差退化；不是甲方验收标准，也不等同于统计显著性结论。没有气候态数据时不报告 ACC。

`infer.py` 和 `reference_forecast.py` 默认只运行 `--sample-index 0`，可指定其他样本；报告记录数据路径、索引和噪声种子，评估器拒绝来源/精度不匹配及运行时强制几何/路由的结果。`--amp-dtype float16` 可诊断原版默认 autocast 路径，当前正式候选仍为 FP32 数学 attention。

```bash
python-paddle examples/mosaic/evaluate_weather.py \
  --data /path/to/era5_sample.npz --stats /path/to/norm_stats_era5.npz \
  --actual /path/to/paddle_run --reference /path/to/torch_native_run \
  --sample-index 0 --output /path/to/weather_report.json
```

多个样本分别运行两个推理脚本，再为 `--actual`、`--reference`、`--sample-index` 依次提供等长列表。聚合时先平均各样本 MSE，再开方，不能直接平均 RMSE；相邻日期仍不能代表跨季节正式测试集。

首个真实样本（2020-01-02 初始化，24h 预报），独立 Paddle 数学 attention 与原 Triton sparse / Torch SDPA 对照、均 FP32：82 个变量全部通过上述内部筛查，最差退化为 2 米温度的 0.43974%；其 RMSE 分别为 0.77260 K 和 0.76922 K。所有变量优于持续性预报。此结果只说明这一样本的任务误差未明显退化，不能用来证明原版内核逐点对齐或整体迁移完成。

扩展至 2020-01-02/03/04 三个初始化日期的独立 24h 预报后：82 个变量的聚合 RMSE 最差退化 `0.13484%`，全部 246 个“样本×变量”组合的最差退化 `0.54617%`，均满足内部 1% 筛查线。这里没有自回归滚动，也没有多季节覆盖；不代表原版内核逐点对齐或正式任务验收。

### 短期自回归滚动

两个推理入口均支持 `--rollout-steps 3`（默认 1），需要至少三份连续样本的目标帧。只用第一个样本的两帧历史初始化，后续将模型预测反归一化后滚动写入历史；不读取真实下一帧作为输入。每步噪声由相同 NumPy RNG 顺序生成，时间增量匹配原版 ERA5 `[1, 4/365.25]`。这是 ERA5 24h 专用诊断，不用于 HRES。

```bash
# 分别运行 infer.py 和 reference_forecast.py，均增加 --rollout-steps 3
python-paddle examples/mosaic/evaluate_rollout.py \
  --data /path/to/era5_three_samples.npz --stats /path/to/norm_stats_era5.npz \
  --actual /path/to/paddle_rollout --reference /path/to/torch_rollout \
  --output /path/to/rollout_report.json
```

2020-01-02 单初始化、FP32、独立路由，原版 Triton sparse / SDPA 对照的结果：

| 预报时效 | 82 变量中最差 RMSE 退化 | 2m 温度 Paddle / 原版 RMSE (K) |
| --- | ---: | ---: |
| 24h | 0.43974% | 0.77260 / 0.76922 |
| 48h | 0.89308% | 1.04105 / 1.04592 |
| 72h | 0.79301% | 1.35253 / 1.37131 |

所有变量、所有时效满足内部 1% 筛查线。只验证一个初始化日期、一个噪声种子，仍缺跨季节滚动和更长滚动评估。测试覆盖历史反馈、时间推进、显式噪声序列和原输入不被修改。

持续性基线方面：24h/48h 的 82 变量全部优于持续性预报；72h 有 81/82 优于该基线，例外为 `temperature_50`（Paddle 2.61719 K、原版 2.62429 K、持续性 2.53312 K）。不能将相对原版不退化表述为每个变量均具有预报优势。

## 数据和权重

原模型使用 WeatherBench2 ERA5/HRES 1.5° 数据。官方 HF 权重入口已重定向为 [AMLab-UvA/mosaic](https://huggingface.co/AMLab-UvA/mosaic)，初始验证锁定 revision `c2aaa3e0799e2a0e8fba8c79beb6d3ba8e2a4954`。ERA5 与 HRES 的历史帧数、时间步长、邻居数和归一化统计不同，不能互换。

采样输入字典：`state` 为 `[B,N,history,lon,lat,channels]`，`time` 为 `[B,N,2]` 的日/年进度，可选 `noise` 为 `[B*N*samples,noise_dim]` 的标准正态输入。模型输出是归一化预报，需要按官方统计反归一化后评估。

模型调用前必须用相同经纬度、静态变量分别调用 `initialize_static_vars` 和 `initialize_interpolation`，以建立确定性的几何状态。

## 官方权重与真实输入复现命令

下载 HF 的 `checkpoints/era5_best.pt`、`data/static_vars.npz`、`data/norm_stats_era5.npz` 后执行：

```bash
python-torch examples/mosaic/convert_checkpoint.py --stage export \
  --input /path/to/era5_best.pt --output /path/to/era5_weights.npz \
  --expected-sha256 187df31a0caef3e934e4eb8a11506c9fea518ff0e02ce6d1f804fc6c8a78a940

python-paddle examples/mosaic/convert_checkpoint.py --stage convert \
  --input /path/to/era5_weights.npz --variant era5 \
  --static /path/to/static_vars.npz --output /path/to/paddle-era5

python-paddle examples/mosaic/prepare_data.py --samples 1 \
  --init-time 2020-01-02T00:00 --output /path/to/era5_sample.npz

python-paddle examples/mosaic/infer.py --model /path/to/paddle-era5 \
  --static /path/to/static_vars.npz --stats /path/to/norm_stats_era5.npz \
  --data /path/to/era5_sample.npz --device gpu:0 --output /path/to/predictions
```

数据提取依赖 `zarr<3`、`fsspec[http]`、`pandas`，仅读取需要的时间块，不下载整个 ERA5。转换时严格校验所有可训练参数名称和形状，几何 buffer 重新生成并保存。

若要精确保留原版预处理，在下面的 Torch 对照运行中增加 `--geometry-output /path/to/source_geometry.npz`，然后重新执行转换命令并增加 `--geometry /path/to/source_geometry.npz`。转换器要求覆盖全部 65 个 buffer，并检查名称、形状、dtype、有限值，记录来源 SHA256；原版输出的带 `buffer.` 前缀 trace NPZ 也可作为该参数的输入。此方式不会依赖运行时 Torch，也不会冻结选块路由。原生 checkpoint 已包含静态场和网格，切换经纬度/静态输入时必须重新初始化对应 buffer，不能沿用旧网格。

`reference_forecast.py --help` 给出原网络对照入口，`--sparse-backend native` 使用原 Triton 稀疏内核（需 Triton/GPU），`math` 使用 chunked Torch SDPA。两者的局部/压缩路径均为 Torch SDPA，当前不是原版 FlashAttention 端到端基线。

## 编译器诊断

```bash
python-paddle examples/mosaic/benchmark.py \
  --fixture /path/to/alignment --device gpu:0 \
  --output /path/to/cinn_small.json
```

此脚本使用结构对齐的小模型、固定权重和显式噪声；分别记录首次编译、预热后的动态/静态耗时及输出差。不能用其代替完整模型正式 CINN 验收。

完整权重可用 `benchmark.py --model /path/to/paddle_model --static ... --stats ... --data ... --output ... --device gpu:0 --warmup 2 --steps 5` 探测，不能同时传 `--fixture`。先进行动态预热和同步计时，再记录编译/首次执行，随后静态预热和计时，最后比较输出；当前完整模型仍属探测阶段。编译失败会保留错误和动态基线；若外部限时终止，报告会停留在最后已保存阶段，不能将其视为完成。

完整 ERA5 权重、真实单样本、RTX3090 的两次探测吞吐提升分别 22.73% / 22.52%。第二次 warmup3/steps10：动态 1.46270s、CINN 1.19380s、编译首次执行 229.283s。最大输出差 0.00216770、平均差 1.31741e-5，严格 `atol=1e-4` **失败**，脚本非零退出；与此同时编译后逐变量天气 RMSE 相对动态最差退化仅 0.000602857%，单样本任务筛查通过。报告分开记录两项结论，不能以任务误差小覆盖严格输出失败。完整输出保存在报告同名 `.npz`，便于后续定位。尚非多样本/滚动编译验证或正式 CINN 验收。

完整模型 benchmark 增加 `--samples 3`，使用同一个编译图依次检查前三个真实输入，计时仍只使用第一个样本。报告 `cases` 为逐样本差异，`task_metric_screening.aggregate_variables` 为先平均 MSE 再开方的聚合天气指标。为兼容既有报告，顶层 `max_diff`/`mean_diff` 及 `task_metric_screening.variables` 仍描述样本0；跨样本严格判定看 `precision_pass` 和 `multi_sample_worst_diff`。

注意 Paddle `to_static(layer)` 会原地替换 `layer.forward`：脚本保留编译前的绑定方法作为后续样本的动态基线，不能调用已转换的同一 Layer 假装动态对照。该行为有独立回归测试。

2026-10-22 三样本结果：动态1.46553s、CINN1.19144s，吞吐提升23.0056%，编译首次226.332s。三个样本动态输出与此前独立动态运行逐点完全一致。每个样本最大输出差分别0.00216770/0.00197840/0.00360155，严格atol1e-4仍失败。单个样本×变量的最差天气RMSE退化0.00108478%，聚合82变量最差退化0.000412214%，所有样本任务筛查通过。只在样本0计时，精度覆盖三个相邻日期，未验证编译后自回归滚动。CINN保持可选诊断，不作为默认或已完成验收的后端。

2026-10-25 显式设置 `FLAGS_cinn_nvrtc_cubin_with_fmad=0 FLAGS_use_fast_math=0`
复测相同三个输入：动态和编译输出均与上述历史结果逐点相同，最差差异仍为
0.00360155，严格精度失败；本次吞吐提升22.04%。关闭这些开关没有修复精度，
不能作为推荐配置。报告 `compiler_environment` 记录编译器环境变量的显式覆盖值，
`null` 表示未设置、使用运行时默认值；它不表示该开关在当前构建中的有效默认值。

同日设置 `FLAGS_deny_cinn_ops='sum;mean;max;softmax'`，保留原生归约与 softmax：
三个样本最大差异为0.00391579/0.00358009/0.00344467，严格精度仍失败；
动态1.46068s、编译1.64666s，吞吐下降11.29%，首次编译323.058s。
动态输出与旧基线逐点相同。该方案没有同时改善精度与速度，不作为推荐配置。

单独设置 `FLAGS_enable_fuse_parallel_matmul_pass=0` 关闭并行矩阵乘合并：
三样本最大差异为0.00994182/0.00244045/0.00152922，严格精度仍失败；
动态1.46920s、编译1.19822s，吞吐提升22.62%，首次编译209.845s。
该配置没有解决逐点精度，同样不作为推荐配置。上述试验不改变默认数学实现、
权重或验收阈值，也不能替代仍未通过的春夏逐日期天气筛查。

首次 GPU 诊断记录：106,592 参数，动态 14.2357ms，CINN 3.69774ms，吞吐提升 284.985%，输出 max diff 1.49e-7；首次编译及执行 135.137s。该结果仅限上述小配置。

完整精度诊断可在 `infer.py` 和 `reference_forecast.py` 增加 `--trace`：逐层保留最多 16,384 个抽样元素，完整保存第一层输入、所有几何 buffer 和 top-k 路由索引。因此完整模型 trace 可达数百 MB，并会引入同步/CPU 拷贝开销，开启后的耗时不能用于性能比较。

```bash
python-paddle examples/mosaic/compare_traces.py \
  --paddle /path/to/paddle_trace_run --reference /path/to/torch_trace_run \
  --output /path/to/comparison.json
```

对比脚本将选块集合变化和集合内顺序变化分开统计，不输出验收结论。`infer.py --reference-trace` 可替换几何、`--reference-routes` 可冻结路由，二者均为隔离诊断开关，报告会明确标记，不能将其当作独立迁移结果。

原始 Triton BF16 在线 softmax 的独立前向诊断：

```bash
python-torch examples/mosaic/validate_sparse_kernel.py --framework torch \
  --source /path/to/mosaic --output /path/to/kernel_check
python-paddle examples/mosaic/validate_sparse_kernel.py --framework paddle \
  --output /path/to/kernel_check --device gpu:0
```

固定输入 `[1,512,4,64]`、两组 KV heads、块长 128、每次选 3 块，数学 Paddle 算子对原 Triton 的 max diff 为 `0.00276860`，按原版 BF16 转换和逐块在线 softmax 顺序模拟后为 `0.000205725`（平均差 `2.41152e-7`）。仍非逐位一致，且未验证自定义反向；该模拟仅在示例诊断代码中，不替换架构中的可微数学 attention。

### 超线样本的逐调用量化与敏感性诊断

`reference_forecast.py --sparse-diagnostics` 在原版完整预报的每次稀疏调用中固定相同 Q/K/V 和路由，比较独立数学 attention、仅QKV量化、BF16在线softmax。每次仍返回原内核输出，替代结果不反馈后续层；`--sparse-sensitivity` 还将Q各元素朝正无穷方向移动一个FP32 ULP，保持K/V/路由不变，测量原内核与数学attention的局部响应。两开关仅支持FP32/native参考，不验证反向，也不能用于速度benchmark。

```bash
# 沿用上面的完整 reference_forecast.py 命令，增加：
# --sparse-sensitivity --output /path/to/new_diagnostic
# 逐调用结果保存在 new_diagnostic/sparse_diagnostics.json
```

2026-10-22 春/夏样本各14次调用：固定实际原版输入，BF16在线模拟相对原始内核的算子RMSE比数学实现降低约156–1328倍；但仍有残余差异。诊断开启/关闭的最终原版预报逐点完全相同。最敏感的均为调用12（0起始）：

| 样本 | Q最大单ULP变化 | 跨BF16舍入边界的Q元素 | 原内核局部输出最大变化 | 数学attention局部输出最大变化 |
| --- | ---: | ---: | ---: | ---: |
| 春季 | 7.63e-6 | 622 / 37,748,736 | 0.08050 | 0.0003073 |
| 夏季 | 7.63e-6 | 644 / 37,748,736 | 0.22717 | 0.0002892 |

这里是**局部注意力激活**，不是完整模型归一化预报或天气RMSE。人工朝同一方向的ULP扰动不等于实际跨框架误差分布，但证明即使固定路由，BF16舍入边界也能放大微小输入误差；不能把全部差异仅归为top-k路由，或只依据隔离算子改善就认定整网对齐。

完整Paddle独立推理中，`--sparse-backend native-emulation` 的春/夏最差天气RMSE退化反而为2.96705%/1.59630%（3/7变量超限）；新增 `--sparse-backend qkv-bf16` 只保留QKV量化、全选块联合FP32 softmax，分别2.08862%/1.33539%（4/2变量超限）。两者均未改善默认数学路径，不作为修复或验收结果。`qkv-bf16` 与 `native-emulation` 均强制inference-only、禁止用于LoRA或全参数训练；默认仍是可微数学attention。新增CPU测试用独立dense mask核对QKV量化GQA/gather，并验证训练调用被拒绝。
