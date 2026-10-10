# ClimaX

开始训练、评估前，请先从 [Zenodo（record 7064308）](https://zenodo.org/records/7064308) 下载 ClimateBench v1.0 数据集，包括 `train_val.tar.gz`（839 MB）与 `test.tar.gz`（74 MB），分别解压。

原始数据的空间分辨率不是 5.625°，需要用案例自带脚本做一次重采样。脚本依赖的 xesmf 需用 conda 安装：

```sh
conda install -c conda-forge xesmf
```

然后在 `examples/climax` 目录下运行：

```sh
python regrid_climatebench.py <解压后的数据目录> --save_path <重采样输出目录> --ddeg_out 5.625
```

test 的 ssp245 数据要与历史期拼接才能构成完整样本，而 ClimateBench 发布包里 `inputs_historical.nc`、`outputs_historical.nc` 只在 train_val 目录中。请把这两个文件软链或复制到 test 目录，否则会导致加载测试集时文件缺失。

运行本案例还需安装以下依赖：

```sh
pip install -r examples/climax/requirements.txt
```

权重方面，ClimaX 官方在 [Hugging Face tungnd/climax](https://huggingface.co/tungnd/climax) 发布的 5.625° checkpoint（`5.625deg.ckpt`，412 MB）需要转换成 `.pdparams` 格式。转换后的 checkpoint 结构为 `{"state_dict": {"net.*": ...}}`，配置项 `TRAIN.init_state_path` 指向它即可完成严格加载（见 `examples/climax/main.py` 中的 `_load_weights_strict`）。

=== "模型训练命令"

    ``` sh
    cd examples/climax && python main.py mode=train
    # 开启 CINN 编译（to_static + CINN）
    CLIMAX_USE_CINN=1 python main.py mode=train
    ```

=== "模型评估命令"

    ``` sh
    cd examples/climax && python main.py mode=eval EVAL.pretrained_model_path=<checkpoint 路径>
    ```

## 1. 背景简介

气候模拟数据在持续增长，但各机构产出的数据在变量集合、时空分辨率上并不统一，传统的物理数值模式又难以直接吸收这些数据。ClimaX（[arXiv 2301.10343](https://arxiv.org/abs/2301.10343)）针对这一现状，把 Vision Transformer（ViT）用作气候基础模型的骨干：训练时用掩码重建的自监督目标，在 ERA5 等多来源气候数据上预训练，让同一套网络适应不同变量组合与分辨率的输入；预训练得到的权重再迁移到天气预报、降尺度、气候投影等下游任务，只需替换或微调少量模块。

本案例实现的是 ClimaX 在 ClimateBench 气候投影任务上的下游微调与评估。ClimateBench 基于 CMIP6 的情景模拟数据构建，输入 CO2、SO2、CH4、BC 四个变量在过去若干时间步的全球分布，模型输出近地面气温 tas 的未来变化。数据重采样到 5.625° 网格（32×64）后，任务可以看作“多变量、多时间步的图像序列到单变量图像”的回归问题，正好落在 ViT 骨干的处理范围内。

## 2. 模型原理

本章节结合 PaddleScience 的实现代码介绍 ClimaX 的结构，详细的理论推导请阅读原论文。

### 2.1 ClimaX 主干

ClimaX 的骨干是一个 ViT 编码器。与处理单一变量图像的普通 ViT 不同，它为每个输入变量单独配置一个 patch embedding，把该变量的全球场切成 2×2 的 patch 并映射为 token；变量之间靠可学习的变量 embedding 区分身份。全部变量的 token 序列再借助一个可学习查询向量的 cross-attention 聚合成单一序列，这样无论输入几个变量，送入 Transformer 的都是一个统一的 token 序列。

各个模块的构建代码如下：

```py linenums="110" title="ppsci/arch/climax/arch.py"
--8<--
ppsci/arch/climax/arch.py:110:128
--8<--
```

变量聚合的代码如下：

```py linenums="222" title="ppsci/arch/climax/arch.py"
--8<--
ppsci/arch/climax/arch.py:222:233
--8<--
```

序列加上二维正弦位置 embedding 之后，还要加一个 lead time embedding：它由线性层把预报时效（lead time）标量映射到 embed_dim 维，让同一个网络处理不同时效的预测。之后就是标准的 Transformer encoder：

```py linenums="235" title="ppsci/arch/climax/arch.py"
--8<--
ppsci/arch/climax/arch.py:235:260
--8<--
```

### 2.2 ClimateBench 下游变体

气候投影任务的输入是 `[B, T, Vi, H, W]` 的五维张量：T 个连续历史时间步上 Vi 个变量的全球场。`ClimaXClimateBench` 在 ClimaX 主干之外增加了处理时间维的模块，并把原来的多变量解码头换成单个线性头：

```py linenums="110" title="ppsci/arch/climax/climatebench.py"
--8<--
ppsci/arch/climax/climatebench.py:110:127
--8<--
```

前向计算的主流程是：每个时间步分别完成 patch embedding、变量聚合与 Transformer 编码，得到各时间步的 token 序列；序列加上时间位置 embedding 后，各时间步的 token 取均值并按时间排开，用 `time_agg` 这一层 cross-attention 把整个时间窗口压缩成一个全局向量；最后由线性头直接回归出整幅 tas 场。

```py linenums="136" title="ppsci/arch/climax/climatebench.py"
--8<--
ppsci/arch/climax/climatebench.py:136:171
--8<--
```

线性头输出 32×64=2048 个值，reshape 成 `[B, 1, H, W]` 的预测场：

```py linenums="188" title="ppsci/arch/climax/climatebench.py"
--8<--
ppsci/arch/climax/climatebench.py:188:196
--8<--
```

本案例配置了 `freeze_encoder: true`：Transformer 块中除 LayerNorm 之外的所有参数被冻结，训练只更新 patch embedding、各 embedding 层、时间聚合模块和解码头。这对应原论文“冻结编码器做下游微调”的设置，也显著缩小了需要保存的优化器状态。

## 3. 模型训练

### 3.1 数据集介绍

案例使用 ClimateBench v1.0 的 5.625° 版本。训练与验证数据来自六组模拟：ssp126、ssp370、ssp585、historical、hist-GHG、hist-aer，按 `train_ratio: 0.9` 划分训练集与验证集；测试数据来自 ssp245，不参与训练。非 historical 的模拟会先与 historical 段在时间维上拼接，`len_historical` 固定为 165 个时间步。

每个样本的输入由 CO2、SO2、CH4、BC 四个变量、连续 `history: 10` 个时间步的场构成，标签是对应时点的 tas 场。归一化统计量只在训练分区上拟合，验证集与测试集通过 `set_normalize` 复用同一组统计量，避免信息泄漏。

数据集构建参数由 `_dataset_kwargs` 统一装配，train/val/test 三个分区共用同一组参数、只有 `partition` 不同：

```py linenums="51" title="examples/climax/main.py"
--8<--
examples/climax/main.py:51:65
--8<--
```

### 3.2 模型训练

#### 3.2.1 约束器构建

本案例基于数据驱动方法求解问题，训练约束用 `SupervisedConstraint` 构建。训练损失是普通 MSE（`utils.mse_loss`），数据加载配置如下：

```py linenums="184" title="examples/climax/main.py"
--8<--
examples/climax/main.py:184:194
--8<--
```

```py linenums="197" title="examples/climax/main.py"
--8<--
examples/climax/main.py:197:202
--8<--
```

`iters_per_epoch` 直接取约束器数据加载器的长度，不再手工指定。

#### 3.2.2 评估器构建

训练过程中每个 epoch 都要做验证，所以构建了两个 `SupervisedValidator`：`Val` 用验证集，`Test` 用测试集。两者的损失与指标一致：纬度加权 MSE 作为验证损失（`make_w_mse_loss`），指标为纬度加权的 `w_mse`、`w_rmse`、`w_nrmse`（`make_lat_weighted_metrics`），纬度权重按 `cos(lat) / mean(cos(lat))` 计算。`w_mse` 在归一化空间计算，`w_rmse` 与 `w_nrmse` 先做反归一化。这些函数在 `examples/climax/utils.py` 中实现。

```py linenums="68" title="examples/climax/main.py"
--8<--
examples/climax/main.py:68:83
--8<--
```

两个评估器共用训练集拟合的归一化统计量。注意 `validator` 字典里 `Val` 放在最后，这样 Solver 保存最优模型时跟踪的是 `Val/w_mse`，与原始实现选择 checkpoint 的口径一致。

#### 3.2.3 模型构建与权重加载

模型直接由配置文件实例化，本案例的全部模型超参（`img_size`、`patch_size`、`embed_dim`、`depth`、`num_heads` 等）都写在 `conf/climax.yaml` 的 `MODEL` 段：

```py linenums="228" title="examples/climax/main.py"
--8<--
examples/climax/main.py:228:236
--8<--
```

权重加载发生在构建 Solver 之前，`TRAIN.init_state_path` 指向完整初始状态（backbone 加头部，严格加载，键不匹配会直接报错）；否则 `TRAIN.pretrained_path` 指向 MAE 预训练 checkpoint，做宽松加载——位置 embedding 会插值到当前网格，channel embedding 改名为 var embedding，token embedding 与头部参数丢弃后从头训练。

#### 3.2.4 学习率与优化器构建

学习率调度用步进单位的线性 warmup 加余弦退火 `ppsci.optimizer.lr_scheduler.LinearWarmupCosineAnnealingLR`：峰值 5e-4，warmup 60 步，`eta_min` 1e-8。`warmup_epochs` 与 `max_epochs` 都按优化器步数计数（每 epoch 793 步），越过 600 步的调度长度后按 SGDR 式重启。

优化器是 AdamW，参数分成两组：名称含 `var_embed`、`pos_embed`、`time_pos_embed` 的参数不做权重衰减，其余参数按 `weight_decay: 1e-5` 衰减。被 `freeze_encoder` 冻结的参数（`stop_gradient` 为真）不进入优化器：

```py linenums="156" title="examples/climax/main.py"
--8<--
examples/climax/main.py:156:179
--8<--
```

#### 3.2.5 模型训练

上述对象按顺序传给 `ppsci.solver.Solver`，然后启动训练。训练结束后会自动对测试集做一次评估：

```py linenums="244" title="examples/climax/main.py"
--8<--
examples/climax/main.py:244:264
--8<--
```

`to_static` 不被配置文件控制，由环境变量 `CLIMAX_USE_CINN=1` 统一控制：该变量同时打开 CINN 相关 FLAGS 并把 `to_static=True` 传给 Solver。被 `to_static` 包装的模型组网总会用 CINN 编译，因而单个开关同时改变这两处功能。环境变量必须在 `import paddle` 之前设置，`main.py` 已在文件开头处理好，命令行直接 `CLIMAX_USE_CINN=1 python main.py mode=train` 即可。

### 3.3 评估模型

#### 3.3.1 评估器构建

`mode=eval` 路径只构建测试集评估器。归一化统计量仍然在训练分区上重新拟合（与训练时同一套逻辑），保证评估口径不变：

```py linenums="269" title="examples/climax/main.py"
--8<--
examples/climax/main.py:269:278
--8<--
```

#### 3.3.2 加载模型并进行评估

待评估的 checkpoint 由 `EVAL.pretrained_model_path` 指定，用 `_load_weights_strict` 严格加载。Solver 内置的加载函数无法解开 `{"state_dict": ...}` 包装结构，所以结构判断放在 `main.py` 里做：

```py linenums="281" title="examples/climax/main.py"
--8<--
examples/climax/main.py:281:299
--8<--
```

## 4. 完整代码

数据集接口：

```py linenums="1" title="ppsci/data/dataset/climatebench_dataset.py"
--8<--
ppsci/data/dataset/climatebench_dataset.py
--8<--
```

模型结构：

```py linenums="1" title="ppsci/arch/climax/arch.py"
--8<--
ppsci/arch/climax/arch.py
--8<--
```

```py linenums="1" title="ppsci/arch/climax/climatebench.py"
--8<--
ppsci/arch/climax/climatebench.py
--8<--
```

训练损失与评估指标：

```py linenums="1" title="examples/climax/utils.py"
--8<--
examples/climax/utils.py
--8<--
```

模型训练：

```py linenums="1" title="examples/climax/main.py"
--8<--
examples/climax/main.py
--8<--
```

数据重采样脚本：

```py linenums="1" title="examples/climax/regrid_climatebench.py"
--8<--
examples/climax/regrid_climatebench.py
--8<--
```

配置文件：

```py linenums="1" title="examples/climax/conf/climax.yaml"
--8<--
examples/climax/conf/climax.yaml
--8<--
```

## 5. 结果展示

以下数据分别对应前向数值对齐、训练指标对齐和 CINN 训练加速。

前向数值对齐：与 torch 参考实现加载同一份权重、输入同一批数据，forward pass 输出的平均绝对差为 2.24e-08（GPU 上的 eval 前向，动态图为 2.19e-08），处于 FP32 浮点误差量级，可认为两边计算图等价。

训练对齐：从相同初始权重出发，固定 seed、FP32、单卡、batch_size=1，各训练 5 个 epoch 后在测试集上对比最终指标，五项纬度加权指标与 torch 基线的最大差距为 7.3%（w_nrmseg_tas）：

| 指标（test，5-epoch 训练后） | PaddleScience 实现 | torch 参考实现 |
| :--------------------------- | :----------------- | :------------- |
| w_mse                        | 0.13394            | 0.13900        |
| w_rmse                       | 0.36180            | 0.36871        |
| w_nrmse_tas                  | 0.34945            | 0.34081        |
| w_nrmses_tas                 | 0.17455            | 0.17788        |
| w_nrmseg_tas                 | 0.03498            | 0.03258        |
