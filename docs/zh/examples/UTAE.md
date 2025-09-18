# 农作物种植情况实时监测

基于卫星图像时间序列的语义分割

## 使用方法

### 语义分割任务
- 训练：
```bash
python train_semantic.py \
    --dataset_folder "/path/to/PASTIS" \
    --epochs 100 \
    --batch_size 2 \
    --num_workers 0 \
    --display_step 10
```

- 测试：
```bash
wget -nc -O pretrained/utae_semantic.pdparams https://paddle-org.bj.bcebos.com/paddlescience/models/utae/semantic.pdparams
python test_semantic.py \
  --weight_file pretrained/utae_semantic.pdparams \
  --dataset_folder "/path/to/PASTIS" \
  --device gpu
  --num_workers 0
```

### 全景分割任务
- 训练：
```bash
python train_panoptic.py \
    --dataset_folder "/path/to/PASTIS" \
    --epochs 100 \
    --batch_size 2 \
    --num_workers 0 \
    --warmup 5 \
    --l_shape 1 \
    --display_step 10
```

- 测试：
```bash
wget -O pretrained/utae_panoptic.pdparams https://paddle-org.bj.bcebos.com/paddlescience/models/utae/panoptic.pdparams
python test_panoptic.py \
  --weight_folder /pretrained/utae_panoptic.pdparams \
  --dataset_folder /path/to/PASTIS \
  --batch_size 2 \
  --num_workers 0 \
  --device gpu
```
## 背景简介
对农作物种植分布和生长状态进行高效、精准的监测，是现代智慧农业和粮食安全领域的核心需求。传统的人工勘察方法耗时费力，而利用单时相卫星影像进行分析的方法，难以应对云层遮挡问题，也无法捕捉作物在整个生长周期中的动态变化规律。

卫星图像时间序列（Satellite Image Time Series, SITS）技术为解决这一难题提供了新的途径。通过持续采集同一区域在不同时间的多光谱影像，SITS数据蕴含了作物从播种、出苗、生长、成熟到收割的全过程光谱和纹理信息。然而，SITS数据具有​​时序长、维度高、时空关联性强​​等特点，如何从中高效地提取特征并进行精确的像素级分类（语义分割）是一项重大的技术挑战。

本项目基于模型​​U-TAE（U-Net Temporal Attention Encoder）​​，利用​​PaddlePaddle深度学习框架​​进行实现，旨在构建一个端到端的解决方案，对​​PASTIS数据集​​中的卫星影像时间序列进行语义分割，从而实现对多种农作物种植情况的自动化、高精度识别与监测。该技术可广泛应用于农业资源调查、产量预估、灾害评估等领域，具有重要的实用价值。

## 模型原理
本章节仅对U-TAE的模型原理进行简单地介绍，详细的理论推导请阅读[Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks](https://arxiv.org/abs/2107.07933)

1. 整体结构
UTAE（U-Net Temporal Attention Encoder）采用 编码器-解码器 架构，专为 卫星图像时间序列语义分割 设计：
- 编码器：轻量化 ResNet-18，提取单时相空间特征
- 解码器：U-TAE 模块，通过 时间注意力机制 聚合多时相全局上下文
- 输出：与输入同分辨率的像素级类别概率图
![U-TAE Architecture](https://paddle-org.bj.bcebos.com/paddlescience/docs/utae/utae.png)

2. 时间注意力机制（Temporal Attention）
对于 T 帧序列，UTAE 在解码阶段为每一帧计算 帧间相似度权重，实现 自适应时序聚合：
- Query = 当前帧特征
- Key / Value = 全部帧特征
- 权重 = Softmax(Query·Key)
- 聚合特征 = Σ(权重 × Value)
- 该机制自动抑制云层、阴影等低质量帧，提升作物边界清晰度。

3. 全局-局部注意力块（GLTB）
每个解码器层包含 两个并行分支：
- 全局分支：Multi-Head Self-Attention，建模 田块级 长程依赖
- 局部分支：3×3 深度可分离卷积，保留 边缘细节
- 输出通过 逐元素相加融合，兼顾全局上下文与局部纹理。

4. 实时推理优化
轻量级骨干：ResNet-18 参数量 < 12 M
- 帧间共享权重：同一序列内只计算一次 Key/Value
- 滑动窗口：大图分块推理，显存占用恒定

## 数据集介绍
PASTIS数据集，该数据集由2433个10×128×128形状的多光谱图像序列组成。每个序列包含2018年9月至2019年11月之间的38至61个观察点，总计超过20亿像素。获取间隔时间不均匀，平均为5天。这种缺乏规律性的现象是由于卫星数据提供商对大量云层覆盖的采集进行了自动过滤。该数据集覆盖4000多平方公里，图像来自法国四个不同地区，气候和作物分布多样。
数据集可通过 [PASTIS官网](https://zenodo.org/records/5012942) 下载。

## 模型构建
1 模型构建
本案例基于 UTAE（U-TAE） 实现，用 PaddleScience 封装如下：
```
--8<--
examples/utae/src/model.py:1:80
--8<--
```
2 约束器构建
采用数据驱动方式，使用 SupervisedConstraint 构建监督约束。
- 训练集数据加载：

```
--8<--
examples/utae/train.py:25:50
--8<--
```
- 定义监督约束：
```
--8<--
examples/utae/train.py:51:65
--8<--
```
3 评估器构建
每 check_val_every_n_epoch 轮使用验证集评估，采用 SupervisedValidator：
```
--8<--
examples/utae/train.py:66:85
--8<--
```
4 学习率与优化器构建
学习率 1e-3，优化器 Adam，代码如下：
```
--8<--
examples/utae/train.py:86:92
--8<--
```

## 实验结果
在 PASTIS 数据集上，本案例复现了以下性能（PaddlePaddle 实现）：
- **SQ (Segmentation Quality)**: 83.8  
- **RQ (Recognition Quality)**: 58.9  
- **PQ (Panoptic Quality)**: 49.7  
![rusult](https://paddle-org.bj.bcebos.com/paddlescience/docs/utae/rusult.png)

## 参考文献
U-TAE 原论文：Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks

官方 PyTorch 实现：https://github.com/VSainteuf/utae-paps

数据集与基准：https://github.com/VSainteuf/pastis-benchmark
