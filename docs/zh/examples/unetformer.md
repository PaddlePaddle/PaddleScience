# UNetFormer

!!! note

    1. 运行之前，建议快速了解一下[数据集](#31)和[数据读取方式](#32-dataset-api)。
    2. 将[Vaihingen数据集]下载到`data`目录中对应的子目录（如`data/vaihingen/train_images`）。
    3. 运行tools/vaihingen_patch_split.py处理原数据集，得到可供训练的数据。

文件数据集结构如下  
```none
airs
├── unetformer(code)
├── model_weights (save the model weights trained on ISPRS vaihingen)
├── fig_results (save the masks predicted by models)
├── lightning_logs (CSV format training logs)
├── data
│   ├── vaihingen
│   │   ├── train_images (original)
│   │   ├── train_masks (original)
│   │   ├── test_images (original)
│   │   ├── test_masks (original)
│   │   ├── test_masks_eroded (original)
│   │   ├── train (processed)
│   │   ├── test (processed)
```

=== "模型训练命令"

    ``` sh
    # 将[Vaihingen数据集]下载到`data`目录中对应的子目录（如`data/vaihingen/train_images`）
    # 创建训练数据集
    python tools/vaihingen_patch_split.py --img-dir "data/vaihingen/train_images" --mask-dir "data/vaihingen/train_masks" --output-img-dir "data/vaihingen/train/images_1024" --output-mask-dir "data/vaihingen/train/masks_1024" --mode "train" --split-size 1024 --stride 512
    # 创建测试数据集
    python tools/vaihingen_patch_split.py --img-dir "data/vaihingen/test_images" --mask-dir "data/vaihingen/test_masks_eroded" --output-img-dir "data/vaihingen/test/images_1024" --output-mask-dir "data/vaihingen/test/masks_1024" --mode "val" --split-size 1024 --stride 1024 --eroded
    # 创建masks_1024_rgb可视化数据集
    python tools/vaihingen_patch_split.py --img-dir "data/vaihingen/test_images" --mask-dir "data/vaihingen/test_masks" --output-img-dir "data/vaihingen/test/images_1024" --output-mask-dir "data/vaihingen/test/masks_1024_rgb" --mode "val" --split-size 1024 --stride 1024 --gt
    # 模型训练
    python train_supervision.py -c config/vaihingen/unetformer.py
    ```

=== "模型评估命令"

    ``` sh
    # 下载处理好的[Vaihingen测试数据集](https://paddle-org.bj.bcebos.com/paddlescience/datasets/unetformer/test.zip)，并解压。  
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/unetformer/test.zip -P ./data/vaihingen/
    unzip -q ./data/vaihingen/test.zip -d data/vaihingen/
    # 下载预训练模型文件
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/models/unetformer/unetformer-r18-512-crop-ms-e105_epoch0_best.pdparams -P ./model_weights/vaihingen/unetformer-r18-512-crop-ms-e105/
    python vaihingen_test.py -c config/vaihingen/unetformer.py -o fig_results/vaihingen/unetformer --rgb
    ```

## 1. 背景简介

遥感城市场景图像的语义分割在众多实际应用中具有广泛需求，例如土地覆盖制图、城市变化检测、环境保护和经济评估等领域。在深度学习技术快速发展的推动下，卷积神经网络（CNN）多年来一直主导着语义分割领域。CNN采用分层特征表示方式，展现出强大的局部信息提取能力。然而卷积层的局部特性限制了网络捕获全局上下文信息的能力。近年来，作为计算机视觉领域的热点研究方向，Transformer架构在全局信息建模方面展现出巨大潜力，显著提升了图像分类、目标检测特别是语义分割等视觉相关任务的性能。

本文提出了一种基于Transformer的解码器架构，构建了类UNet结构的Transformer网络（UNetFormer），用于实时城市场景分割。为实现高效分割，UNetFormer选择轻量级ResNet18作为编码器，并在解码器中开发了高效的全局-局部注意力机制，以同时建模全局和局部信息。本文提出的基于Transformer的解码器与Swin Transformer编码器结合后，在Vaihingen数据集上也取得了当前最佳性能（91.3% F1分数和84.1% mIoU）。

## 2. 模型原理

本段落仅简单介绍模型原理，具体细节请阅读[UNetFormer: A UNet-like Transformer for Efficient
Semantic Segmentation of Remote Sensing Urban
Scene Imagery](https://arxiv.org/abs/2109.08937)。

### 2.1 模型结构

UNetFormer是一种基于transformer的解码器的深度学习网络，下图显示了模型的整体结构。

![UNetFormer1](https://paddle-org.bj.bcebos.com/paddlescience/docs/unetformer/unetformer.png)

- `ResBlock`是resnet18网络的各个模块。

- `GLTB`由全局-局部注意、MLP、两个batchnorm层和两个加和操作组成。

### 2.2 损失函数

判别器的损失函数由两部分组成，主损失函数$\mathcal{L}_{\text {p }}$为SoftCrossEntropyLoss交叉熵损失函数$\mathcal{L}_{c e}$和DiceLoss损失函数$\mathcal{L}_{\text {dice }}$。其表达式为：

$$
\mathcal{L}_{c e}=-\frac{1}{N} \sum_{n=1}^{N} \sum_{k=1}^{K} y_{k}^{(n)} \log \hat{y}_{k}^{(n)}
$$  

$$
\mathcal{L}_{\text {dice }}=1-\frac{2}{N} \sum_{n=1}^{N} \sum_{k=1}^{K} \frac{\hat{y}_{k}^{(n)} y_{k}^{(n)}}{\hat{y}_{k}^{(n)}+y_{k}^{(n)}}
$$  

$$
\mathcal{L}_{\text {p }}=\mathcal{L}_{c e}+\mathcal{L}_{\text {dice }}
$$  

其中N、K分别表示样本数量和类别数量。$y^{(n)}$和$\hat{y}^{(n)}$表示标签的one-hot编码和相应的softmax输出，$\mathrm{n} \in[1, \ldots, \mathrm{n}]$。

为了更好的结合，我们选择交叉熵函数作为辅助损失函数${L}_{a u x}$，并且乘以系数$\alpha$总损失函数其表达式为：

$$
\mathcal{L}=\mathcal{L}_{p}+\alpha \times \mathcal{L}_{a u x}
$$

其中，$\alpha$默认为0.4。

## 3. 模型构建
以下我们讲解释用PaddleScience构建UnetFormer的关键部分。

### 3.1 数据集介绍

数据集采用了[ISPRS](https://www.isprs.org/)开源的[Vaihingen](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)数据集。

ISPRS提供了城市分类和三维建筑重建测试项目的两个最先进的机载图像数据集。该数据集采用了由高分辨率正交照片和相应的密集图像匹配技术产生的数字地表模型（DSM）。这两个数据集区域都涵盖了城市场景。Vaihingen是一个相对较小的村庄，有许多独立的建筑和小的多层建筑，该数据集包含33幅不同大小的遥感图像，每幅图像都是从一个更大的顶层正射影像图片提取的，图像选择的过程避免了出现没有数据的情况。顶层影像和DSM的空间分辨率为9 cm。遥感图像格式为8位TIFF文件，由近红外、红色和绿色3个波段组成。DSM是单波段的TIFF文件，灰度等级（对应于DSM高度）为32位浮点值编码。

![image-vaihingen](https://paddle-org.bj.bcebos.com/paddlescience/docs/unetformer/overview_tiles.jpg)

每个数据集已手动分类为6个最常见的土地覆盖类别。

①不透水面 (RGB: 255, 255, 255)

②建筑物(RGB: 0, 0, 255)

③低矮植被 (RGB: 0, 255, 255)

④树木 (RGB: 0, 255, 0)

⑤汽车(RGB: 255, 255, 0)

⑥背景 (RGB: 255, 0, 0)

背景类包括水体和与其他已定义类别不同的物体（例如容器、网球场、游泳池），这些物体通常属于城市场景中的不感兴趣的语义对象。

### 3.2 构建dataset API

由于一份数据集由33个超大遥感图片组成组成。为了方便训练，我们自定义一个图像分割程序，将原始图片分割为1024×1024大小的可训练图片，程序代码具体信息在GeoSeg/tools/vaihingen_patch_split.py中可以看到。

### 3.3 模型构建

本案例的模型搭建代码如下



参数配置如下：
``` py linenums="12"
--8<--
examples/unetformer/config/vaihingen/unetformer.py:12:36
--8<--
```

### 3.4 loss函数

UNetFormer的损失函数由SoftCrossEntropyLoss交叉熵损失函数和DiceLoss损失函数组成

#### 3.4.1 SoftCrossEntropyLoss


``` py linenums="13"
--8<--
examples/unetformer/geoseg/losses/soft_ce.py:13:43
--8<--
```

#### 3.4.2 DiceLoss

``` py linenums="36"
--8<--
examples/unetformer/geoseg/losses/dice.py:36:145
--8<--
```

#### 3.4.2 JointLoss  
SoftCrossEntropyLoss和DiceLoss将使用JointLoss进行组合

``` py linenums="23"
--8<--
examples/unetformer/geoseg/losses/joint_loss.py:23:40
--8<--
```
#### 3.4.2 UNetFormerLoss  
``` py linenums="93"
--8<--
examples/unetformer/geoseg/losses/useful_loss.py:93:114
--8<--
```

### 3.5 优化器构建

UNetFormer使用AdamW优化器，可直接调用`paddle.optimizer.AdamW`构建，代码如下：

``` py linenums="65"
--8<--
examples/unetformer/config/vaihingen/unetformer.py:65:76
--8<--
```

### 3.6 模型训练

``` py linenums="236"
--8<--
examples/unetformer/train_supervision.py:236:300
--8<--
```


### 3.7 模型测试

``` py linenums="61"
--8<--
examples/unetformer/vaihingen_test.py:61:121
--8<--
```

## 4. 结果展示

使用[Vaihingen](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)数据集的训练结果。

|  F1  |  mIOU  |  OA  |
| :----: | :----: | :----: |
| 0.9062 | 0.8318 | 0.9283 |  

![image-vaihingen1](https://paddle-org.bj.bcebos.com/paddlescience/docs/unetformer/top_mosaic_09cm_area38_0_6.tif)

![image-vaihingen2](https://paddle-org.bj.bcebos.com/paddlescience/docs/unetformer/result.png)

两张图片对比可以看出模型已经精确地分割出遥感图片中建筑、树木、汽车等物体的轮廓，并且很好地处理了重叠区域。
## 6. 参考文献

- [UNetFormer: A UNet-like Transformer for Efficient Semantic Segmentation of Remote Sensing Urban Scene Imagery](https://arxiv.org/abs/2109.08937)
- [https://github.com/WangLibo1995/GeoSeg](https://github.com/WangLibo1995/GeoSeg)
