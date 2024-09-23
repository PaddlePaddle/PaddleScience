# VelocityGAN

!!! note

    1. 运行之前，建议快速了解一下[数据集](#31)和[数据读取方式](#32-dataset-api)。
    2. 将[OpenFWI数据集](https://openfwi-lanl.github.io/docs/data.html#vel)下载到`FWIOpenData`目录中对应的子目录（如`Flatvel_A`）。
    3. 将yaml配置文件中的`anno`参数与数据集对应。

=== "模型训练命令"
    ``` sh
    python velocityGAN.py
    ```

=== "模型评估命令"
    ``` sh
    python velocityGAN.py model=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/velocitygan/velocitygan_pretrained.pdparams
    ```

| 预训练模型 | 指标                                        |
| :--------- | :------------------------------------------ |
| [velocitygan_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/velocitygan/velocitygan_pretrained.pdparams) | MAE: 0.0669<br>RMSE: 0.0947<br>SSIM: 0.8511 |

## 1. 背景简介

地下速度图像在地球科学领域具有重要作用。它反映了地震波在地下各个区域的传播速度，为探测地球内部结构提供了关键信息。地震波形反演方法被广泛应用于重构地下速度成像。传统的物理驱动的求解方法是一个数值优化过程，需要经历多次迭代并求解波动方程。这不仅计算成本高，而且通常只能达到局部最优解，导致图像精度有限。基于数据驱动的深度学习方法可以减轻这些问题，在更短的时间内生成更高精度的速度图像。

VelocityGAN就是一个具体的例子。它是一个端到端的框架，能够直接从原始地震波形数据生成高质量的速度图像。论文表明，VelocityGAN 超过了传统的物理驱动波形反演方法，并在数据驱动的基准测试中达到了SOTA的性能。

## 2. 模型原理

作为一种数据驱动的深度学习方法，VelocityGAN可以直接学习波形数据到速度图像的映射关系，而无需求解波动方程。本段落仅简单介绍模型原理，具体细节请阅读[VelocityGAN: Data-Driven Full-Waveform Inversion Using Conditional Adversarial Networks](https://arxiv.org/abs/1809.10262v6)。

### 2.1 模型结构

VelocityGAN是一个条件对抗网络，包含了一个image-to-image的生成器和一个CNN的判别器。下图显示了模型的整体结构。

![velocityGAN](https://paddle-org.bj.bcebos.com/paddlescience/docs/velocitygan/velocityGAN.png)

- `Generator`是一种Encoder-Decoder结构的卷积神经网络。Encoder从地震波形数据中提取特征，并逐步将其压缩成潜在向量（latent vector）；Decoder则根据这个潜在向量推算出相应的速度图。

- `Discriminator`是由9层卷积块组成的模型。输入速度图像，输出图像的真实性分数。

### 2.2 损失函数

判别器的损失函数采用了Wasserstein损失和梯度惩罚。其表达式为：

$$
L_d = \underset{\tilde{x} \sim \mathbb{P}_g}{\mathbb{E}} D(\tilde{x}) - \underset{x \sim \mathbb{P}_r}{\mathbb{E}}D(x) + \lambda \underset{\hat{x} \sim \mathbb{P}_{\hat{x}}}{\mathbb{E}} \left[ \left( \| \nabla_{\hat{x}} D(\hat{x}) \|_2 - 1 \right)^2 \right]
$$

其中$\mathbb{P}_g$是生成器的分布，$\mathbb{P}_r$是真实数据的分布，$\mathbb{P}_{\hat{x}}$是来自$\mathbb{P}_g$和$\mathbb{P}_r$的混合插值样本。

生成器的损失函数是对抗性损失[$- \underset{\tilde{x} \sim \mathbb{P}_g}{\mathbb{E}}D(\tilde{x})$]和内容损失（MAE、MSE）的组合。其表达式为：

$$
L_g = - \underset{\tilde{x} \sim \mathbb{P}_g}{\mathbb{E}}D(\tilde{x}) + \frac{\lambda_1}{w\cdot h} \sum_{i=1}^{w} \sum_{j=1}^{h} \left| \tilde{v}(i,j) - v(i,j) \right| + \frac{\lambda_2}{w\cdot h}\sum_{i=1}^{w} \sum_{j=1}^{h} \left( \tilde{v}(i,j) - v(i,j) \right)^2
$$

其中，$w$和$h$分别为速度图的宽和高，$v(\cdot)$和$\tilde{v}(\cdot)$分别表示速度图的真实像素值和预测像素值。$\lambda_1$和$\lambda_2$为超参数，用于调节两项损失的相对重要性。

## 3. 模型构建

接下来开始讲解如何使用PaddleScience框架实现VelocityGAN。以下内容仅对关键步骤进行阐述，其余细节请参考 [API文档](https://paddlescience-docs.readthedocs.io/zh-cn/latest/zh/api/arch/)。

### 3.1 数据集介绍

数据集采用了[SMILE Team](https://smileunc.github.io/)开源的[OpenFWI](https://openfwi-lanl.github.io/docs/data.html#vel)数据集。

OpenFWI一共12份数据集，共分成了四类：Vel Family、Fault Family、Style Family和Kimberlina Family。本案例主要采用了前两类，其配置信息如下：

![image-20240830153600238](https://paddle-org.bj.bcebos.com/paddlescience/docs/velocitygan/vel_family.png)

![image-20240830153613634](https://paddle-org.bj.bcebos.com/paddlescience/docs/velocitygan/fault_family.png)

其中，每份数据集都包含了波形数据和对应的速度图像。下图展示了每份数据集中速度图像的一个示例。

![image-20240830154311787](https://paddle-org.bj.bcebos.com/paddlescience/docs/velocitygan/data.png)

可以看到，Vel Family包含了地质界面平直和弯曲的两种情况，而Fault Family在此基础上增加了一些地质断层。

每个样本都包含了一张速度图像和五张波形数据，如下图所示。

![image-20240830154807670](https://paddle-org.bj.bcebos.com/paddlescience/docs/velocitygan/sample.png)

其中，5个红星排成一排代表地面上的五个震源，70个接收器也同样布置在地面上。地震波向下传播后会反弹回来，接收器每隔0.001秒记录一次数据，共计1000个。因此，生成了一个形状为（5，1000，70）的地震波形数据集。

注意：所有数据并非真实采集的数据，而是模拟生成的。具体细节请阅读[OpenFWI: Large-Scale Multi-Structural Benchmark Datasets for Seismic Full Waveform Inversion](https://arxiv.org/abs/2111.02926)。

### 3.2 构建dataset API

由于一份数据集由120个数据文件组成，传入所有文件路径是很麻烦的。为了方便读取数据，可以将所有路径打包成一个文本文件。通过依次解析其中的路径，从而读取所有数据。由于这种特殊的读取方式，我们无法使用PaddleScience内置的dataset API，所以自定义了`ppsci.data.dataset.FWIDataset`。

下面给出dataloader的配置代码：
``` py linenums="120"
--8<--
examples/velocityGAN/velocityGAN.py:120:141
--8<--
```
其中，`dataset`使用我们自定义的`FWIDataset`，`anno`传入的是文本文件的路径，它包含了所有数据文件的路径。

### 3.3 模型构建

本案例的VelocityGAN没有被内置在PaddleScience中，需要额外实现，因此我们自定义了`ppsci.arch.VelocityGenerator`和`ppsci.arch.VelocityDiscriminator`。

模型的构建代码如下：

``` py linenums="112"
--8<--
examples/velocityGAN/velocityGAN.py:112:114
--8<--
```

参数配置如下：
``` yaml linenums="41"
--8<--
examples/velocityGAN/conf/velocityGAN.yaml:41:58
--8<--
```

### 3.4 自定义loss

VelocityGAN的损失函数有点复杂，需要我们自定义实现。PaddleScience提供了用于自定loss函数的API——`ppsci.loss.FunctionalLoss`。方法为先定义loss函数，再将函数名作为参数传给 `FunctionalLoss`。需要注意，自定义loss函数的输入输出需要是字典的格式。

#### 3.4.1 Generator的loss

Generator的loss包含了L1 loss 、L2 loss和对抗性损失。这三项loss都有对应的权重，如果某一项 loss 的权重为 0，则表示训练中不添加该 loss 项。

``` py linenums="24"
--8<--
examples/velocityGAN/functions.py:24:53
--8<--
```

#### 3.4.2 Discriminator的loss

Discriminator的loss包含了Wasserstein损失和梯度惩罚。其中，只有梯度惩罚项有权重参数。
``` py linenums="68"
--8<--
examples/velocityGAN/functions.py:68:119
--8<--
```

注意：

``` py linenums="80"
--8<--
examples/velocityGAN/functions.py:80:80
--8<--
```

表示pred变量不参与梯度计算。这是因为pred仅作为Discriminator的输入，不需要考虑它的梯度。并且，pred是Generator的输出，如果不停止梯度计算，Generator的参数梯度会在判别器训练的时候累加，并最终影响生成器第一个批次的训练。

### 3.5 约束构建

本案例使用`ppsci.constraint.SupervisedConstraint`构建约束。

构建代码如下：

``` py linenums="143"
--8<--
examples/velocityGAN/velocityGAN.py:143:158
--8<--
```

其中，`output_expr`指定了如何构建`output_dict`，而`name`为约束的名字，方便后续对其索引。

约束构建完成后，需要创建成字典的形式，方便后续传入给`ppsci.solver.Solver`。

### 3.6 优化器构建

VelocityGAN使用AdamW优化器，可直接调用`ppsci.optimizer.AdamW`构建，代码如下：

``` py linenums="160"
--8<--
examples/velocityGAN/velocityGAN.py:160:165
--8<--
```

### 3.7 Solver构建

将构建好的模型、约束、优化器和其它参数传递给 `ppsci.solver.Solver`。

``` py linenums="167"
--8<--
examples/velocityGAN/velocityGAN.py:167:184
--8<--
```

### 3.8 模型训练

``` py linenums="186"
--8<--
examples/velocityGAN/velocityGAN.py:186:190
--8<--
```

### 3.9 自定义metric

本案例的评估指标为：MAE(Mean Absolute Error), RMSE(Root Mean Squared Error)和SSIM(Structural SIMilarity)。其中，PaddleScience提供了MAE和RMSE的API，而SSIM需要我们额外实现。

PaddleScience提供了用于自定metric函数的API——`ppsci.metric.FunctionalMetric`。方法为先定义metric函数，再将函数名作为参数传给 `FunctionalMetric`。需要注意，自定义metric函数的输入输出需要是字典的格式。

SSIM的实现代码如下：
``` py linenums="199"
--8<--
examples/velocityGAN/functions.py:199:312
--8<--
```

### 3.10 Validator构建

本案例使用`ppsci.validate.SupervisedValidator`构建评估器。

``` py linenums="56"
--8<--
examples/velocityGAN/velocityGAN.py:56:68
--8<--
```

### 3.11 模型评估

将模型、评估器和权重路径传递给`ppsci.solver.Solver`后，通过`solver.eval()`启动评估。

``` py linenums="70"
--8<--
examples/velocityGAN/velocityGAN.py:70:78
--8<--
```

### 3.12 可视化

评估完成后，我们以图片的形式对结果进行可视化，代码如下：

``` py linenums="80"
--8<--
examples/velocityGAN/velocityGAN.py:80:94
--8<--
```

## 4. 完整代码

``` py linenums="1" title="velocityGAN.py"
--8<--
examples/velocityGAN/velocityGAN.py
--8<--
```

## 5. 结果展示

使用[FlatVel-A](https://drive.google.com/drive/folders/1NIdjiYhjWSV9NHn7ZEFYTpJxzvzxqYRb)数据集的训练结果。

|  MAE   |  RMSE  |  SSIM  |
| :----: | :----: | :----: |
| 0.0669 | 0.0947 | 0.8511 |

![image-20240914192445180](https://paddle-org.bj.bcebos.com/paddlescience/docs/velocitygan/flatvel_a_1.png)

![image-20240914192456002](https://paddle-org.bj.bcebos.com/paddlescience/docs/velocitygan/flatvel_a_2.png)

## 6. 参考文献

- [VelocityGAN: Data-Driven Full-Waveform Inversion Using Conditional Adversarial Networks](https://arxiv.org/abs/1809.10262v6)

- [OpenFWI: Large-Scale Multi-Structural Benchmark Datasets for Seismic Full Waveform Inversion](https://arxiv.org/abs/2111.02926)

- [参考代码](https://github.com/lanl/OpenFWI?tab=readme-ov-file#ref2)
