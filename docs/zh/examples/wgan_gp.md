# WGANGP

!!! note

    1. 运行之前将[Cifar10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)下载,并更新wgangp_cifar10.yaml中的data_path
    2. 运行之前将[MINST](http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz)下载，并更新wgangp_mnist.yaml中的data_path

=== "模型训练命令"

```sh
# CIFAR10实验
python wgangp_cifar10.py
# MNIST实验
python wgangp_mnist.py
# 玩具数据集实验
python wgangp_toy.py
```

=== "模型评估命令"

```sh
# CIFAR10实验
python wgangp_cifar10.py mode=eval EVAL.pretrained_gen_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/wgangp/model_generator_cifar10.pdparams #EVAL.pretrained_dis_model_path为从https://paddle-org.bj.bcebos.com/paddlescience/models/wgangp/model_discriminator_cifar10.pdparams下载后模型地址
# MNIST实验
python wgangp_mnist.py mode=eval EVAL.pretrained_gen_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/wgangp/model_generator_mnist.pdparams #EVAL.pretrained_dis_model_path为从https://paddle-org.bj.bcebos.com/paddlescience/models/wgangp/model_discriminator_mnist.pdparams下载后模型地址
# 玩具数据集实验
python wgangp_toy.py mode=eval EVAL.pretrained_gen_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/wgangp/model_generator_toy_8gaussians.pdparams #EVAL.pretrained_dis_model_path为从https://paddle-org.bj.bcebos.com/paddlescience/models/wgangp/model_discriminator_toy_8gaussians.pdparams下载后模型地址
```

| 预训练模型                                                                                      | 指标      |
|:-------------------------------------------------------------------------------------------|:--------|
| [wgangp_cifar10_gen_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/wgangp/model_generator_cifar10.pdparams) <br> [wgangp_cifar10_dis_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/wgangp/model_discriminator_cifar10.pdparams) | IS: 5.2 |



## 1. 背景简介

在数字图像处理和机器学习领域，生成对抗网络（GANs）因其卓越的图像生成能力而受到广泛关注。然而，传统的GAN架构在训练过程中可能会遇到不稳定的问题，尤其是在生成高分辨率或复杂场景的图像时。为了解决这些问题，研究人员提出了带有梯度惩罚的Wasserstein生成对抗网络（WGAN-GP），它不仅增强了训练过程的稳定性，还显著提升了生成图像的质量。

WGAN-GP通过改进损失函数来最小化真实数据分布与生成数据分布之间的差异，并引入梯度惩罚机制以确保训练过程中的平滑性和稳定性。这种优化方法克服了传统GAN中常见的模式崩溃问题，同时促进了更高效的训练和更逼真的图像生成。

## 2. 模型原理

WGAN-GP提出一种替代权重剪裁的方法：对评论者输入梯度的范数施加惩罚。在几乎无需超参数调整的情况下稳定训练多种GAN架构.

### 2.1 模型结构

WGAN-GP是一个条件对抗网络，包含了一个noise-to-image的生成器和一个CNN的判别器。下面显示了模型的整体结构。

```
    noise===>generator===>fake_image==
                                      ==>discriminator===>Wasserstein Loss+Gradient Penalty
                               image==
```

- `Generator`是一种卷积神经网络。

- `Discriminator`是由卷积块组成的模型。输入图像，输出图像的真实性分数。

### 2.2 损失函数

判别器的损失函数采用了Wasserstein损失和梯度惩罚。其表达式为：

$$
L_d = \underset{\tilde{x} \sim \mathbb{P}_g}{\mathbb{E}} D(\tilde{x}) - \underset{x \sim \mathbb{P}_r}{\mathbb{E}}D(x) + \lambda \underset{\hat{x} \sim \mathbb{P}_{\hat{x}}}{\mathbb{E}} \left[ \left( \| \nabla_{\hat{x}} D(\hat{x}) \|_2 - 1 \right)^2 \right]
$$

其中$\mathbb{P}_g$是生成器的分布，$\mathbb{P}_r$是真实数据的分布，$\mathbb{P}_{\hat{x}}$是来自$\mathbb{P}_g$和$\mathbb{P}_r$的混合插值样本。

生成器的损失函数是对抗性损失[$- \underset{\tilde{x} \sim \mathbb{P}_g}{\mathbb{E}}D(\tilde{x})$]。其表达式为：

$$
L_g = - \underset{\tilde{x} \sim \mathbb{P}_g}{\mathbb{E}}D(\tilde{x})
$$

其中$\mathbb{P}_g$是生成器的分布

## 3. 模型构建

接下来开始讲解如何使用PaddleScience框架实现WGAN-GP。以下内容仅对关键步骤进行阐述，其余细节请参考 [API文档](https://paddlescience-docs.readthedocs.io/zh-cn/latest/zh/api/arch/)。

### 3.1 数据集介绍

数据集采用了[Cifar10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)数据集、[MNIST](http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz)和玩具数据集(swissroll/8gaussians/25gaussians)。

Cifar10数据集包含60000张32x32彩色图像，共分为10个类别，每个类别6000张图像。

Cifar10数据集有3个版本

| Version          | Size        | md5sum                               |
|:-----------------|:------------|:-------------------------------------|
| CIFAR-100 python | 161 MB      | eb9058c3a382ffc7106e4002c42a8d85     |
| CIFAR-100 Matlab | 175 MB      | 6a4bfa1dcd5c9453dda6bb54194911f4     |
| CIFAR-100 binary | 161 MB      | 03b5dce01913d631647c71ecec9e9cb8     |

本实现使用的为CIFAR-100 python版本

MNIST数据集包含60000张28x28灰度图像，共分为10个类别，每个类别6000张图像。

玩具数据集

Swissroll：三维非线性流形数据集，呈现连续卷曲的螺旋结构，

8gaussians：二维合成数据集，包含八个对称分布的高斯簇，各簇中心均匀分布于圆周，

25gaussians：高密度高斯混合数据集，由25个规则排列的二维高斯分布构成，簇间距紧凑。

### 3.2 构建dataset API

由于Cifar10数据集由5个数据文件组成，由于数据集组织方式，我们无法直接使用PaddleScience内置的dataset API，所以先把所有数据读取出来，再使用```ppsci.data.dataset.array_dataset.NamedArrayDataset```。

下面给出Cifar10数据集读取的代码：
``` py linenums="167"
--8<--
examples/wgangp/functions.py:167:177
--8<--
```
其中`data_path`传入的是CIFAR-10的路径。

下面给出dataloader的配置代码：
``` py linenums="108"
--8<--
examples/wgangp/wgangp_cifar10.py:108:122
--8<--
```

由于MNIST数据集无法直接使用PaddleScience内置的dataset API，所以先把所有数据读取出来，再使用```ppsci.data.dataset.array_dataset.NamedArrayDataset```。

下面给出MNIST数据集读取的代码：
``` py linenums="368"
--8<--
examples/wgangp/functions.py:368:377
--8<--
```

下面给出dataloader的配置代码：
``` py linenums="101"
--8<--
examples/wgangp/wgangp_mnist.py:101:114
--8<--
```

由于玩具数据集无法直接使用PaddleScience内置的dataset API，所以先把所有数据生成出来，再使用```ppsci.data.dataset.array_dataset.NamedArrayDataset```。

下面给出玩具数据集的生成代码
``` py linenums="194"
--8<--
examples/wgangp/functions.py:194:236
--8<--
```

下面给出dataloader的配置代码：
``` py linenums="94"
--8<--
examples/wgangp/wgangp_toy.py:94:107
--8<--
```

### 3.3 模型构建

本案例的WGAN-GP没有被内置在PaddleScience中，需要额外实现，因此我们自定义了`WganGpCifar10Generator`和`WganGpCifar10Discriminator`、`WganGpMnistGenerator`和`WganGpMnistDiscriminator`、`WganGpToyGenerator`和`WganGpToyDiscriminator`。

模型的构建代码如下：

`WganGpCifar10Generator`和`WganGpCifar10Discriminator`
``` py linenums="92"
--8<--
examples/wgangp/wgangp_cifar10.py:92:93
--8<--
```

`WganGpMnistGenerator`和`WganGpMnistDiscriminator`
``` py linenums="87"
--8<--
examples/wgangp/wgangp_mnist.py:87:88
--8<--
```

`WganGpToyGenerator`和`WganGpToyDiscriminator`
``` py linenums="80"
--8<--
examples/wgangp/wgangp_toy.py:80:81
--8<--
```

参数配置如下：

`WganGpCifar10Generator`和`WganGpCifar10Discriminator`
```yaml linenums="29"
--8<--
examples/wgangp/conf/wgangp_cifar10.yaml:29:43
--8<--
```

`WganGpMnistGenerator`和`WganGpMnistDiscriminator`
```yaml linenums="29"
--8<--
examples/wgangp/conf/wgangp_mnist.yaml:29:38
--8<--
```

`WganGpToyGenerator`和`WganGpToyDiscriminator`
```yaml linenums="29"
--8<--
examples/wgangp/conf/wgangp_toy.yaml:29:37
--8<--
```

### 3.4 自定义loss

WGAN-GP的损失函数较复杂，需要我们自定义实现。PaddleScience提供了用于自定loss函数的API——`ppsci.loss.FunctionalLoss`。方法为先定义loss函数，再将函数名作为参数传给 `FunctionalLoss`。需要注意，自定义loss函数的输入输出需要是字典的格式。

#### 3.4.1 Generator的loss

Cifar10_Generator的loss包含了对抗性损失和分类损失。这两项loss都有对应的权重，如果某一项 loss 的权重为 0，则表示训练中不添加该 loss 项。
``` py linenums="16"
--8<--
examples/wgangp/functions.py:16:44
--8<--
```

MNIST_Generator的loss只包含了对抗性损失。
``` py linenums="313"
--8<--
examples/wgangp/functions.py:313:328
--8<--
```

Toy_Generator的loss只包含了对抗性损失。
``` py linenums="238"
--8<--
examples/wgangp/functions.py:238:254
--8<--
```

#### 3.4.2 Discriminator的loss

Cifar10_Discriminator的loss包含了Wasserstein损失和梯度惩罚以及分类损失。其中，只有分类损失项有权重参数。
``` py linenums="46"
--8<--
examples/wgangp/functions.py:46:95
--8<--
```

MNIST_Discriminator的loss包含了Wasserstein损失和梯度惩罚。
``` py linenums="330"
--8<--
examples/wgangp/functions.py:330:366
--8<--
```

Toy_Discriminator的loss包含了Wasserstein损失和梯度惩罚。
``` py linenums="256"
--8<--
examples/wgangp/functions.py:256:292
--8<--
```

### 3.5 约束构建

所有案例均使用`ppsci.constraint.SupervisedConstraint`构建约束。

构建代码如下：

针对Cifar10的实验
``` py linenums="125"
--8<--
examples/wgangp/wgangp_cifar10.py:125:141
--8<--
```

针对MNIST的实验
``` py linenums="117"
--8<--
examples/wgangp/wgangp_mnist.py:117:132
--8<--
```

针对玩具数据集的实验
``` py linenums="110"
--8<--
examples/wgangp/wgangp_toy.py:110:125
--8<--
```

### 3.6 优化器构建

WGANGP使用Adam优化器，可直接调用`ppsci.optimizer.Adam`构建，代码如下：

针对Cifar10的实验
``` py linenums="144"
--8<--
examples/wgangp/wgangp_cifar10.py:144:158
--8<--
```

针对MNIST的实验
``` py linenums="135"
--8<--
examples/wgangp/wgangp_mnist.py:135:137
--8<--
```

针对玩具数据集的实验
``` py linenums="128"
--8<--
examples/wgangp/wgangp_toy.py:128:131
--8<--
```

### 3.7 Solver构建

将构建好的模型、约束、优化器和其它参数传递给 `ppsci.solver.Solver`。

针对Cifar10的实验
``` py linenums="161"
--8<--
examples/wgangp/wgangp_cifar10.py:161:178
--8<--
```

针对MNIST的实验
``` py linenums="140"
--8<--
examples/wgangp/wgangp_mnist.py:140:157
--8<--
```

针对玩具数据集的实验
``` py linenums="134"
--8<--
examples/wgangp/wgangp_toy.py:134:151
--8<--
```

### 3.8 模型训练

针对Cifar10的实验
``` py linenums="181"
--8<--
examples/wgangp/wgangp_cifar10.py:181:186
--8<--
```

针对MNIST的实验
``` py linenums="160"
--8<--
examples/wgangp/wgangp_mnist.py:160:165
--8<--
```

针对玩具数据集的实验
``` py linenums="154"
--8<--
examples/wgangp/wgangp_toy.py:154:159
--8<--
```

### 3.9 自定义metric

案例中只有针对Cifar10的案例有评估指标为Inception Score，MNIST和Toy案例没有评估指标。由于metric为空会报错所以自定义了一个无效metric

所以我们额外实现了两个metric

PaddleScience提供了用于自定metric函数的API——`ppsci.metric.FunctionalMetric`。方法为先定义metric函数，再将函数名作为参数传给 `FunctionalMetric`。需要注意，自定义metric函数的输入输出需要是字典的格式。

Inception Score的实现代码如下：
``` py linenums="97"
--8<--
examples/wgangp/functions.py:97:154
--8<--
```

invalid_metric的代码如下
``` py linenums="389"
--8<--
examples/wgangp/functions.py:389:391
--8<--
```

### 3.10 Validator构建

本案例使用`ppsci.validate.SupervisedValidator`构建评估器。

针对Cifar10的实验
``` py linenums="53"
--8<--
examples/wgangp/wgangp_cifar10.py:53:62
--8<--
```

针对MNIST的实验
``` py linenums="46"
--8<--
examples/wgangp/wgangp_mnist.py:46:54
--8<--
```

针对玩具数据集的实验
``` py linenums="46"
--8<--
examples/wgangp/wgangp_toy.py:46:52
--8<--
```

### 3.11 模型评估

将模型、评估器和权重路径传递给`ppsci.solver.Solver`后，通过`solver.eval()`启动评估。

针对Cifar10的实验
``` py linenums="65"
--8<--
examples/wgangp/wgangp_cifar10.py:65:74
--8<--
```

针对MNIST的实验
``` py linenums="56"
--8<--
examples/wgangp/wgangp_mnist.py:56:65
--8<--
```

针对玩具数据集的实验
``` py linenums="55"
--8<--
examples/wgangp/wgangp_toy.py:55:63
--8<--
```

### 3.12 可视化

评估完成后，我们以图片的形式对结果进行可视化，代码如下：

针对Cifar10的实验
``` py linenums="76"
--8<--
examples/wgangp/wgangp_cifar10.py:76:87
--8<--
```

针对MNIST的实验
``` py linenums="67"
--8<--
examples/wgangp/wgangp_mnist.py:67:83
--8<--
```

针对玩具数据集的实验
``` py linenums="65"
--8<--
examples/wgangp/wgangp_toy.py:65:75
--8<--
```

## 4. 完整代码

针对Cifar10的实验
``` py
--8<--
examples/wgangp/wgangp_cifar10.py
--8<--
```

针对MNIST的实验
``` py
--8<--
examples/wgangp/wgangp_mnist.py
--8<--
```

针对玩具数据集的实验
``` py
--8<--
examples/wgangp/wgangp_toy.py
--8<--
```

## 6. 参考文献

- [Improved Training of Wasserstein GANs 论文](https://arxiv.org/abs/1704.00028)

- [参考代码](https://github.com/igul222/improved_wgan_training)
