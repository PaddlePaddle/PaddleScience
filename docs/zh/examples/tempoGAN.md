# tempoGAN(temporally Generative Adversarial Networks)

<a href="https://aistudio.baidu.com/aistudio/projectdetail/6521709" class="md-button md-button--primary" style>AI Studio快速体验</a>

=== "模型训练命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_train.mat -P datasets/tempoGAN/
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_valid.mat -P datasets/tempoGAN/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_train.mat --create-dirs -o ./datasets/tempoGAN/2d_train.mat
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_valid.mat --create-dirs -o ./datasets/tempoGAN/2d_valid.mat
    python tempoGAN.py
    ```

=== "模型评估命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_train.mat -P datasets/tempoGAN/
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_valid.mat -P datasets/tempoGAN/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_train.mat --create-dirs -o ./datasets/tempoGAN/2d_train.mat
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_valid.mat --create-dirs -o ./datasets/tempoGAN/2d_valid.mat
    python tempoGAN.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/tempoGAN/tempogan_pretrained.pdparams
    ```

=== "模型导出命令"

    ``` sh
    python tempoGAN.py mode=export
    ```

=== "模型推理命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_valid.mat -P datasets/tempoGAN/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_valid.mat --create-dirs -o ./datasets/tempoGAN/2d_valid.mat
    python tempoGAN.py mode=infer
    ```

| 预训练模型  | 指标 |
|:--| :--|
| [tempogan_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/tempoGAN/tempogan_pretrained.pdparams) | MSE: 4.21e-5<br>PSNR: 47.19<br>SSIM: 0.9974 |

## 1. 背景简介

流体模拟方面的问题，捕捉湍流的复杂细节一直是数值模拟的长期挑战，用离散模型解决这些细节会产生巨大的计算成本，对于人类空间和时间尺度上的流动来说，很快就会变得不可行。因此流体超分辨率的需求应运而生，它旨在通过流体动力学模拟和深度学习技术将低分辨率流体模拟结果恢复为高分辨率结果，以减少生成高分辨率流体过程中的巨大计算成本。该技术可以应用于各种流体模拟，例如水流、空气流动、火焰模拟等。

生成式对抗网络 GAN(Generative Adversarial Networks) 是一种使用无监督学习方法的深度学习网络，GAN 网络中（至少）包含两个模型：生成器(Generator) 和判别器(Discriminator)，生成器用于生成问题的输出，判别器用于判断输出的真假，两者在相互博弈中共同优化，最终使得生成器的输出接近真实值。

tempoGAN 在 GAN 网络的基础上新增了一个与时间相关的判别器 Discriminator_tempo，该判别器的网络结构与基础判别器相同，但输入为时间连续的几帧数据，而不是单帧数据，从而将时序纳入考虑范围。

本问题主要使用该网络，通过输入的低密度流体数据，得到对应的高密度流体数据，大大节省时间和计算成本。

## 2. 问题定义

本问题包含三个模型：生成器(Generator)、判别器(Discriminator)和与时间相关的判别器(Discriminator_tempo)，根据 GAN 网络的训练流程，这三个模型交替训练，训练顺序依次为：Discriminator、Discriminator_tempo、Generator。
GAN 网络为无监督学习，本问题网络设计中将目标值作为一个输入值，输入网络进行训练。

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。为了快速理解 PaddleScience，接下来仅对模型构建、约束构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 数据集介绍

数据集为使用开源代码包 [mantaflow](http://mantaflow.com/install.html) 生成的 2d 流体数据集，数据集中包括一定数量连续帧的低、高密度流体图像转化成的数值，以字典的形式存储在 `.mat` 文件中。

运行本问题代码前请下载 [训练数据集](https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_train.mat) 和 [验证数据集](https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_valid.mat)， 下载后分别存放在路径：

``` yaml linenums="27"
--8<--
examples/tempoGAN/conf/tempogan.yaml:27:28
--8<--
```

### 3.2 模型构建

<figure markdown>
  ![tempoGAN-arch](https://paddle-org.bj.bcebos.com/paddlescience/docs/tempoGAN/tempoGAN_arch.png){ loading=lazy style="margin:0 auto"}
  <figcaption> tempoGAN 网络模型</figcaption>
</figure>

上图为tempoGAN 完整的模型结构图，但本问题只针对较为简单的情况进行处理，不涉及包含速度和涡度的输入、3d、数据增强、advection operator 等部分，如果您对这些文档中未包含的内容感兴趣，可以自行修改代码并进行进一步实验。

如上图所示，Generator 的输入为低密度流体数据的插值，输出为生成的高密度流体模拟数据，Discriminator 的输入为低密度流体数据的插值分别与 Generator 生成的高密度流体模拟数据、目标高密度流体数据的拼接， Discriminator_tempo 的输入为多帧连续的 Generator 生成的高密度流体模拟数据以及目标高密度流体数据。

虽然输入输出的组成看起来较为复杂，但本质都是流体的密度数据，因此 3 个网络的映射函数都是 $f: \mathbb{R}^1 \to \mathbb{R}^1$。

与简单的 MLP 网络不同，根据要解决的问题不同，GAN 的生成器和判别器有多种网络结构可以选择，在此不再赘述。由于这种独特性，本问题中的 tempoGAN 网络没有被内置在 PaddleScience 中，需要额外实现。

本问题中的 Generator 是一个拥有 4 层改良 Res Block 的模型，Discriminator 和 Discriminator_tempo 为同一个拥有 4 层卷积结果的模型，两者网络结构相同但输入不同。Generator、Discriminator 和 Discriminator_tempo 的网络参数也需要额外定义。

具体代码请参考 [完整代码](#4) 中 gan.py 文件。

由于 GAN 网络中生成器和判别器的中间结果要相互调用，参与对方的 loss 计算，因此使用 Model List 实现，用 PaddleScience 代码表示如下：

``` py linenums="57"
--8<--
examples/tempoGAN/tempoGAN.py:57:76
--8<--
```

注意到上述代码中定义的网络输入与实际网络输入不完全一样，因此需要对输入进行transform。

### 3.3 transform构建

Generator 的输入为低密度流体数据的插值，而数据集中保存的为原始的低密度流体数据，因此需要进行一个插值的 transform。

``` py linenums="270"
--8<--
examples/tempoGAN/functions.py:270:275
--8<--
```

Discriminator 和 Discriminator_tempo 对输入的 transform 更为复杂，分别为：

``` py linenums="360"
--8<--
examples/tempoGAN/functions.py:360:394
--8<--
```

其中：

``` py linenums="369"
--8<--
examples/tempoGAN/functions.py:369:369
--8<--
```

表示停止参数的计算梯度，这样设置是因为这个变量在这里仅作为 Discriminator 和 Discriminator_tempo 的输入，在反向计算时不应该参与梯度回传，如果不进行这样的设置，由于这个变量来源于 Generator 的输出，在反向传播时梯度会沿着这个变量传给 Generator，从而改变 Generator 中的参数，这显然不是我们想要的。

这样，我们就实例化出了一个拥有 Generator、Discriminator 和 Discriminator_tempo 并包含输入 transform 的神经网络模型 `model list`。

### 3.4 参数和超参数设定

我们需要指定问题相关的参数，如数据集路径、各项 loss 的权重参数等。

``` yaml linenums="27"
--8<--
examples/tempoGAN/conf/tempogan.yaml:27:37
--8<--
```

注意到其中包含 3 个 bool 类型的变量 `use_amp`、`use_spatialdisc` 和 `use_tempodisc`，它们分别表示是否使用混合精度训练(AMP)、是否使用 Discriminator 和是否使用 Discriminator_tempo，当 `use_spatialdisc` 和 `use_tempodisc` 都被设置为 `False` 时，本问题的网络结构将会变为一个单纯的 Genrator 模型，不再是 GAN 网络了。

同时需要指定训练轮数和学习率等超参数，注意由于 GAN 网络训练流程与一般单个模型的网络不同，`EPOCHS` 的设置也有所不同。

``` yaml linenums="73"
--8<--
examples/tempoGAN/conf/tempogan.yaml:73:76
--8<--
```

### 3.5 优化器构建

训练使用 Adam 优化器，学习率在 `Epoch` 达到一半时减小到原来的 $1/20$，因此使用 `Step` 方法作为学习率策略。如果将 `by_epoch` 设为 True，学习率将根据训练的 `Epoch` 改变，否则将根据 `Iteration` 改变。

``` py linenums="78"
--8<--
examples/tempoGAN/tempoGAN.py:78:94
--8<--
```

### 3.6 约束构建

本问题采用无监督学习的方式，虽然不是以监督学习方式进行训练，但此处仍然可以采用监督约束 `SupervisedConstraint`，在定义约束之前，需要给监督约束指定文件路径等数据读取配置，因为 tempoGAN 属于自监督学习，数据集中没有标签数据，而是使用一部分输入数据作为 `label`，因此需要设置约束的 `output_expr`。

``` py linenums="122"
--8<--
examples/tempoGAN/tempoGAN.py:122:125
--8<--
```

#### 3.6.1 Generator 的约束

下面是约束的具体内容，要注意上述提到的 `output_expr`：

``` py linenums="98"
--8<--
examples/tempoGAN/tempoGAN.py:98:127
--8<--
```

`SupervisedConstraint` 的第一个参数是监督约束的读取配置，其中 `dataset` 字段表示使用的训练数据集信息，各个字段分别表示：

1. `name`： 数据集类型，此处 `NamedArrayDataset` 表示从 Array 中读取的 `.mat` 类型的数据集；
2. `input`： Array 类型的输入数据；
3. `label`： Array 类型的标签数据；
4. `transforms`： 所有数据 transform 方法，此处 `FunctionalTransform` 为PaddleScience 预留的自定义数据 transform 类，该类支持编写代码时自定义输入数据的 transform，具体代码请参考 [自定义 loss 和 data transform](#38-loss-data-transform)；

`batch_size` 字段表示 batch的大小；

`sampler` 字段表示采样方法，其中各个字段表示：

1. `name`： 采样器类型，此处 `BatchSampler` 表示批采样器；
2. `drop_last`： 是否需要丢弃最后无法凑整一个 mini-batch 的样本，默认值为 False；
3. `shuffle`： 是否需要在生成样本下标时打乱顺序，默认值为 False；

第二个参数是损失函数，此处的 `FunctionalLoss` 为 PaddleScience 预留的自定义 loss 函数类，该类支持编写代码时自定义 loss 的计算方法，而不是使用诸如 `MSE` 等现有方法，具体代码请参考 [自定义 loss 和 data transform](#38-loss-data-transform)。

第三个参数是约束条件的 `output_expr`，如上所述，是为了让程序可以将输入数据作为 `label`。

第四个参数是约束条件的名字，我们需要给每一个约束条件命名，方便后续对其索引。

在约束构建完毕之后，以我们刚才的命名为关键字，封装到一个字典中，方便后续访问，由于本问题设置了`use_spatialdisc` 和 `use_tempodisc`，导致 Generator 的部分约束不一定存在，因此先封装一定存在的约束到字典中，当其余约束存在时，在向字典中添加约束元素。

``` py linenums="129"
--8<--
examples/tempoGAN/tempoGAN.py:129:160
--8<--
```

#### 3.6.2 Discriminator 的约束

``` py linenums="164"
--8<--
examples/tempoGAN/tempoGAN.py:164:201
--8<--
```

各个参数含义与[Generator 的约束](#361-generator)相同。

#### 3.6.3 Discriminator_tempo 的约束

``` py linenums="205"
--8<--
examples/tempoGAN/tempoGAN.py:205:244
--8<--
```

各个参数含义与[Generator 的约束](#361-generator)相同。

### 3.7 可视化器构建

因为 GAN 网络训练的特性，本问题不使用 PaddleScience 中内置的可视化器，而是自定义了一个用于实现推理的函数，该函数读取验证集数据，得到推理结果并将结果以图片形式保存下来，在训练过程中按照一定间隔调用该函数即可在训练过程中监控训练效果。

``` py linenums="154"
--8<--
examples/tempoGAN/functions.py:154:230
--8<--
```

### 3.8 自定义 loss 和 data transform

由于本问题采用无监督学习，数据中不存在标签数据，loss 为计算得到，因此需要自定义 loss 。方法为先定义相关函数，再将函数名作为参数传给 `FunctionalLoss`。需要注意自定义 loss 函数的输入输出参数需要与 PaddleScience 中如 `MSE` 等其他函数保持一致，即输入为模型输出 `output_dict` 等字典变量，输出为 loss 值 `paddle.Tensor`。

#### 3.8.1 Generator 的 loss

Generator 的 loss 提供了 l1 loss、l2 loss、输出经过 Discriminator 判断的 loss 和 输出经过 Discriminator_tempo 判断的 loss。这些 loss 是否存在根据权重参数控制，若某一项 loss 的权重参数为 0，则表示训练中不添加该 loss 项。

``` py linenums="277"
--8<--
examples/tempoGAN/functions.py:277:346
--8<--
```

#### 3.8.2 Discriminator 的 loss

Discriminator 为判别器，它的作用是判断数据为真数据还是假数据，因此它的 loss 为 Generator 产生的数据应当判断为假而产生的 loss 和 目标值数据应当判断为真而产生的 loss。

``` py linenums="396"
--8<--
examples/tempoGAN/functions.py:396:410
--8<--
```

#### 3.8.3 Discriminator_tempo 的 loss

Discriminator_tempo 的 loss 构成 与 Discriminator 相同，只是所需数据不同。

``` py linenums="412"
--8<--
examples/tempoGAN/functions.py:412:428
--8<--
```

#### 3.8.4 自定义 data transform

本问题提供了一种输入数据处理方法，将输入的流体密度数据随机裁剪一块，然后进行密度值判断，若裁剪下来的块密度值低于阈值则重新裁剪，直到密度满足条件或裁剪次数达到阈值。这样做主要是为了减少训练所需的显存，同时对裁剪下来的块密度值的判断保证了块中信息的丰富程度。[参数和超参数设定](#34)中 `tile_ratio` 表示原始尺寸是块的尺寸的几倍，即若`tile_ratio` 为 2，裁剪下来的块的大小为整张原始图片的四分之一。

``` py linenums="431"
--8<--
examples/tempoGAN/functions.py:431:489
--8<--
```

注意，此处代码仅提供 data transform 的思路。当前代码中简单的分块方法由于输入包含的信息变少，显然会影响训练效果，因此本问题中当显存充足时，应当将 `tile_ratio` 设置为 1，当显存不足时，也建议优先考虑使用混合精度训练来减少现存占用。

### 3.9 模型训练

完成上述设置之后，首先需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练。

``` py linenums="247"
--8<--
examples/tempoGAN/tempoGAN.py:247:258
--8<--
```

注意 GAN 类型的网络训练方法为多个模型交替训练，与单一模型或多模型分阶段训练不同，不能简单的使用 `solver.train` API，具体代码请参考 [完整代码](#4) 中 tempoGAN.py 文件。

### 3.10 模型评估

#### 3.10.1 训练中评估

训练中仅在特定 `Epoch` 保存特定图片的目标结果和模型输出结果，训练结束后针对最后一个 `Epoch` 的输出结果进行一次评估，以便直观评价模型优化效果。不使用 PaddleScience 中内置的评估器，也不在训练过程中进行评估:

``` py linenums="287"
--8<--
examples/tempoGAN/tempoGAN.py:287:293
--8<--
```

``` py linenums="307"
--8<--
examples/tempoGAN/tempoGAN.py:307:323
--8<--
```

具体代码请参考 [完整代码](#4) 中 tempoGAN.py 文件。

#### 3.10.2 eval 中评估

本问题的评估指标为，将模型输出的超分结果与实际高分辨率图片做对比，使用三个指标 MSE(Mean-Square Error) 、PSNR(Peak Signal-to-Noise Ratio) 、SSIM(Structural SIMilarity) 来评价图片相似度。因此没有使用 PaddleScience 中的内置评估器，也没有 `Solver.eval()` 过程。

``` py linenums="326"
--8<--
examples/tempoGAN/tempoGAN.py:326:406
--8<--
```

另外，其中：

``` py linenums="396"
--8<--
examples/tempoGAN/tempoGAN.py:396:403
--8<--
```

提供了保存模型输出结果的选择，以便更直观的看出超分后的结果，是否开启由配置文件 `EVAL` 中的 `save_outs` 指定：

``` yaml linenums="91"
--8<--
examples/tempoGAN/conf/tempogan.yaml:91:94
--8<--
```

## 4. 完整代码

完整代码包含 PaddleScience 具体训练流程代码 tempoGAN.py 和所有自定义函数代码 functions.py，另外还向 `ppsci.arch` 添加了网络结构代码 gan.py，一并显示在下面，如果需要自定义网络结构，可以作为参考。

``` py linenums="1" title="tempoGAN.py"
--8<--
examples/tempoGAN/tempoGAN.py
--8<--
```

``` py linenums="1" title="functions.py"
--8<--
examples/tempoGAN/functions.py
--8<--
```

``` py linenums="1" title="gan.py"
--8<--
ppsci/arch/gan.py
--8<--
```

## 5. 结果展示

使用混合精度训练后，在测试集上评估与目标之间的 MSE、PSNR、SSIM，评估指标的值为：

| MSE | PSNR | SSIM |
| :---: | :---: | :---: |
| 4.21e-5 | 47.19 | 0.9974 |

一个流体超分样例的输入、模型预测结果、[数据集介绍](#31)中开源代码包 mantaflow 直接生成的结果如下，模型预测结果与生成的目标结果基本一致。

<figure markdown>
  ![input](https://paddle-org.bj.bcebos.com/paddlescience/docs/tempoGAN/input.gif){ loading=lazy }
  <figcaption>输入的低密度流体</figcaption>
</figure>

<figure markdown>
  ![pred-amp02](https://paddle-org.bj.bcebos.com/paddlescience/docs/tempoGAN/pred_amp02.gif){ loading=lazy }
  <figcaption>混合精度训练后推理得到的高密度流体</figcaption>
</figure>

<figure markdown>
  ![target](https://paddle-org.bj.bcebos.com/paddlescience/docs/tempoGAN/target.gif){ loading=lazy }
  <figcaption> 目标高密度流体</figcaption>
</figure>

## 6. 参考文献

- [tempoGAN: A Temporally Coherent, Volumetric GAN for Super-resolution Fluid Flow](https://dl.acm.org/doi/10.1145/3197517.3201304)

- [参考代码](https://github.com/thunil/tempoGAN)
