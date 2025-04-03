# A Transformer Model for Symbolic regression towards Scientific Discovery

=== "模型训练命令"

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer4sr/data_generated.tar
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer4sr/data_generated.tar -o data_generated.tar
    # unzip it
    tar -xvf data_generated.tar
    python transformer4sr.py
    ```

=== "模型评估命令"

    ``` sh
    # download srsd dataset from huggingface
    git clone https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_easy/
    git clone https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_medium/
    git clone https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_hard/
    # running
    python transformer4sr.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/transformer4sr/transformer4sr_pretrained.pdparams
    ```

=== "模型导出命令"

    ``` sh
    python transformer4sr.py mode=export
    ```

=== "模型推理命令"

    ``` sh
    python transformer4sr.py mode=infer
    ```

| 预训练模型  | 指标 |
|:--| :--|
| [transformer4sr_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/transformer4sr/transformer4sr_pretrained.pdparams) | Mean ZSS distance(srsd-feynman_easy): 0.658 +- 0.390<br>Hit rate(srsd-feynman_easy): 8/30<br>Mean ZSS distance(srsd-feynman_medium): 0.674 +- 0.331<br>Hit rate(srsd-feynman_medium): 8/37<br>Mean ZSS distance(srsd-feynman_hard): 0.737 +- 0.188<br>Hit rate(srsd-feynman_hard): 1/39 |

## 1. 背景简介

符号回归（SR）搜索最能描述数值数据集的数学表达式，它大致分为三类，即 GP-based SR（基于遗传编程的符号回归）、ML-based SR（基于机器学习的符号回归）、DL-based SR（基于深度学习的符号回归）。基于遗传编程的 SR 算法的计算成本通常很高，因此该案例使用了一种针对符号回归的新 Transformer 模型，并将最佳模型应用于 SRSD 数据集（科学发现数据集的符号回归）进行推理和测试。

## 2. 问题定义

作者提出了一种基于 Transformer 网络的 SR 模型，称为 Transformer4SR (A Transformer Model for Symbolic regression towards Scientific Discovery)，该模型用于处理封闭库问题，将符号回归的预定义词汇用特定的方法转化为 tokens。在该案例中，输入数据通过程序生成，数据通过模型后得到输出结果，再将其转化回符号表示，最终得到数据的符号回归结果。

下图为该方法的网络结构图，该结构基于 Transformer 有编码器 Encoder 和解码器 Decoder 两个主要部分。
作者提出三种编码器架构：MLP、Att 或 Mix，在本案例中，主要实现了 Mix 下的编码器结构。解码器是标准的 Transformer 解码器。在训练期间，编码器接收表格数据集，解码器接收 tokens 的真实值序列，而在推理过程中，解码器是独立的，并以自动回归的方式预测 tokens。

<figure markdown>
  ![model](https://paddle-org.bj.bcebos.com/paddlescience/docs/transformer4sr/transformer4sr_model.png){ loading=lazy }
  <figcaption>transformer4sr 模型结构图</figcaption>
</figure>

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 数据集生成与下载

#### 3.1.1 数据生成

该案例使用的训练数据为案例自主生成的数据：先用符号库中的符号随机生成大量公式；再通过一定的筛选机制筛除无效或不合规的公式；最后再根据这些公式进行采样；最终得到输入数据集和标签数据。

生成数据的参数信息如下：

``` yaml linenums="32"
--8<--
examples/transformer4sr/conf/transformer4sr.yaml:32:48
--8<--
```

其中`num_init_trials`为随机产生的初始方程数，这个值越大生成的数据越多，在原论文中这个值为 1000000。

设置相关参数后，可使用如下命令生成数据集：

``` python
python generate_datasets.py
```

#### 3.1.2 数据下载

我们也提前生成了一个比原始训练数据规模小 10 倍的数据集（即`num_init_trials`为 100000），以便简单的进行模型训练，并提供了下载链接：

``` sh
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer4sr/data_generated.tar
tar -xvf data_generated.tar
```

该案例在开源符号回归数据集 [SRSD](https://arxiv.org/abs/2206.10540)(Rethinking Symbolic Regression Datasets and Benchmarks for Scientific Discovery) 上进行模型验证，因此需要预下载此数据集，该数据集存放在 huggingface 上，地址分别为[srsd-feynman_easy](https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_easy)、[srsd-feynman_medium](https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_medium)、[srsd-feynman_hard](https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_hard)。可从对应网页下载或使用 git 下载：

``` sh
git clone https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_easy/
git clone https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_medium/
git clone https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_hard/
```

### 3.2 数据读取

由于数据读取和转换比较复杂，数据集相关的函数被定义在 functions_data.py 文件中，并在模型训练、验证等过程中被调用。

训练时需要读取自主生成的数据集：

``` py linenums="37"
--8<--
examples/transformer4sr/transformer4sr.py:37:45
--8<--
```

验证时需要读取 SRSD 数据集：

``` py linenums="134"
--8<--
examples/transformer4sr/transformer4sr.py:134:143
--8<--
```

数据相关参数定义在 yaml 文件中：

``` yaml linenums="50"
--8<--
examples/transformer4sr/conf/transformer4sr.yaml:50:58
--8<--
```

### 3.3 模型构建

在本问题中，我们使用神经网络 `Transformer` 作为模型。

``` py linenums="46"
--8<--
examples/transformer4sr/transformer4sr.py:46:55
--8<--
```

为了在计算时，准确快速地访问具体变量的值，我们在这里指定网络模型的输入变量名是 `("input", "target_seq")`，输出变量名是 `("output", )`，这些命名与后续代码保持一致。

### 3.4 优化器构建

本案例使用一种自定义的学习率策略 `LambdaDecay`，该学习率策略支持自定义学习率衰减函数。训练过程会调用优化器来更新模型参数，此处选择较为常用的 `Adam` 优化器。

``` py linenums="56"
--8<--
examples/transformer4sr/transformer4sr.py:56:68
--8<--
```

### 3.5 约束构建

在本案例中，我们使用监督数据集对模型进行训练，因此需要构建监督约束 `SupervisedConstraint`：

``` py linenums="69"
--8<--
examples/transformer4sr/transformer4sr.py:69:91
--8<--
```

`SupervisedConstraint` 的第一个参数是监督约束的读取配置，其中 `dataset` 字段表示使用的训练数据集信息，各个字段分别表示：

1. `name`： 数据集类型，此处 `NamedArrayDataset` 表示数据集的类型为 Array；
2. `input`： 输入数据；
3. `label`： 标签数据；

`batch_size` 字段表示 batch的大小；

`sampler` 字段表示采样方法，其中各个字段表示：

1. `name`： 采样器类型，此处 `BatchSampler` 表示批采样器；
2. `drop_last`： 是否需要丢弃最后无法凑整一个 mini-batch 的样本；
3. `shuffle`： 是否需要在生成样本下标时打乱顺序；

第二个参数是损失函数，此处的 `FunctionalLoss` 为 PaddleScience 自定义 loss 函数类，该类支持编写代码时自定义 loss 的计算方法，loss 的具体实现为其参数 `cross_entropy_loss_func`， 这是一个被定义在 functions_loss_metric.py 文件中的函数，如下所示：

``` py linenums="26"
--8<--
examples/transformer4sr/functions_loss_metric.py:26:30
--8<--
```

第三个参数是约束条件的名字，我们需要给每一个约束条件命名，方便后续对其索引。

在约束构建完毕之后，以我们刚才的命名为关键字，封装到一个字典中，方便后续访问：

``` py linenums="92"
--8<--
examples/transformer4sr/transformer4sr.py:92:94
--8<--
```

### 3.6 评估器构建

在训练过程中通常会按一定轮数间隔，用验证集(测试集)评估当前模型的训练情况，因此使用 `ppsci.validate.SupervisedValidator` 构建评估器，构建过程与 [约束构建](#35) 类似。

``` py linenums="95"
--8<--
examples/transformer4sr/transformer4sr.py:95:116
--8<--
```

其中评估指标为自定义的指标计算函数 `compute_inaccuracy`，在 functions_loss_metric.py 文件中：

``` py linenums="32"
--8<--
examples/transformer4sr/functions_loss_metric.py:32:48
--8<--
```

### 3.7 超参数设定

设置训练轮数等参数，如下所示。

``` yaml linenums="71"
--8<--
examples/transformer4sr/conf/transformer4sr.yaml:71:77
--8<--
```

### 3.8 模型训练、评估

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估。

``` py linenums="117"
--8<--
examples/transformer4sr/transformer4sr.py:117:131
--8<--
```

### 3.9 模型验证与结果可视化

该案例在开源符号回归数据集 [SRSD](https://arxiv.org/abs/2206.10540)(Rethinking Symbolic Regression Datasets and Benchmarks for Scientific Discovery) 上进行模型验证，验证时采用自回归的方式运行解码器，不需要运行编码器。验证时的指标为一种基于树编辑距离的归一化指标 ZSS。

``` py linenums="157"
--8<--
examples/transformer4sr/transformer4sr.py:157:183
--8<--
```

可视化的代码定义在文件 functions_vis.py 中，除了对验证集中的结果进行可视化外，还提供了一个公式为 $25*x1+x2*log(x1)$ 的 demo 的可视化结果：

``` py linenums="205"
--8<--
examples/transformer4sr/transformer4sr.py:205:209
--8<--
```

## 4. 完整代码

``` py linenums="1" title="transformer4sr.py"
--8<--
examples/transformer4sr/transformer4sr.py
--8<--
```

## 5. 结果展示

下方展示了模型在公式 $25*x1+x2*log(x1)$ 上的预测结果。

<figure markdown>
  ![res_demo](https://paddle-org.bj.bcebos.com/paddlescience/docs/transformer4sr/res_demo.png){ loading=lazy }
  <figcaption>demo 公式上的预测结果</figcaption>
</figure>

其中 $C$ 表示常量，可以看到模型预测结果与真实公式基本一致。

## 6. 参考资料

* [A Transformer Model for Symbolic regression towards Scientific Discovery](https://arxiv.org/abs/2312.04070)
* [参考代码](https://github.com/omron-sinicx/transformer4sr)

```
@inproceedings{lalande2023,
    title = {A Transformer Model for Symbolic Regression towards Scientific Discovery},
    author = {Florian Lalande and Yoshitomo Matsubara and Naoya Chiba and Tatsunori Taniai and Ryo Igarashi and Yoshitaka Ushiku},
    booktitle = {NeurIPS 2023 AI for Science Workshop},
    year = {2023},
    url = {https://openreview.net/forum?id=AIfqWNHKjo},
}
```
