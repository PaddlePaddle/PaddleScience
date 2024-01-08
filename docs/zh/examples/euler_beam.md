# Euler Beam

=== "模型训练命令"

    ``` sh
    python euler_beam.py
    ```

=== "模型评估命令"

    ``` sh
    python euler_beam.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/euler_beam/euler_beam_pretrained.pdparams
    ```

| 预训练模型  | 指标 |
|:--| :--|
| [euler_beam_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/euler_beam/euler_beam_pretrained.pdparams) | loss(L2Rel_Metric): 0.00000<br>L2Rel.u(L2Rel_Metric): 0.00080 |

## 1. 问题定义

Euler Beam 公式：

$$
\dfrac{\partial^{4} u}{\partial x^{4}} + 1 = 0, x \in [0, 1]
$$

边界条件：

$$
u''(1)=0, u'''(1)=0
$$

狄利克雷条件：

$$
u(0)=0
$$

诺依曼边界条件：

$$
u'(0)=0
$$

## 2. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 2.1 模型构建

在 Euler Beam 问题中，每一个已知的坐标点 $x$ 都有对应的待求解的未知量 $u$
，我们在这里使用比较简单的 MLP(Multilayer Perceptron, 多层感知机) 来表示 $x$ 到 $u$ 的映射函数 $f: \mathbb{R}^1 \to \mathbb{R}^1$ ，即：

$$
u = f(x)
$$

上式中 $f$ 即为 MLP 模型本身，用 PaddleScience 代码表示如下

``` py linenums="36"
--8<--
examples/euler_beam/euler_beam.py:36:37
--8<--
```

其中，用于初始化模型的参数通过配置文件进行配置：

``` yaml linenums="34"
--8<--
examples/euler_beam/conf/euler_beam.yaml:34:38
--8<--
```

接着通过指定 MLP 的层数、神经元个数，我们就实例化出了一个拥有 3 层隐藏神经元，每层神经元数为 20 的神经网络模型 `model`。

### 2.2 方程构建

Euler Beam 的方程构建可以直接使用 PaddleScience 内置的 `Biharmonic`，指定该类的参数 `dim` 为 1，`q` 为 -1，`D` 为1。

``` py linenums="42"
--8<--
examples/euler_beam/euler_beam.py:42:43
--8<--
```

### 2.3 计算域构建

本文中 Euler Beam 问题作用在以 (0.0, 1.0) 的一维区域上，
因此可以直接使用 PaddleScience 内置的空间几何 `Interval` 作为计算域。

``` py linenums="39"
--8<--
examples/euler_beam/euler_beam.py:39:40
--8<--
```

### 2.4 约束构建

在本案例中，我们使用了两个约束条件在计算域中指导模型的训练分别是作用于采样点上的方程约束和作用于边界点上的约束。

在定义约束之前，需要给每一种约束指定采样点个数，表示每一种约束在其对应计算域内采样数据的数量，以及通用的采样配置。

``` yaml linenums="49"
--8<--
examples/euler_beam/conf/euler_beam.yaml:49:50
--8<--
```

#### 2.4.1 内部点约束

以作用在内部点上的 `InteriorConstraint` 为例，代码如下：

``` py linenums="51"
--8<--
examples/euler_beam/euler_beam.py:51:59
--8<--
```

#### 2.4.2 边界约束

同理，我们还需要构建边界的约束。但与构建 `InteriorConstraint` 约束不同的是，由于作用区域是边界，因此我们使用 `BoundaryConstraint` 类，代码如下：

``` py linenums="60"
--8<--
examples/euler_beam/euler_beam.py:60:73
--8<--
```

### 2.5 超参数设定

接下来我们需要在配置文件中指定训练轮数，此处我们按实验经验，使用一万轮训练轮数，评估间隔为一千轮。

``` yaml linenums="41"
--8<--
examples/euler_beam/conf/euler_beam.yaml:41:52
--8<--
```

### 2.6 优化器构建

训练过程会调用优化器来更新模型参数，此处选择较为常用的 `Adam` 优化器。

``` py linenums="80"
--8<--
examples/euler_beam/euler_beam.py:80:81
--8<--
```

### 2.7 评估器构建

在训练过程中通常会按一定轮数间隔，用验证集（测试集）评估当前模型的训练情况，因此使用 `ppsci.validate.GeometryValidator` 构建评估器。

``` py linenums="89"
--8<--
examples/euler_beam/euler_beam.py:89:102
--8<--
```

### 2.8 可视化器构建

在模型评估时，如果评估结果是可以可视化的数据，我们可以选择合适的可视化器来对输出结果进行可视化。

本文中的输出数据是一个曲线图，因此我们只需要将评估的输出数据保存成 **png** 文件即可。代码如下：

``` py linenums="104"
--8<--
examples/euler_beam/euler_beam.py:104:114
--8<--
```

### 2.9 模型训练、评估与可视化

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估、可视化。

``` py linenums="117"
--8<--
examples/euler_beam/euler_beam.py:117:141
--8<--
```

## 3. 完整代码

``` py linenums="1" title="euler_beam.py"
--8<--
examples/euler_beam/euler_beam.py
--8<--
```

## 4. 结果展示

使用训练得到的模型对上述计算域中均匀取的共 `NPOINT_TOTAL` 个点 $x_i$ 进行预测，预测结果如下所示。图像中横坐标为 $x$，纵坐标为对应的预测结果 $u$。

<figure markdown>
  ![euler_beam](https://paddle-org.bj.bcebos.com/paddlescience/docs/euler_beam/euler_beam.png){ loading=lazy }
  <figcaption>模型预测结果</figcaption>
</figure>
