# Volterra integral equation

<a href="TODO" class="md-button md-button--primary" style>AI Studio快速体验</a>

## 1. 问题简介

Volterra integral equation(沃尔泰拉积分方程)是一种积分方程，即方程中含有对待求解函数的积分运算，其有两种形式，如下所示

$$
\begin{aligned}
  f(t) &= \int_a^t K(t, s) x(s) d s \\
  x(t) &= f(t)+\int_a^t K(t, s) x(s) d s
\end{aligned}
$$

本案例以第二种方程为例，使用深度学习的方式进行求解。

## 2. 问题定义

假设存在如下 IDE 方程：

$$
u(t) = -\dfrac{du}{dt} + \int_{t_0}^t e^{t-s} u(s) d s
$$

其中 $u(t)$ 就是待求解的函数，而 $-\dfrac{du}{dt}$ 对应了 $f(t)$，$e^{t-s}$ 对应了 $K(t,s)$。
因此可以利用神经网络模型，以 $t$ 为输入，$u(t)$ 为输出，根据上述方程构建微分约束，进行无监督学习最终拟合出待求解的函数 $u(t)$。

为了方便在计算机中进行求解，我们将上式进行移项，让积分项作为右侧，非积分项移动到左侧，如下所示：

$$
u(t) + \dfrac{du}{dt} = \int_{t_0}^t e^{t-s} u(s) d s
$$

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 模型构建

在上述问题中，我们确定了输入为 $t$，输出为 $u(t)$，因此我们使用，用 PaddleScience 代码表示如下：

``` py linenums="37"
--8<--
examples/ide/volterra_ide.py:37:38
--8<--
```

为了在计算时，准确快速地访问具体变量的值，我们在这里指定网络模型的输入变量名是 `"x"`，输出变量名是 `"u"`，接着通过指定 `MLP` 的隐藏层层数、神经元个数，我们就实例化出了神经网络模型 `model`。

### 3.2 计算域构建

Volterra_IDE 问题的积分域是 $a$ ~ $t$，其中 `a` 为固定常数 0，`t` 的范围为 0 ~ 5，因此可以使用PaddleScience 内置的一维几何 `TimeDomain` 作为计算域。

``` py linenums="40"
--8<--
examples/ide/volterra_ide.py:40:42
--8<--
```

### 3.3 方程构建

由于 Volterra_IDE 使用的是积分方程，因此可以直接使用 PaddleScience 内置的 `ppsci.equation.Volterra`，并指定所需的参数：积分下限 `a`、`t` 的离散取值点数 `num_points`、一维高斯积分点的个数 `quad_deg`、$K(t,s)$ 核函数 `kernel_func`、$u(t) - f(t)$ 等式右侧表达式 `func`。

``` py linenums="44"
--8<--
examples/ide/volterra_ide.py:44:64
--8<--
```

### 3.4 约束构建

#### 3.4.1 内部点约束

本文采用无监督学习的方式，对移项后方程的左、右两侧进行约束，让其尽量相等。

由于等式右侧涉及到积分计算（实际采用高斯积分近似计算），因此在 0 ~ 5 区间内采样出多个 `t_i` 点后，还需要计算其用于高斯积分的点集，即对每一个 `(0,t_i)` 区间，都计算出一一对应的高斯积分点集 `quad_i` 和点权 `weight_i`。PaddleScience 将这一步作为输入数据的预处理，加入到代码中，如下所示

``` py linenums="66"
--8<--
examples/ide/volterra_ide.py:66:108
--8<--
```

#### 3.4.2 初值约束

在 $t=0$ 时，有以下初值条件：

$$
u(0) = e^{-t} \cosh(t)|_{t=0} = e^{0} \cosh(0) = 1
$$

因此可以加入 `t=0` 时的初值条件，代码如下所示

``` py linenums="110"
--8<--
examples/ide/volterra_ide.py:110:128
--8<--
```

在微分方程约束、初值约束构建完毕之后，以我们刚才的命名为关键字，封装到一个字典中，方便后续访问。

``` py linenums="129"
--8<--
examples/ide/volterra_ide.py:129:133
--8<--
```

### 3.5 超参数设定

接下来我们需要指定训练轮数和学习率，此处我们按实验经验，让 `L-BFGS` 优化器进行一轮优化即可，但一轮优化内的 `max_iters` 数可以设置为一个较大的一个数 `15000`。

``` py linenums="135"
--8<--
examples/ide/volterra_ide.py:135:136
--8<--
```

### 3.6 优化器构建

训练过程会调用优化器来更新模型参数，此处选择较为常用的 `LBFGS` 优化器。

``` py linenums="138"
--8<--
examples/ide/volterra_ide.py:138:146
--8<--
```

### 3.7 评估器构建

在训练过程中通常会按一定轮数间隔，用验证集（测试集）评估当前模型的训练情况，因此使用 `ppsci.validate.GeometryValidator` 构建评估器。

``` py linenums="148"
--8<--
examples/ide/volterra_ide.py:148:163
--8<--
```

评价指标 `metric` 选择 `ppsci.metric.L2Rel` 即可。

其余配置与 [3.4 约束构建](#34) 的设置类似。

### 3.8 模型训练

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练。

``` py linenums="165"
--8<--
examples/ide/volterra_ide.py:165:181
--8<--
```

### 3.9 结果可视化

在模型训练完毕之后，我们可以手动构造 0 ~ 5 区间内均匀 100 个点作为评估的积分上限 `t` 作为输入数据进行预测，并可视化结果。

``` py linenums="183"
--8<--
examples/ide/volterra_ide.py:183:
--8<--
```

## 4. 完整代码

``` py linenums="1" title="volterra_ide.py"
--8<--
examples/ide/volterra_ide.py
--8<--
```

## 5. 结果展示

<figure markdown>
  ![result](../../images/volterra/Volterra_IDE.png){ loading=lazy }
</figure>

## 6. 参考文献

- [DeepXDE - Antiderivative operator from an unaligned dataset](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Volterra_IDE.py)
- [Gaussian quadrature](https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval)
- [Volterra integral equation](https://en.wikipedia.org/wiki/Volterra_integral_equation)
