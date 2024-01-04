# NSFNets

<a href="https://aistudio.baidu.com/projectdetail/7305373" class="md-button md-button--primary" style>AI Studio快速体验</a>

=== "模型评估命令"

    ``` sh
    # VP_NSFNet1
    python VP_NSFNet1.py    mode=eval  pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/nsfnet/nsfnet1.pdparams

    # VP_NSFNet2
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/NSFNet/cylinder_nektar_wake.mat -P ./data/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/NSFNet/cylinder_nektar_wake.mat --output ./data/cylinder_nektar_wake.mat

    python VP_NSFNet2.py    mode=eval  data_dir=./data/cylinder_nektar_wake.mat  pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/nsfnet/nsfnet2.pdparams

    # VP_NSFNet3
    python VP_NSFNet3.py    mode=eval  pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/nsfnet/nsfnet3.pdparams
    ```

=== "模型训练命令"

    ``` sh
    # VP_NSFNet1
    python VP_NSFNet1.py

    # VP_NSFNet2
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/NSFNet/cylinder_nektar_wake.mat -P ./data/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/NSFNet/cylinder_nektar_wake.mat --output ./data/cylinder_nektar_wake.mat
    python VP_NSFNet2.py data_dir=./data/cylinder_nektar_wake.mat

    # VP_NSFNet3
    python VP_NSFNet3.py
    ```


## 1. 背景简介
 最近几年,深度学习在很多领域取得了非凡的成就,尤其是计算机视觉和自然语言处理方面,而受启发于深度学习的快速发展,基于深度学习强大的函数逼近能力,神经网络在科学计算领域也取得了成功,现阶段的研究主要分为两大类,一类是将物理信息以及物理限制加入损失函数来对神经网络进行训练, 其代表有 PINN 以及 Deep Retz Net,另一类是通过数据驱动的深度神经网络算子,其代表有 FNO 以及 DeepONet。这些方法都在科学实践中获得了广泛应用,比如天气预测,量子化学,生物工程,以及计算流体等领域。而为充分探索PINN对流体方程的求解能力,本次复现[论文](https://arxiv.org/abs/2003.06496)作者设计了NSFNets，并且先后使用具有解析解或数值解的二维、三维纳韦斯托克方程以及使用DNS方法进行高精度求解的数据集作为参考, 进行正问题求解训练。论文实验表明PINN对不可压纳韦斯托克方程具有优秀的数值求解能力, 本项目主要目标是使用PaddleScience复现论文所实现的高精度求解纳韦斯托克方程的代码。
## 2. 问题定义
本问题所使用的为最经典的PINN模型,对此不再赘述。

主要介绍所求解的几类纳韦斯托克方程：

不可压纳韦斯托克方程可以表示为：

$$\frac{\partial \mathbf{u}}{\partial t}+(\mathbf{u} \cdot \nabla) \mathbf{u} =-\nabla p+\frac{1}{Re} \nabla^2 \mathbf{u} \quad \text { in } \Omega,$$

$$\nabla \cdot \mathbf{u} =0 \quad  \text { in } \Omega,$$

$$\mathbf{u} =\mathbf{u}_{\Gamma} \quad \text { on } \Gamma_D,$$

$$\frac{\partial \mathbf{u}}{\partial n} =0 \quad \text { on } \Gamma_N.$$

### 2.1 Kovasznay flow(NSFNet1)
我们使用 Kovasznay 流作为第一个测试用例来演示 NSFnets 的性能。 该二维稳态纳维-斯托克斯流具有以下解析解:

$$u(x, y)=1-e^{\lambda x} \cos (2 \pi y),$$

$$v(x, y)=\frac{\lambda}{2 \pi} e^{\lambda x} \sin (2 \pi y),$$

$$p(x, y)=\frac{1}{2}\left(1-e^{2 \lambda x}\right),$$

其中

$$\lambda=\frac{1}{2 \nu}-\sqrt{\frac{1}{4 \nu^2}+4 \pi^2}, \quad \nu=\frac{1}{Re}=\frac{1}{40} .$$

我们考虑计算域为 $[−0.5, 1.0] × [−0.5, 1.5]$。 我们首先确定优化策略。 每个边界上有 $101$ 个具有固定空间坐标的点,即 $Nb = 4 × 101$。为了计算 NSFnet 的方程损失,在域内随机选择 $2,601$ 个点。 这种稳定流动没有初始条件。 我们使用 Adam 优化器来提供一组更好的初始神经网络可学习变量。 然后,使用L-BFGS-B对神经网络进行微调以获得更高的精度。 L-BFGS-B的训练过程根据增量容差自动终止。 在本节中,我们在 L-BFGS-B 训练之前使用 $3 × 10^4$ Adam 迭代,学习率为 $10^{−3}$。 Adam 迭代次数的影响在论文附录 A 的图 A.1 中讨论,我们还研究了 NSFnet 在采样点和边界点数量方面的性能。
### 2.2 Cylinder wake (NSFNet2)
这里我们使用 NSFnets 模拟 $Re = 100$ 时圆柱体后面的 $2D$ 涡旋脱落。圆柱体放置在 $(x, y) = (0, 0)$ 处,直径 $D = 1$。高保真 DNS 数据来自 [$M. Raissi 2019$](https://www.sciencedirect.com/science/article/am/pii/S0021999118307125) 用作参考并为 NSFnet 训练提供边界和初始数据。 我们考虑由 $[1, 8] × [−2, 2]$ 定义的域,时间间隔为 $[0, 7]$（超过一个脱落周期）,时间步长 $Δt = 0.1$。 对于训练数据,我们沿 $x$ 方向边界放置 $100$ 个点,沿 y 方向边界放置 $50$ 个点来控制边界条件,并使用域内的 $140,000$ 个时空分散点来计算残差。 NSFnet 包含 $10$ 个隐藏层,每层有 $100$ 个神经元。[Cylinder wake AIstudio数据集链接](https://aistudio.baidu.com/datasetdetail/236213)。
### 2.3 Beltrami flow (NSFNet3)
$$u(x, y, z, t)= -a\left[e^{a x} \sin (a y+d z)+e^{a z} \cos (a x+d y)\right] e^{-d^2 t}, $$

$$v(x, y, z, t)= -a\left[e^{a y} \sin (a z+d x)+e^{a x} \cos (a y+d z)\right] e^{-d^2 t}, $$

$$w(x, y, z, t)= -a\left[e^{a z} \sin (a x+d y)+e^{a y} \cos (a z+d x)\right] e^{-d^2 t}, $$

$$p(x, y, z, t)= -\frac{1}{2} a^2\left[e^{2 a x}+e^{2 a y}+e^{2 a z}+2 \sin (a x+d y) \cos (a z+d x) e^{a(y+z)} +2 \sin (a y+d z) \cos (a x+d y) e^{a(z+x)} +2 \sin (a z+d x) \cos (a y+d z) e^{a(x+y)}\right] e^{-2 d^2 t}.$$  
## 3. 问题求解
### 3.1 模型构建
本文使用PINN经典的MLP模型进行训练。
``` py linenums="175"
--8<--
examples/nsfnet/VP_NSFNet3.py:175:175
--8<--
```
### 3.2 超参数设定
指定残差点、边界点、初值点的个数,以及可以指定边界损失函数和初值损失函数的权重
``` py linenums="178"
--8<--
examples/nsfnet/VP_NSFNet3.py:178:186
--8<--
```
### 3.3 数据生成
因数据集为解析解,我们先构造解析解函数
``` py linenums="10"
--8<--
examples/nsfnet/VP_NSFNet3.py:10:51
--8<--
```

然后先后取边界点、初值点、以及用于计算残差的内部点（具体取法见[论文](https://arxiv.org/abs/2003.06496)节3.3）以及生成测试点。
``` py linenums="187"
--8<--
examples/nsfnet/VP_NSFNet3.py:187:214
--8<--
```
### 3.4 约束构建
由于我们边界点和初值点具有解析解,因此我们使用监督约束
``` py linenums="266"
--8<--
examples/nsfnet/VP_NSFNet3.py:266:277
--8<--
```

其中alpha和beta为该损失函数的权重,在本代码中与论文中描述一致,都取为100

使用内部点构造纳韦斯托克方程的残差约束
``` py linenums="280"
--8<--
examples/nsfnet/VP_NSFNet3.py:280:297
--8<--
```
### 3.5 评估器构建
使用在数据生成时生成的测试点构造的测试集用于模型评估：
``` py linenums="305"
--8<--
examples/nsfnet/VP_NSFNet3.py:305:319
--8<--
```

### 3.6 优化器构建
与论文中描述相同,我们使用分段学习率构造Adam优化器,其中可以通过调节_epoch_list_来调节训练轮数。
``` py linenums="321"
--8<--
examples/nsfnet/VP_NSFNet3.py:321:331
--8<--
```

### 3.7 模型训练与评估
完成上述设置之后,只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`。

``` py linenums="333"
--8<--
examples/nsfnet/VP_NSFNet3.py:333:350
--8<--
```

最后启动训练即可：

``` py linenums="351"
--8<--
examples/nsfnet/VP_NSFNet3.py:351:352
--8<--
```


## 4. 完整代码
NSFNet1:
``` py linenums="1" title="NSFNet1.py"
--8<--
examples/nsfnet/VP_NSFNet1.py
--8<--
```
NSFNet2:
``` py linenums="1" title="NSFNet2.py"
--8<--
examples/nsfnet/VP_NSFNet2.py
--8<--
```
NSFNet3:
``` py linenums="1" title="NSFNet3.py"
--8<--
examples/nsfnet/VP_NSFNet3.py
--8<--
```
## 5. 结果展示

主要参考论文数据，和参考代码的数据。

### 5.1 NSFNet1(Kovasznay flow)

| velocity | paper | code | PaddleScience | NN size |
|:--|:--|:--|:--|:--|
| u | 0.072% | 0.080% | 0.056% | 4 × 50 |
| v | 0.058% | 0.539% | 0.399% | 4 × 50 |
| p | 0.027% | 0.722% | 1.123% | 4 × 50 |

如表格所示,第2,3,4列分别为论文,其他开发者和PaddleScience复现的$L_{2}$误差Kovasznay flow在$x$, $y$方向的速度$u$, $v$的$L_{2}$误差为0.055%和0.399%, 指标均优于论文(Table 2)和参考代码。
### 5.2 NSFNet2(Cylinder wake)
Cylinder wake在$t=0$时刻预测的$L_{2}$误差, 如表格所示, Cylinder flow在$x$, $y$方向的速度$u$, $v$的$L_{2}$误差为0.138%和0.488%, 指标接近论文(Figure 9)和代码。

| velocity | paper (VP-NSFnet, $\alpha=\beta=1$) | paper (VP-NSFnet, dynamic weights)  | code | PaddleScience  | NN size |
|:--|:--|:--|:--|:--|:--|
| u | 0.09% | 0.01% | 0.403% | 0.138% | 4 × 50 |
| v | 0.25% | 0.05% | 1.5%   | 0.488% | 4 × 50 |
| p | 1.9%  | 0.8%  |  /     | /      | 4 × 50 |

NSFNet2(2D Cylinder Flow)案例的速度场如下图所示, 第一行的两张图片为圆柱尾部绕流区域, 第一行的图片表示在$x$流线方向上的流速$u$的数值分布, 左侧为DNS高保真数据作为参考, 右侧为神经网络预测值, 蓝色为较小值, 绿色为较大值, 分布区域为 $x=[1,8]$, $y=[-2, 2]$, 第二行的图片表示在$y$展向方向上的流速$v$的分布,左侧为DNS高保真数据参考值, 右侧为神经网络预测值, 分布区域为 $x=[1,8]$, $y=[-2, 2]$。

![image](https://paddle-org.bj.bcebos.com/paddlescience/docs/NSFNet/Cylinder%20wake.gif)

根据速度场，我们可以计算涡流场, 如图所示, 为NSFNet2(2D Cylinder Flow)案例在$t=4.0$时刻的涡流场的等值线图, 我们根据$x$, $y$方向的流速$u$, $v$,通过涡量计算公式, 计算得到如图所示涡量图, 涡结构连续性好, 和论文一致, 计算分布区域为$x=[1, 8]$, $y=[-2, 2]$。

![image](https://paddle-org.bj.bcebos.com/paddlescience/docs/NSFNet/NSFNet2_vorticity.png)
### 5.3 NSFNet3(Beltrami flow)
测试数据集(解析解)相对误差如表格所示, Beltrami flow在$x$, $y$, $z$方向的速度$u$, $v$, $w$的$L_{2}$误差为0.059%, 0.082%和0.0732%, 优于代码数据。

| velocity |  code(NN size:10×100) | PaddleScience (NN size:10×100)|
|:--|:--|:--|
| u | 0.0766% | 0.059% |
| v | 0.0689% | 0.082% |
| w | 0.1090% | 0.073% |
| p | /       | /      |

Beltrami flow在 $ t=1 $ 时刻, $ z=0 $平面上的预测相对误差, 如表格所示, Beltrami flow在$x, y, z$方向的速度$u, v, w$的$L_{2}$误差为0.115%, 0.199%和0.217%, 压力$p$的$L_{2}$误差为0.1.986%, 均优于论文数据(Table 4. VP)。

| velocity | paper(NN size:7×50) | PaddleScience(NN size:10×100) |
|:--|:--|:--|
| u | 0.1634±0.0418% | 0.115% |
| v | 0.2185±0.0530% | 0.199% |
| w | 0.1783±0.0300% | 0.217% |
| p | 8.9335±2.4350% | 1.986% |

Beltrami flow速度场,如图所示,左侧为解析解参考值,右侧为神经网络预测值,蓝色为较小值,红色为较大值,分布区域为$x=[-1,1]$, $y=[-1, 1]$, 第一行为在$x$方向上的流速$u$的分布,第二行为在$y$方向上的流速$v$的分布,第三行为在$z$方向上流速$w$的分布。

![image](https://paddle-org.bj.bcebos.com/paddlescience/docs/NSFNet/Beltrami%20flow.gif)

## 6. 结果说明
我们使用PINN对不可压纳韦斯托克方程进行数值求解。在PINN中,随机选取的时间和空间的坐标被当作输入值,所对应的速度场以及压强场被当作输出值,使用初值、边界条件当作监督约束以及纳韦斯托克方程本身的当作无监督约束条件加入损失函数进行训练。我们针对三个不同类型的PINN纳韦斯托克方程, 设计了三个不同的流体案例, 即NSFNet1、NSFNet2、NSFNet3。通过损失函数的下降、网络预测结果与高保真DNS数据，以及解析解的$L_{2}$误差的降低，可以证明神经网络在求解纳韦斯托克方程中的收敛性, 表明NSFNets的架构拥有对不可压纳韦斯托克方程的求解能力。而通过实验结果表明, 三个使用NSFNet的正问题案例，都可以很好的逼近参考解, 并且我们发现增加边界约束, 以及初值约束的权重可以使得神经网络拥有更好的逼近效果。

## 7. 参考资料
[NSFnets (Navier-Stokes Flow nets): Physics-informed neural networks for the incompressible Navier-Stokes equations](https://arxiv.org/abs/2003.06496)

[Github NSFnets](https://github.com/Alexzihaohu/NSFnets/tree/master)
