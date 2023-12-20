# NSFNet

<a href="https://aistudio.baidu.com/studio/project/partial/verify/6832363/da4e1b9b08f14bd4baf9b8b6922b5b7e" class="md-button md-button--primary" style>AI Studio快速体验</a>

=== "模型训练命令"

    ``` sh
    # VP_NSFNet1
    python VP_NSFNet1.py

    # VP_NSFNet2
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/NSFNet/cylinder_nektar_wake.mat -P ./npy_data/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/NSFNet/cylinder_nektar_wake.mat --output ./npy_data/cylinder_nektar_wake.mat
    python VP_NSFNet1.py DATASET_PATH=https://paddle-org.bj.bcebos.com/paddlescience/datasets/NSFNet/cylinder_nektar_wake.mat

    # VP_NSFNet3
    python VP_NSFNet3.py
    ```
=== "模型评估命令"

    ``` sh
    # VP_NSFNet1
    python VP_NSFNet1.py    mode=eval  EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/nsfnet/nsfnet1.pdparams

    # VP_NSFNet2
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/NSFNet/cylinder_nektar_wake.mat -P ./npy_data/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/NSFNet/cylinder_nektar_wake.mat --output ./npy_data/cylinder_nektar_wake.mat

    python VP_NSFNet2.py    mode=eval  DATASET_PATH=https://paddle-org.bj.bcebos.com/paddlescience/datasets/NSFNet/cylinder_nektar_wake.mat  EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/nsfnet/nsfnet2.pdparams

    # VP_NSFNet3
    python VP_NSFNet3.py    mode=eval  EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/nsfnet/nsfnet3.pdparams
    ```
## 1. 背景简介
 最近几年，深度学习在很多领域取得了非凡的成就，尤其是计算机视觉和自然语言处理方面，而受启发于深度学习的快速发展，基于深度学习强大的函数逼近能力，神经网络在科学计算领域也取得了成功，现阶段的研究主要分为两大类，一类是将物理信息以及物理限制加入损失函数来对神经网络进行训练, 其代表有 PINN 以及 Deep Retz Net，另一类是通过数据驱动的深度神经网络算子，其代表有 FNO 以及 DeepONet。这些方法都在科学实践中获得了广泛应用，比如天气预测，量子化学，生物工程，以及计算流体等领域。而为充分探索PINN对流体方程的求解能力，本次复现[论文](https://arxiv.org/abs/2003.06496)作者先后使用具有解析解或数值解的二维、三维纳韦斯托克方程以及使用DNS方法进行高进度求解的数据集进行训练。论文实验表明PINN对不可压纳韦斯托克方程具有优秀的数值求解能力，本项目主要目标是使用PaddleScience复现论文所实现的高精度求解纳韦斯托克方程的代码。
## 2. 问题定义
本问题所使用的为最经典的PINN模型，对此不再赘述。

主要介绍所求解的几类纳韦斯托克方程：

不可压纳韦斯托克方程可以表示为：

$$\frac{\partial \mathbf{u}}{\partial t}+(\mathbf{u} \cdot \nabla) \mathbf{u} =-\nabla p+\frac{1}{Re} \nabla^2 \mathbf{u} \quad \text { in } \Omega,$$

$$\nabla \cdot \mathbf{u} =0 \quad  \text { in } \Omega,$$

$$\mathbf{u} =\mathbf{u}_{\Gamma} \quad \text { on } \Gamma_D,$$

$$\frac{\partial \mathbf{u}}{\partial n} =0 \quad \text { on } \Gamma_N.$$

### 2.1 Kovasznay flow(NSFNet1)
$$u(x, y)=1-e^{\lambda x} \cos (2 \pi y),$$

$$v(x, y)=\frac{\lambda}{2 \pi} e^{\lambda x} \sin (2 \pi y),$$

$$p(x, y)=\frac{1}{2}\left(1-e^{2 \lambda x}\right),$$

其中

$$\lambda=\frac{1}{2 \nu}-\sqrt{\frac{1}{4 \nu^2}+4 \pi^2}, \quad \nu=\frac{1}{Re}=\frac{1}{40} .$$

### 2.2 Cylinder wake (NSFNet1)
该方程并没有解析解，为雷诺数为100的数值解，[AIstudio数据集](https://aistudio.baidu.com/datasetdetail/236213)。
### 2.3 Beltrami flow (NSFNet1)
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
指定残差点、边界点、初值点的个数，以及可以指定边界损失函数和初值损失函数的权重
``` py linenums="178"
--8<--
examples/nsfnet/VP_NSFNet3.py:178:186
--8<--
```
### 3.3 数据生成
因数据集为解析解，我们先构造解析解函数
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
由于我们边界点和初值点具有解析解，因此我们使用监督约束
``` py linenums="266"
--8<--
examples/nsfnet/VP_NSFNet3.py:266:277
--8<--
```

其中alpha和beta为该损失函数的权重，在本代码中与论文中描述一致，都取为100

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
examples/nsfnet/VP_NSFNet3.py:305:314
--8<--
```

### 3.6 优化器构建
与论文中描述相同，我们使用分段学习率构造Adam优化器，其中可以通过调节_epoch_list_来调节训练轮数。
``` py linenums="316"
--8<--
examples/nsfnet/VP_NSFNet3.py:316:325
--8<--
```

### 3.7 模型训练与评估
完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`。

``` py linenums="328"
--8<--
examples/nsfnet/VP_NSFNet3.py:328:345
--8<--
```

最后启动训练即可：

``` py linenums="348"
--8<--
examples/nsfnet/VP_NSFNet3.py:348:348
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

### NSFNet1(Kovasznay flow)
| velocity | paper  | code | PaddleScience  |
|:--|:--|:--|:--|
| u                 | 0.084% | 0.062%             | 0.055%  |
| v                 | 0.425% | 0.431%             | 0.399%  |

### NSFNet2(Cylinder wake)
Cylinder wake 在T=0 时刻预测的相对误差

| velocity | paper  | code | PaddleScience  |
|:--|:--|:--|:--|
| u                 | / | 0.403%           | 0.138%  |
| v                 | / | 1.5%             |  0.488% |

速度场

![image](https://paddle-org.bj.bcebos.com/paddlescience/docs/NSFNet/Cylinder%20wake.gif)

涡流场（t=4.0）

![image](https://paddle-org.bj.bcebos.com/paddlescience/docs/NSFNet/NSFNet2_vorticity.png)
### NSFNet3(Beltrami flow)
测试数据集相对误差

| velocity |  code | PaddleScience  |
|:--|:--|:--|
| u                 |  0.0766%        | 0.059%  |
| v                 |  0.0689%        | 0.082%  |
| w                 |   0.109%        | 0.0732%  |

Beltrami flow 在T=1时刻的预测相对误差

| velocity | code   | PaddleScience  |
|:--|:--|:--|
| u                 | 0.426%            | 0.115%  |
| v                 | 0.366%            | 0.199%  |
| w                 | 0.587%            | 0.217%  |

Beltrami flow速度场在
![image](https://paddle-org.bj.bcebos.com/paddlescience/docs/NSFNet/Beltrami%20flow.gif)

## 6. 结果说明
我们使用PINN对不可压纳韦斯托克方程进行数值求解。在PINN中，随机选取的时间和空间的坐标被当作输入值，所对应的速度场以及压强场被当作输出值，使用初值、边界条件当作监督约束以及纳韦斯托克方程本身的当作无监督约束条件加入损失函数进行训练。我们针对三个不同类型的PINN纳韦斯托克方程，设计了三个不同结构神经网络，即NSFNet1、NSFNet2、NSFNet3。通过损失函数的下降可以证明神经网络在求解纳韦斯托克方程中的收敛性，表明PINN拥有对不可压纳韦斯托克方程的求解能力。而通过实验结果表明，三个NSFNet方程都可以很好的逼近对应的纳韦斯托克方程，并且，我们发现增加边界约束以及初值约束的权重可以使得神经网络拥有更好的逼近效果。
## 7. 参考资料
[NSFnets (Navier-Stokes Flow nets): Physics-informed neural networks for the incompressible Navier-Stokes equations](https://arxiv.org/abs/2003.06496)

[Github NSFnets](https://github.com/Alexzihaohu/NSFnets/tree/master)
