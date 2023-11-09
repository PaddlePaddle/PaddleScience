# NSFNet 

<a href="https://aistudio.baidu.com/studio/project/partial/verify/6832363/da4e1b9b08f14bd4baf9b8b6922b5b7e" class="md-button md-button--primary" style>AI Studio快速体验</a>

=== "模型训练命令"

    ``` sh
    # VP_NSFNet1
    python VP_NSFNet1.py

    # VP_NSFNet2
    # linux
    wget 
    # windows
    # curl 
    python VP_NSFNet2.py

    # VP_NSFNet3
    python VP_NSFNet3.py
    ```

=== "模型评估命令"

    ``` sh
    # VP_NSFNet1
    python VP_NSFNet1_eval.py

    # VP_NSFNet2
    # linux
    wget 
    # windows
    # curl 
    python VP_NSFNet2_eval.py

    # VP_NSFNet3
    python VP_NSFNet3_eval.py
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

### 2.1 Kovasznay flow
$$u(x, y)=1-e^{\lambda x} \cos (2 \pi y),$$

$$v(x, y)=\frac{\lambda}{2 \pi} e^{\lambda x} \sin (2 \pi y),$$

$$p(x, y)=\frac{1}{2}\left(1-e^{2 \lambda x}\right),$$

其中
$$\lambda=\frac{1}{2 \nu}-\sqrt{\frac{1}{4 \nu^2}+4 \pi^2}, \quad \nu=\frac{1}{Re}=\frac{1}{40} .$$
### 2.2 Cylinder wake
该方程并没有解析解，为雷诺数为100的数值解，[AIstudio数据集](https://aistudio.baidu.com/datasetdetail/236213)。
### 2.3 Beltrami flow
$$u(x, y, z, t)= -a\left[e^{a x} \sin (a y+d z)+e^{a z} \cos (a x+d y)\right] e^{-d^2 t}, $$

$$v(x, y, z, t)= -a\left[e^{a y} \sin (a z+d x)+e^{a x} \cos (a y+d z)\right] e^{-d^2 t}, $$

$$w(x, y, z, t)= -a\left[e^{a z} \sin (a x+d y)+e^{a y} \cos (a z+d x)\right] e^{-d^2 t}, $$

$$p(x, y, z, t)= -\frac{1}{2} a^2\left[e^{2 a x}+e^{2 a y}+e^{2 a z}+2 \sin (a x+d y) \cos (a z+d x) e^{a(y+z)} +2 \sin (a y+d z) \cos (a x+d y) e^{a(z+x)} +2 \sin (a z+d x) \cos (a y+d z) e^{a(x+y)}\right] e^{-2 d^2 t}.$$
## 3. 问题求解
为节约篇幅，问题求解以NSFNet3为例。
### 3.1 模型构建
本文使用PINN经典的MLP模型进行训练。
``` py linenums="33"
--8<--
examples/hpinns/holography.py:33:38
--8<--
```
### 3.2 超参数设定
指定残差点、边界点、初值点的个数，以及可以指定边界损失函数和初值损失函数的权重
``` py linenums="41"
--8<--
examples/hpinns/holography.py:41:50
--8<--
```
### 3.3 数据生成
因数据集为解析解，我们先构造解析解函数
``` py linenums="10"
--8<--
examples/hpinns/holography.py:10:21
--8<--
```

然后先后取边界点、初值点、以及用于计算残差的内部点（具体取法见[论文](https://arxiv.org/abs/2003.06496)节3.3）以及生成测试点。
``` py linenums="53"
--8<--
examples/hpinns/holography.py:53:125
--8<--
```
### 3.4 约束构建
由于我们边界点和初值点具有解析解，因此我们使用监督约束
``` py linenums="173"
--8<--
examples/hpinns/holography.py:173:185
--8<--
```

其中alpha和beta为该损失函数的权重，在本代码中与论文中描述一致，都取为100

使用内部点构造纳韦斯托克方程的残差约束
``` py linenums="188"
--8<--
examples/hpinns/holography.py:188:203
--8<--
```
### 3.5 评估器构建
使用在数据生成时生成的测试点构造的测试集用于模型评估：
``` py linenums="208"
--8<--
examples/hpinns/holography.py:208:216
--8<--
```

### 3.6 优化器构建
与论文中描述相同，我们使用分段学习率构造Adam优化器，其中可以通过调节_epoch_list_来调节训练轮数。
``` py linenums="219"
--8<--
examples/hpinns/holography.py:219:226
--8<--
```

### 3.7 模型训练与评估
完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`。

``` py linenums="230"
--8<--
examples/epnn/epnn.py:230:247
--8<--
```

最后启动训练即可：

``` py linenums="249"
--8<--
examples/epnn/epnn.py:249:249
--8<--
```


## 4. 完整代码
``` py linenums="1" title="epnn.py"
--8<--
examples/NSFNet/VP_NSFNet1.py
examples/NSFNet/VP_NSFNet2.py
examples/NSFNet/VP_NSFNet3.py
--8<--
```
## 5. 结果展示
### NSFNet1:
| size 4*50 | paper  | code(without BFGS) | PaddleScience  |
|-------------------|--------|--------------------|---------|
| u                 | 0.084% | 0.062%             | 0.055%  |
| v                 | 0.425% | 0.431%             | 0.399%  |
### NSFNet2:
T=0
| size 10*100| paper  | code(without BFGS) | PaddleScience  |
|-------------------|--------|--------------------|---------|
| u                 | /| 0.403%         | 0.138%  |
| v                 | / | 1.5%             |  0.488% |

速度场

![image](ttps://github.com/DUCH714/hackthon5th53/blob/develop/examples/NSFNet/fig/Cylinder%20wake.gif)

涡流场（t=4.0）

![image](https://github.com/DUCH714/hackthon5th53/blob/develop/examples/NSFNet/fig/NSFNet2_vorticity.png)
### NSFNet3:
Test dataset:
| size 10*100 |  code(without BFGS) | PaddleScience  |
|-------------------|--------------------|---------|
| u                 |  0.0766%        | 0.059%  |
| v                 | 0.0689%            | 0.082%  |
| w                 |   0.109%           | 0.0732%  |

T=1
| size 10*100 | paper   | PaddleScience  |
|-------------------|--------|---------|
| u                 | 0.426%        | 0.115%  |
| v                 | 0.366%            | 0.199%  |
| w                 | 0.587%            | 0.217%  |

速度场 z=0
![image](https://github.com/DUCH714/hackthon5th53/blob/develop/examples/NSFNet/fig/Beltrami%20flow.gif)

## 6. 参考资料
[NSFnets (Navier-Stokes Flow nets): Physics-informed neural networks for the incompressible Navier-Stokes equations](https://arxiv.org/abs/2003.06496)

[Github NSFnets](https://github.com/Alexzihaohu/NSFnets/tree/master)
