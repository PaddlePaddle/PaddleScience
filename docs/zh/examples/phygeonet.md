# PhyGeoNet

=== "模型训练命令"

    ``` sh

    python main.py

## 1. 背景简介
 最近几年，深度学习在很多领域取得了非凡的成就，尤其是计算机视觉和自然语言处理方面，而受启发于深度学习的快速发展，基于深度学习强大的函数逼近能力，神经网络在科学计算领域也取得了成功，现阶段的研究主要分为两大类，一类是将物理信息以及物理限制加入损失函数来对神经网络进行训练, 其代表有 PINN 以及 Deep Retz Net，另一类是通过数据驱动的深度神经网络算子，其代表有 FNO 以及 DeepONet。这些方法都在科学实践中获得了广泛应用，比如天气预测，量子化学，生物工程，以及计算流体等领域。
## 2. 问题定义
在实际科学计算问题中，很多偏微分方程的求解域是复杂且非均匀的，而现有神经网络往往针对规则的求解域。

因此，本文针对对物理信息神经网络边界信息处理困难的问题，针对性地提出了通过坐标变化将不规则边界非均匀网格变成规则边界均匀网格的方法，除此之外，本文基于变成均匀网格后在卷积神经网络上的优势，提出针对性的物理信息卷积神经网络。
## 3. 问题求解
为节约篇幅，问题求解以heat equation为例。
### 3.1 模型构建
本文使用提出的USCNN模型进行训练。
``` py linenums="179"
--8<--
examples/hpinns/holography.py:179:188
--8<--
```
### 3.2 数据读取
我们从数据集中读取数据，公开数据集在[AIstudio](https://aistudio.baidu.com/datasetdetail/253292)
``` py linenums="166"
--8<--
examples/hpinns/holography.py:166:172
--8<--
```
### 3.3 输出转化函数构建
本文为强制边界约束，使用相对应的输出转化函数
``` py linenums="215"
--8<--
examples/hpinns/holography.py:215:235
--8<--
```
### 3.4 约束构建
构建相对应约束条件，由于边界约束为强制约束，约束条件主要为残差约束
``` py linenums="192"
--8<--
examples/hpinns/holography.py:192:211
--8<--
```

### 3.5 评估器构建
由于评估器使用不同的输出转化函数，因此本文不构建评估器，转而调用 evaluate() 函数

``` py linenums="254"
--8<--
examples/hpinns/holography.py:254:282
--8<--
```
### 3.6 优化器构建
与论文中描述相同，我们使用恒定学习率构造Adam优化器。
``` py linenums="190"
--8<--
examples/hpinns/holography.py:190:190
--8<--
```

### 3.7 模型训练
完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`。

``` py linenums="239"
--8<--
examples/epnn/epnn.py:239:247
--8<--
```

最后启动训练即可：

``` py linenums="249"
--8<--
examples/epnn/epnn.py:249:249
--8<--
```


## 4. 完整代码
``` py linenums="1" title="phygeonet.py"
--8<--
examples/phygeonet/heat_equation/main.py
examples/phygeonet/heat_equation_parameterized_bc/main.py
--8<--
```
## 5. 结果展示
### heat_equation:
|  measurement  |source code | PaddleScience  |
|-----------------|--------------------|---------|
| mRes                | 2.46            | 0.815  |
| ev                  | 0.125            | 0.095  |
### heat_equation_parameterized_bc:
| measurement  | source code | PaddleScience  |
|----------------|--------------------|---------|
| mRes                 | 2013         | 992  |
| ev                  | 0.050            |  0.027 |



## 6. 总结
本文通过使用调和映射构造坐标变换函数，使得物理信息网络可以在不规则非均匀网格上面进行训练，同时，因为该映射为使用传统方法进行，所以无需训练即可在网络前后嵌入。通过大量实验表明，该网络可以在各种不规则网格问题上表现比SOAT网络突出。
## 7. 参考资料
[NSFnets (Navier-Stokes Flow nets): Physics-informed neural networks for the incompressible Navier-Stokes equations](https://arxiv.org/abs/2003.06496)

[Github NSFnets](https://github.com/Alexzihaohu/NSFnets/tree/master)
