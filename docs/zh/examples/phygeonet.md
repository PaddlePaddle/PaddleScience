# PhyGeoNet
<a href="https://aistudio.baidu.com/projectdetail/7195983" class="md-button md-button--primary" style>AI Studio快速体验</a>

=== "模型训练命令"

    ``` sh

    # heat_equation
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation.npz -P ./data/

    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation.npz --output ./data/heat_equation.npz

    python main.py data_dir=./data/heat_equation.npz

    # heat_equation_bc
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation_bc.npz -P ./data/
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation_bc_test.npz -P ./data/

    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation_bc.npz --output ./data/heat_equation.npz
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation_bc_test.npz --output ./data/heat_equation.npz

    python main.py data_dir=./data/heat_equation_bc.npz

    ```
=== "模型评估命令"

    ``` sh

    # heat_equation
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation.npz -P ./data/

    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation.npz --output ./data/heat_equation.npz

    python main.py mode=eval data_dir=./data/heat_equation.npz  EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/PhyGeoNet/heat_equation_pretrain.pdparams

    # heat_equation_bc
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation_bc.npz -P ./data/
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation_bc_test.npz -P ./data/

    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation_bc.npz --output ./data/heat_equation.npz
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation_bc_test.npz --output ./data/heat_equation.npz

    python main.py mode=eval test_data_dir=./data/heat_equation_bc_test.npz  EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/PhyGeoNet/heat_equation_bc_pretrain.pdparams

    ```

| 模型 | mRes | ev |
| :-- | :-- | :-- |
| [heat_equation](https://paddle-org.bj.bcebos.com/paddlescience/models/PhyGeoNet/heat_equation_pretrain.pdparams)  | 0.815 |0.095|
| [heat_equation_parameterized_bc](https://paddle-org.bj.bcebos.com/paddlescience/models/PhyGeoNet/heat_equation_bc_pretrain.pdparams)  | 992 |0.31|

## 1. 背景简介
 最近几年，深度学习在很多领域取得了非凡的成就，尤其是计算机视觉和自然语言处理方面，而受启发于深度学习的快速发展，基于深度学习强大的函数逼近能力，神经网络在科学计算领域也取得了成功，现阶段的研究主要分为两大类，一类是将物理信息以及物理限制加入损失函数来对神经网络进行训练, 其代表有 PINN 以及 Deep Ritz Net，另一类是通过数据驱动的深度神经网络算子，其代表有 FNO 以及 DeepONet。这些方法都在科学实践中获得了广泛应用，比如天气预测，量子化学，生物工程，以及计算流体等领域。
而在所有不同类型的深度学习网络中，卷积神经网络获得了越来越多的关注，这是因为卷积神经网络中的卷积具有参数共享的性质，可以学习大尺度的时空域。但是卷积神经网络的缺陷是只能处理规则均匀的数据。
## 2. 问题定义
而在实际科学计算问题中，很多偏微分方程的求解域是复杂边界且非均匀的。现有神经网络往往针对具有规则边界以及均匀网格的求解域，所以并没有实际应用效果。

本文针对物理信息神经网络在复杂边界非均匀网格求解域上效果较差的问题，提出了通过坐标变化将不规则边界非均匀网格变成规则边界均匀网格的方法，除此之外，本文利用变成均匀网格后，卷积神经网络的上述优势，提出相对应的的物理信息卷积神经网络。
## 3. 问题求解
为节约篇幅，问题求解以heat equation为例。
### 3.1 模型构建
本文使用提出的USCNN模型进行训练。
``` py linenums="236"
--8<--
examples/phygeonet/heat_equation/main.py:236:236
--8<--
```
### 3.2 数据读取
``` py linenums="227"
--8<--
examples/phygeonet/heat_equation/main.py:227:233
--8<--
```
### 3.3 输出转化函数构建
本文为强制边界约束，使用相对应的输出转化函数
``` py linenums="263"
--8<--
examples/phygeonet/heat_equation/main.py:263:283
--8<--
```
### 3.4 约束构建
构建相对应约束条件，由于边界约束为强制约束，约束条件主要为残差约束
``` py linenums="240"
--8<--
examples/phygeonet/heat_equation/main.py:240:261
--8<--
```

### 3.5 评估器构建
由于评估器使用不同的输出转化函数，因此本文不构建评估器，转而调用 evaluate() 函数

``` py linenums="301"
--8<--
examples/phygeonet/heat_equation/main.py:301:360
--8<--
```
### 3.6 优化器构建
与论文中描述相同，我们使用恒定学习率构造Adam优化器。
``` py linenums="238"
--8<--
examples/phygeonet/heat_equation/main.py:238:238
--8<--
```

### 3.7 模型训练
完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`。

``` py linenums="287"
--8<--
examples/phygeonet/heat_equation/main.py:287:294
--8<--
```

最后启动训练即可：

``` py linenums="296"
--8<--
examples/phygeonet/heat_equation/main.py:296:296
--8<--
```


## 4. 完整代码
``` py linenums="1" title="heat_equation/main.py"
--8<--
examples/phygeonet/heat_equation/main.py
--8<--
```
## 5. 结果展示
Heat equation结果展示:
![image](https://paddle-org.bj.bcebos.com/paddlescience/docs/PhyGeoNet/heat_equation.jpg)

Heat equation with boundary 结果展示：

T=0
![image](https://paddle-org.bj.bcebos.com/paddlescience/docs/PhyGeoNet/heat_equation_bc_1.png)

T=3
![image](https://paddle-org.bj.bcebos.com/paddlescience/docs/PhyGeoNet/heat_equation_bc_2.png)

T=6
![iamge](https://paddle-org.bj.bcebos.com/paddlescience/docs/PhyGeoNet/heat_equation_bc_3.png)

## 6. 总结
本文通过使用调和映射构造坐标变换函数，使得物理信息网络可以在不规则非均匀网格上面进行训练，同时，因为该映射为使用传统方法进行，所以无需训练即可在网络前后嵌入。通过大量实验表明，该网络可以在各种不规则网格问题上表现比SOAT网络突出。
## 7. 参考资料
[PhyGeoNet: Physics-informed geometry-adaptive convolutional neural networks for solving parameterized steady-state PDEs on irregular domain](https://www.sciencedirect.com/science/article/pii/S0021999120308536?via%3Dihub)

[Github PhyGeoNet](https://github.com/Jianxun-Wang/phygeonet/tree/master?tab=readme-ov-file)
