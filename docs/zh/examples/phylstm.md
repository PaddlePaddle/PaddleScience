# PhyLSTM

=== "模型训练命令"

    === "phylstm2"

        ``` sh
        # linux
        wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat --output data_boucwen.mat
        python phylstm2.py
        ```

    === "phylstm3"

        ``` sh
        # linux
        wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat --output data_boucwen.mat
        python phylstm3.py
        ```

=== "模型评估命令"

    === "phylstm2"

        ``` sh
        # linux
        wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat --output data_boucwen.mat
        python phylstm2.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/phylstm/phylstm2_pretrained.pdparams
        ```

    === "phylstm3"

        ``` sh
        # linux
        wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat --output data_boucwen.mat
        python phylstm3.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/phylstm/phylstm3_pretrained.pdparams
        ```

| 预训练模型  | 指标 |
|:--| :--|
| [phylstm2_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/phylstm/phylstm2_pretrained.pdparams) | loss(sup_valid): 0.00799 |
| [phylstm3_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/phylstm/phylstm3_pretrained.pdparams) | loss(sup_valid): 0.03098 |

## 1. 背景简介

我们引入了一种创新的物理知识LSTM框架，用于对缺乏数据的非线性结构系统进行元建模。基本概念是将可用但尚不完整的物理知识（如物理定律、科学原理）整合到深度长短时记忆（LSTM）网络中，该网络在可行的解决方案空间内限制和促进学习。物理约束嵌入在损失函数中，以强制执行模型训练，即使在可用训练数据集非常有限的情况下，也能准确地捕捉潜在的系统非线性。特别是对于动态结构，考虑运动方程的物理定律、状态依赖性和滞后本构关系来构建物理损失。嵌入式物理可以缓解过拟合问题，减少对大型训练数据集的需求，并提高训练模型的鲁棒性，使其具有外推能力，从而进行更可靠的预测。因此，物理知识指导的深度学习范式优于传统的非物理指导的数据驱动神经网络。

## 2. 问题定义

结构系统的元建模旨在开发低保真度（或低阶）模型，以有效地捕捉潜在的非线性输入-输出行为。元模型可以在从高保真度模拟或实际系统感知获得的数据集上进行训练。为了更好地说明，我们考虑一个建筑类型结构并假设地震动力学由低保真度非线性运动方程（EOM）支配：

$$
\mathbf{M} \ddot{\mathbf{u}}+\underbrace{\mathbf{C} \dot{\mathbf{u}}+\lambda \mathbf{K u}+(1-\lambda) \mathbf{K r}}_{\mathbf{h}}=-\mathbf{M} \Gamma a_g
$$

其中M是质量矩阵；C为阻尼矩阵；K为刚度矩阵。

控制方程可以改写成一个更一般的形式：

$$
\ddot{\mathbf{u}}+\mathrm{g}=-\Gamma a_g
$$

## 3. 问题求解

接下来开始讲解如何将问题一步一步地转化为 PaddleScience 代码，用深度学习的方法求解该问题。
为了快速理解 PaddleScience，接下来仅对模型构建、方程构建、计算域构建等关键步骤进行阐述，而其余细节请参考 [API文档](../api/arch.md)。

### 3.1 模型构建

在 PhyLSTM 问题中，建立 LSTM 网络 Deep LSTM network，用 PaddleScience 代码表示如下

``` py linenums="102"
--8<--
examples/phylstm/phylstm2.py:102:107
--8<--
```

DeepPhyLSTM 参数 input_size 是输入大小，output_size 是输出大小，hidden_size 是隐藏层大小，model_type是模型类型。

### 3.2 数据构建

运行本问题代码前请按照下方命令下载 [data_boucwen.mat](https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat)

``` sh
wget -nc -P ./ https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat
```

本案例涉及读取数据构建，如下所示

``` py linenums="37"
--8<--
examples/phylstm/phylstm2.py:37:100
--8<--
```

### 3.3 约束构建

设置训练数据集和损失计算函数，返回字段，代码如下所示：

``` py linenums="119"
--8<--
examples/phylstm/phylstm2.py:119:145
--8<--
```

### 3.4 评估器构建

设置评估数据集和损失计算函数，返回字段，代码如下所示：

``` py linenums="147"
--8<--
examples/phylstm/phylstm2.py:147:174
--8<--
```

### 3.5 超参数设定

接下来我们需要指定训练轮数，此处我们按实验经验，使用 100 轮训练轮数。

``` yaml linenums="39"
--8<--
examples/phylstm/conf/phylstm2.yaml:39:39
--8<--
```

### 3.6 优化器构建

训练过程会调用优化器来更新模型参数，此处选择 `Adam` 优化器并设定 `learning_rate` 为 1e-3。

``` py linenums="177"
--8<--
examples/phylstm/phylstm2.py:177:177
--8<--
```

### 3.7 模型训练与评估

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`。

``` py linenums="178"
--8<--
examples/phylstm/phylstm2.py:178:192
--8<--
```

最后启动训练、评估即可：

``` py linenums="194"
--8<--
examples/phylstm/phylstm2.py:194:197
--8<--
```

## 4. 完整代码

=== "phylstm2"

    ``` py linenums="1" title="phylstm2.py"
    --8<--
    examples/phylstm/phylstm2.py
    --8<--
    ```

=== "phylstm3"

    ``` py linenums="1" title="phylstm3.py"
    --8<--
    examples/phylstm/phylstm3.py
    --8<--
    ```

## 5. 结果展示

PhyLSTM2 案例针对 epoch=100 和 learning\_rate=1e-3 的参数配置进行了实验，结果返回Loss为 0.00799。

PhyLSTM3 案例针对 epoch=200 和 learning\_rate=1e-3 的参数配置进行了实验，结果返回Loss为 0.03098。

## 6. 参考资料

- [Physics-informed multi-LSTM networks for metamodeling of nonlinear structures](https://www.sciencedirect.com/science/article/abs/pii/S0045782520304114)
- [https://github.com/zhry10/PhyLSTM.git](https://github.com/zhry10/PhyLSTM.git)
