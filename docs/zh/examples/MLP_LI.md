# Battery_LI（锂离子电池电极材料性能预测）

## 背景简介

锂离子电池（Lithium-ion Battery, LIB）作为现代储能技术的核心，广泛应用于消费电子、电动汽车、以及可再生能源的存储等领域。电极材料是锂离子电池性能的关键，其性能直接决定了电池的能量密度、功率密度、寿命、和安全性。然而，电极材料的研发是一个复杂且耗时的过程，通常需要实验测试和理论计算相结合，这对时间和资源的消耗非常大。

## 模型原理

该多层感知器（MLP）模型旨在利用从材料项目（Materials Project）数据集中提取的特征，预测锂离子电池电极材料的电化学性能。输入特征包括化学计量属性、晶体结构特性、电子结构属性和其他电池属性。输出为平均电压、比能量和比容量。

## 数据集介绍

| 数据集名称 | 下载链接 |
|-----------|---------|
| 训练集 + 验证集 | [MP_data_down_loading(train+validate).csv](https://paddle-org.bj.bcebos.com/paddlescience%2Fdocs%2FMP_data_down_loading(train%2Bvalidate).csv) |
| 训练集 + 验证集 + 测试集 | [MP_data_down_loading(train+validate+test).csv](https://paddle-org.bj.bcebos.com/paddlescience%2Fdocs%2FMP_data_down_loading(train%2Bvalidate%2Btest).csv) |

数据读取需要额外安装依赖 `bayesian-optimization`，请运行安装命令 `pip install bayesian-optimization`。

## 模型

要查看该模型的具体实现，请参考以下代码文件：`MLP_LI.py`
（未添加评估部分的实现）



## 训练好的模型权重文件

| 预训练模型                        |
|-----------------------------------|
| [MLP_LI_pretrained.pdparams]( https://paddle-org.bj.bcebos.com/paddlescience%2Fmodels%2FMLP_LI_pretrained.pdparams) |


## 模型训练命令
=== "模型训练命令"

    ``` sh  
    # 训练模型
    python MLP_LI.py --train

    # 下载预训练模型（如果需要）
    wget "https://paddle-org.bj.bcebos.com/paddlescience/models/MLP_LI/MLP_LI_pretrained.pdparams"

## 完整代码

``` py linenums="1" title="examples/MLP_LI/MLP_LI.py"
--8<--
examples/MLP_LI/MLP_LI.py
--8<--
```

## 模型性能

模型在测试集上的表现如下：

- **Test Loss**: 0.0058

- **VRMSE 电压**: 0.73
- **CRMSE 比容量**:165.01
- **ERMSE 比能量**: 238.64
- **Average RMSE 平均值**:134.79

此外，模型在各个输出指标上的平均绝对误差（MAE）如下：

- **VMAE 电压**: 0.55
- **CMAE 比容量**: 73.34
- **EMAE 比能量**: 180.10
- **Average MAE 平均值**: 84.66

这些结果表明模型在预测电压方面具有较高的精度，而在预测比容量和比能量方面还有一定的改进空间。

### 图表

#### 1. 电压的性能预测（原始尺度）
此图展示了电压的性能预测。预测值与真实值的比较用于评估模型的准确性。

![电压的性能预测（原始尺度）](https://paddle-org.bj.bcebos.com/paddlescience%2Fdocs%2Fperformance_prediction_voltage.png)

#### 2. 性能预测（原始尺度）
此图展示了模型对所有三个电化学性能（电压、比能量和比容量）的整体预测表现。

![性能预测（原始尺度）](https://paddle-org.bj.bcebos.com/paddlescience%2Fdocs%2Fperformance_prediction_original.png )

#### 3. 初始训练损失
以下图显示了在初始训练阶段的训练和验证损失变化情况（按Epochs）。

![初始训练损失](https://paddle-org.bj.bcebos.com/paddlescience%2Fdocs%2Finitial_training_loss.png)

## 结论
该 MLP 模型在提供的数据集上表现出较强的预测能力，尤其是在电压的预测上。然而，在比容量和比能量的预测上还有进一步改进的空间。未来可以通过更丰富的特征工程、更复杂的模型架构以及优化的超参数调整来提高模型的预测性能。

## 下一步
1. 考虑增加额外的特征或进行特征工程，以提高模型预测的准确性。
2. 尝试不同的神经网络架构或优化策略，以改进性能。
3. 继续进行超参数优化，以获得更好的模型性能。


## 参考资料

Yang, X., Li, Y., Liu, Z., & Zhang, W. (2022)
(https://doi.org/10.1016/j.gee.2022.10.002)
