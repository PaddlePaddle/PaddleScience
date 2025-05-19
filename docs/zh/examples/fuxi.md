# FuXi

=== "模型训练命令"

    暂无

=== "模型评估命令"

    暂无

=== "模型导出命令"

    暂无

=== "模型推理命令"

    ``` sh
    cd examples/fuxi
    # Download sample input data and model weight from https://pan.baidu.com/s/1PDeb-nwUprYtu9AKGnWnNw?pwd=fuxi#list/path=%2F
    unzip Sample_Data.zip
    unzip FuXi_EC.zip

    # modify the path of model and datasets in examples/fuxi/conf, and inference
    pip install -r requirements.txt
    python predict.py
    ```

## 1. 背景简介

FuXi模型是一个机器学习（ML）天气预报系统，旨在生成15天的全球天气预报。它利用了39年的欧洲中期天气预报中心（ECMWF）ERA5再分析数据集，这些数据具有0.25°的空间分辨率和6小时的时间分辨率。FuXi系统的命名来源于中国古代神话中的人物伏羲，他被认为是中国的第一个天气预报员。

FuXi模型开发的关键方面和背景包括：

- 动机：FuXi的开发是出于对当前ML模型在长期天气预报中由于误差累积而产生的局限性的考虑。虽然ML模型在短期预报中已经显示出前景，但在长期预报（例如15天）中实现与欧洲中期天气预报中心（ECMWF）的传统数值天气预报（NWP）模型相当的性能仍然是一个挑战。

- Cascade模型架构：为了解决误差累积的问题，FuXi采用了一种新颖的Cascade ML模型架构。这种架构使用针对特定5天预报时间窗口（0-5天、5-10天和10-15天）优化的预训练模型，以提高不同预报时效的准确性。

- 基础模型: FuXi的基础模型是一个自动回归模型，旨在从高维天气数据中提取复杂特征并学习关系。

- 训练过程：FuXi的训练过程包括预训练和微调两个步骤。预训练步骤优化模型以预测单个时间步，而微调则涉及训练Cascade模型以用于它们各自的预报时间窗口。

- 性能：FuXi系统在15天预报中表现出与ECMWF集合平均（EM）相当的性能，并且在有效预报时效方面优于ECMWF高分辨率预报（HRES）。

模型的总体结构如图所示：

<figure markdown>
  ![result](https://paddle-org.bj.bcebos.com/paddlescience/docs/fuxi/fuxi.png){ loading=lazy style="margin:0 auto;"}
  <figcaption>模型结构</figcaption>
</figure>

FuXi模型使用了第五代ECMWF再分析数据集ERA5。该数据集提供了从1940年1月至今的地表和高空参数的逐小时数据。ERA5数据集是通过同化使用ECMWF的集成预报系统（IFS）模型获得的高质量和丰富的全球观测资料而生成的。ERA5数据被广泛认为是全面而准确的再分析档案，这使其适合作为训练FuXi模型的地面实况。对于FuXi模型，使用了ERA5数据集的一个子集，该子集跨越39年，具有0.25°的空间分辨率和6小时的时间分辨率。该模型旨在预测13个压力层的5个高空大气变量和5个地表变量。
数据集被分为训练集、验证集和测试集。训练集包含1979年至2015年的54020个样本，验证集包含2016年和2017年的2920个样本，样本外测试集包含2018年的1460个样本。此外，还创建了两个参考数据集HRES-fc0和ENS-fc0，以评估ECMWF高分辨率预报（HRES）和集合平均（EM）的性能。

## 2. 模型原理

FuXi模型是一种自回归模型，它利用前两个时间步的天气参数($X^{t-1}$, $X^t$)作为输入，来预测下一个时间步的天气参数($X^{t+1}$)。其中，t、t-1和t+1分别代表当前、前一个和下一个时间步。本模型中使用的时间步长为6小时。通过将模型的输出用作后续预测的输入，该系统可以生成不同预报时效的预报。

使用单个FuXi模型生成15天预报需要进行60次迭代。与基于物理的NWP模型不同，纯数据驱动的ML模型缺乏物理约束，这可能导致长期预报的误差显著增长和不切实际的预测结果。使用自回归多步损失可以有效减少长期预报的累积误差。这种损失函数类似于四维变分数据同化（4D-Var）方法中使用的成本函数，其目的是识别在同化时间窗内与观测结果最佳拟合的初始天气条件。虽然增加自回归步数可以提高长期预报的准确性，但也会降低短期预报的准确性。此外，与增加4D-Var的同化时间窗类似，增加自回归步数需要更多的内存和计算资源来处理训练过程中的梯度。  

在进行迭代预报时，随着预报时效的增加，误差累积是不可避免的。此外，先前的研究表明，单个模型无法在所有预报时效都达到最佳性能。为了优化短期和长期预报的性能，论文提出了一种使用预训练FuXi模型的Cascade模型架构，这些模型经过微调，以在特定的5天预报时间窗内实现最佳性能。这些时间窗被称为FuXi-Short（0-5天）、FuXi-Medium（5-10天）和FuXi-Long（10-15天）。FuXi-Short和FuXi-Medium的输出分别在第20步和第40步被用作FuXi-Medium和FuXi-Long的输入。与Pangu-Weather中使用的贪婪分层时间聚合策略（该策略利用4个分别预测1小时、3小时、6小时和24小时预报时效的模型来减少步数）不同，Cascade FuXi模型不存在时间不一致的问题。

基础FuXi模型的模型架构由三个主要部分组成，如论文所诉：Cube Embedding、U-Transformer和全连接（FC）层。输入数据结合了高空和地面变量，并创建了一个维度为2×70×721×1440的数据立方体，其中2代表前两个时间步（t-1和t），70代表输入变量的总数，721和1440分别代表纬度（H）和经度（W）网格点。  

首先，高维输入数据通过联合时空Cube Embedding被降维到C×180×360，其中C是通道数，设置为1536。Cube Embedding的主要目的是减少输入数据的时间和空间维度，降低数据冗余度。随后，U-Transformer处理嵌入后的数据，并使用一个简单的FC层进行预测。输出结果首先被reshape为70×720×1440，然后通过双线性插值恢复到原始输入形状70×721×1440。

U-Transformer由48个重复的Swin Transformer V2块构建，并按如下方式计算缩放余弦注意力:

$$Attention(Q, K, V) = (cos(Q, K)/\tau +B)V$$

其中B表示相对位置偏差，是一个可学习的标量，在不同的head和层之间不共享。余弦函数是自然归一化的，这导致较小的注意力值。

模型使用预训练权重推理，接下来将介绍模型的推理过程。

## 3. 模型构建

在该案例中，实现了 FuXiPredictor用于ONNX模型的推理：

``` py linenums="44" title="examples/fuxi/predict.py"
--8<--
examples/fuxi/predict.py:44:124
--8<--
```

FuXi采用级联模型结构，通过`fuxi_short.yaml`、`fuxi_medium.yaml`、`fuxi_long.yaml`来预测三个连续的预测时间段（0-5天、5-10天和10-15天）。

## 4. 结果可视化

使用 `examples/fuxi/visualize.py` 进行画图，进行结果可视化。

## 5. 完整代码

``` py linenums="1" title="examples/fuxi/predict.py"
--8<--
examples/fuxi/predict.py
--8<--
```

## 6. 结果展示

模型推理结果包含 60 个 NetCDF 文件，表示从预测时间点开始，未来 15 天内每个模型20个时间步的气象数据。

使用 `examples/fuxi/visualize.py` 进行画图，进行结果可视化。

```python
python3 visualize.py --data_dir outputs_fuxi/ --save_dir outputs_fuxi/ --step 6
```

下图展示了

<figure markdown>
  ![result](https://paddle-org.bj.bcebos.com/paddlescience/docs/fuxi/image.png){ loading=lazy style="margin:0 auto;"}
  <figcaption>未来6小时天气预测结果</figcaption>
</figure>

## 7. 参考资料

- [FuXi: A cascade machine learning forecasting system for 15-day global weather forecast](https://arxiv.org/abs/2306.12873)
