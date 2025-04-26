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


    # inference
    pip install -r requirements.txt
    python predict.py
    ```

## 1. 背景简介

FuXi 是阿里巴巴达摩院开发的一款级联机器学习天气预报系统，其目标是提供长达15天的全球天气预报。尽管现有先进的机器学习模型在10天预测中已展现出超越传统数值预报系统的性能，但长期预测中误差累积仍然是一个挑战。FuXi 的研发旨在克服这一难题，力求在15天的预测中达到与顶尖数值预报系统（如 ECMWF）整体平均水平相当的精度，其开发基于长达39年的 ECMWF ERA5 再分析数据集。

## 2. 模型原理

FuXi 采用级联模型结构，针对三个连续的预测时间段（0-5天、5-10天和10-15天）分别进行了优化。这种设计旨在减缓长期预测中的误差累积。此外，FuXi 还发展出 FuXi-Extreme 模型，该模型在标准的 FuXi 基础上融入了去噪扩散概率模型 (DDPM)， 用于增强前5天地表预报数据的细节和质量。

模型使用预训练权重推理，接下来将介绍模型的推理过程。

## 3. 模型构建

在该案例中，实现了 FuXiPredictor用于ONNX模型的推理：

``` py linenums="74" title="examples/fuxi/predict.py"
--8<--
examples/fuxi/predict.py:46:131
--8<--
```

FuXi采用级联模型结构，通过`fuxi_short.yaml`、`fuxi_medium.yaml`、`fuxi_long.yaml`来预测三个连续的预测时间段（0-5天、5-10天和10-15天）。

## 4. 结果可视化

使用 ncvue 打开保存的 NetCDF 文件, ncvue 具体说明见[ncvue官方文档](https://github.com/mcuntz/ncvue)

## 5. 完整代码

``` py linenums="1" title="examples/fuxi/predict.py"
--8<--
examples/fuxi/predict.py
--8<--
```

## 6. 结果展示

example中展示了15天全球天气预报，具体指标可以使用 ncvue 查看。

## 7. 参考资料

- [FuXi: A cascade machine learning forecasting system for 15-day global weather forecast](https://arxiv.org/abs/2306.12873)
