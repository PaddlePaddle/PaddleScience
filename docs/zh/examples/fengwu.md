# FengWu

=== "模型训练命令"

    暂无

=== "模型评估命令"

    暂无

=== "模型导出命令"

    暂无

=== "模型推理命令"

    ``` sh
    # Download sample input data
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/models/Fengwu/input1.npy -P ./data
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/models/Fengwu/input2.npy -P ./data

    # Download pretrain model weight
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/models/Fengwu/fengwu_v2.onnx -P ./inference

    # inference
    python predict.py
    ```

## 1. 背景简介

随着近年来全球气候变化加剧，极端天气频发，各界对天气预报的时效和精度的期待更是与日俱增。如何提高天气预报的时效和准确度，一直是业内的重点课题。AI大模型“风乌”基于多模态和多任务深度学习方法构建，实现在高分辨率上对核心大气变量进行超过10天的有效预报，并在80%的评估指标上超越DeepMind发布的模型GraphCast。同时，“风乌”仅需30秒即可生成未来10天全球高精度预报结果，在效率上大幅优于传统模型。

## 2. 模型原理

本章节仅对风乌气象大模型的原理进行简单地介绍，详细的理论推导请阅读 [FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead](https://arxiv.org/pdf/2304.02948)。

模型的总体结构如图所示：

<figure markdown>
  ![result](https://paddle-org.bj.bcebos.com/paddlescience/docs/fengwu/model_architecture.png){ loading=lazy style="margin:0 auto;"}
  <figcaption>模型结构</figcaption>
</figure>

模型将气候变量作为不同模态的输入。在 `Modal-Customized Encoder` 中将多个模态的特征进行编码，并使用基于 Transformer 的 `Cross-modal Fuser` 对编码后的特征进行融合，得到联合表示，最后在 `Modal-Customized Decoder` 中从联合表示中分别预测气候变量。

模型使用预训练权重推理，接下来将介绍模型的推理过程。

## 3. 模型构建

在该案例中，实现了 FengWuPredictor用于ONNX模型的推理：

``` py linenums="74" title="examples/fengwu/predict.py"
--8<--
examples/fengwu/predict.py:74:130
--8<--
```

``` yaml linenums="28" title="examples/fengwu/conf/fengwu.yaml"
--8<--
examples/fengwu/conf/fengwu.yaml:28:46
--8<--
```

其中，`input_file` 和 `input_next_file` 分别代表网络模型输入的开始时刻气象数据和6小时后的气象数据。

## 4. 结果可视化

模型推理结果包含 56 个 npy 文件，表示从预测时间点开始，未来 14 天内每隔6小时的气象数据。结果可视化需要先将数据从 npy 转换为 NetCDF 格式，然后采用 ncvue 进行查看。

1. 安装相关依赖
```python
pip install cdsapi netCDF4 ncvue
```

2. 使用脚本进行数据转换
```python
python convert_data.py
```

3. 使用 ncvue 打开转换后的 NetCDF 文件, ncvue 具体说明见[ncvue官方文档](https://github.com/mcuntz/ncvue)

## 5. 完整代码

``` py linenums="1" title="examples/fengwu/predict.py"
--8<--
examples/fengwu/predict.py
--8<--
```

## 6. 结果展示

下图展示了模型的未来6小时平均海平面气压预测结果，更多指标可以使用 ncvue 查看。

<figure markdown>
  ![result](https://paddle-org.bj.bcebos.com/paddlescience/docs/fengwu/image.png){ loading=lazy style="margin:0 auto;"}
  <figcaption>未来6小时平均海平面气压</figcaption>
</figure>

## 7. 参考资料

- [FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead](https://arxiv.org/pdf/2304.02948)
