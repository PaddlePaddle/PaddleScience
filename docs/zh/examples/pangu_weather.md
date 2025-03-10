# Pangu-Weather

=== "模型训练命令"

    暂无

=== "模型评估命令"

    暂无

=== "模型导出命令"

    暂无

=== "模型推理命令"

    ``` sh
    # Download sample input data
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/models/Pangu/input_surface.npy -P ./data
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/models/Pangu/input_upper.npy -P ./data

    # Download pretrain model weight
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/models/Pangu/pangu_weather_1.onnx -P ./inference
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/models/Pangu/pangu_weather_3.onnx -P ./inference
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/models/Pangu/pangu_weather_6.onnx -P ./inference
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/models/Pangu/pangu_weather_24.onnx -P ./inference

    # 1h interval-time model inference
    python predict.py INFER.export_path=inference/pangu_weather_1
    # 3h interval-time model inference
    python predict.py INFER.export_path=inference/pangu_weather_3
    # 6h interval-time model inference
    python predict.py INFER.export_path=inference/pangu_weather_6
    # 24h interval-time model inference
    python predict.py INFER.export_path=inference/pangu_weather_24
    ```

## 1. 背景简介

盘古气象大模型(Pangu-Weather)是首个精度超过传统数值预报方法的 AI 方法，其提供了 1 小时间隔、3 小时间隔、6 小时间隔、24 小时间隔的预训练模型。其使用的数据，包括垂直高度上13个不同气压层，每层五种气象要素（温度、湿度、位势、经度和纬度方向的风速），以及地球表面的四种气象要素（2米温度、经度和纬度方向的10米风速、海平面气压）。1 小时 - 7 天预测精度均高于传统数值方法（即欧洲气象中心的 operational IFS）。

同时，盘古气象大模型在一张V100显卡上只需要1.4秒就能完成24小时的全球气象预报，相比传统数值预报提速10000倍以上。

## 2. 模型原理

本章节仅对盘古气象大模型的原理进行简单地介绍，详细的理论推导请阅读 [Pangu-Weather: A 3D High-Resolution System for Fast and Accurate Global Weather Forecast](https://arxiv.org/pdf/2211.02556)。

模型的总体结构如图所示：

<figure markdown>
  ![result](https://paddle-org.bj.bcebos.com/paddlescience/docs/pangu-weather/model_architecture.png){ loading=lazy style="margin:0 auto;"}
  <figcaption>模型结构</figcaption>
</figure>

其主要思想是使用一个视觉transformer的3D变种来处理复杂的不均匀的气象要素。由于气象数据分辨率很大，因而相比于常见的vision transformer方法，研究人员将网络的encoder和decoder减少到2级（8个block），同时采用Swin transformer的滑窗注意力机制，以减少网络的计算量

模型使用预训练权重推理，接下来将介绍模型的推理过程。

## 3. 模型构建

在该案例中，实现了 PanguWeatherPredictor用于ONNX模型的推理：

``` py linenums="67" title="examples/pangu_weather/predict.py"
--8<--
examples/pangu_weather/predict.py:67:97
--8<--
```

``` yaml linenums="29" title="examples/pangu_weather/conf/pangu_weather.yaml"
--8<--
examples/pangu_weather/conf/pangu_weather.yaml:29:44
--8<--
```

其中，`input_file` 和 `input_surface_file` 分别代表网络模型输入的高空气象数据和地面气象。

## 4. 结果可视化

先将数据从 npy 转换为 NetCDF 格式，然后采用 ncvue 进行可视化

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

``` py linenums="1" title="examples/pangu_weather/predict.py"
--8<--
examples/pangu_weather/predict.py
--8<--
```

## 6. 结果展示

下图展示了模型的温度预测结果，更多指标可以使用 ncvue 查看。

<figure markdown>
  ![result](https://paddle-org.bj.bcebos.com/paddlescience/docs/pangu-weather/temperature.png){ loading=lazy style="margin:0 auto;"}
  <figcaption>温度预测结果</figcaption>
</figure>

## 7. 参考资料

- [Pangu-Weather: A 3D High-Resolution System for Fast and Accurate Global Weather Forecast](https://arxiv.org/pdf/2211.02556)
