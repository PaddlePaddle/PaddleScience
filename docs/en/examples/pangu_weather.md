# Pangu-Weather

=== "Model Training Command"

    None

=== "Model Evaluation Command"

    None

=== "Model Export Command"

    None

=== "Model Inference Command"

    ``` sh
    # Download sample input data
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Pangu/input_surface.npy -P ./data
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Pangu/input_upper.npy -P ./data

    # Download pretrain model weight
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Pangu/pangu_weather_1.onnx -P ./inference
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Pangu/pangu_weather_3.onnx -P ./inference
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Pangu/pangu_weather_6.onnx -P ./inference
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Pangu/pangu_weather_24.onnx -P ./inference

    # 1h interval-time model inference
    python predict.py INFER.export_path=inference/pangu_weather_1
    # 3h interval-time model inference
    python predict.py INFER.export_path=inference/pangu_weather_3
    # 6h interval-time model inference
    python predict.py INFER.export_path=inference/pangu_weather_6
    # 24h interval-time model inference
    python predict.py INFER.export_path=inference/pangu_weather_24
    ```

## 1. Background Introduction

Pangu-Weather is the first AI method whose accuracy exceeds that of traditional numerical forecasting methods. It provides pre-trained models with 1-hour interval, 3-hour interval, 6-hour interval, and 24-hour interval. The data used includes five meteorological elements (temperature, humidity, geopotential, longitude and latitude wind speeds) on 13 different pressure layers in vertical height, and four meteorological elements on the earth's surface (2-meter temperature, 10-meter wind speed in longitude and latitude directions, sea level pressure). The prediction accuracy from 1 hour to 7 days is higher than that of traditional numerical methods (i.e., operational IFS of the European Meteorological Centre).

At the same time, the Pangu-Weather model can complete a 24-hour global weather forecast in just 1.4 seconds on a V100 graphics card, which is more than 10,000 times faster than traditional numerical forecasting.

## 2. Model Principle

This chapter only briefly introduces the principle of the Pangu-Weather model. For detailed theoretical derivation, please read [Pangu-Weather: A 3D High-Resolution System for Fast and Accurate Global Weather Forecast](https://arxiv.org/pdf/2211.02556).

The overall structure of the model is shown in the figure:

<figure markdown>
  ![result](https://paddle-org.bj.bcebos.com/paddlescience/docs/pangu-weather/model_architecture.png){ loading=lazy style="margin:0 auto;"}
  <figcaption>Model Structure</figcaption>
</figure>

Its main idea is to use a 3D variant of a visual transformer to process complex and uneven meteorological elements. Due to the high resolution of meteorological data, compared with common vision transformer methods, researchers reduced the encoder and decoder of the network to 2 levels (8 blocks), and adopted the sliding window attention mechanism of Swin transformer to reduce the computation of the network.

The model uses pre-trained weights for inference. Next, the inference process of the model will be introduced.

## 3. Model Construction

In this case, PanguWeatherPredictor is implemented for inference of the ONNX model:

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

Among them, `input_file` and `input_surface_file` represent the upper-air meteorological data and surface meteorological data input to the network model respectively.

## 4. Result Visualization

First convert the data from npy to NetCDF format, then use ncvue for visualization

1. Install dependencies
```python
pip install cdsapi netCDF4 ncvue
```

2. Use script for data conversion
```python
python convert_data.py
```

3. Use ncvue to open the converted NetCDF file. For detailed instructions on ncvue, see [ncvue official documentation](https://github.com/mcuntz/ncvue)

## 5. Complete Code

``` py linenums="1" title="examples/pangu_weather/predict.py"
--8<--
examples/pangu_weather/predict.py
--8<--
```

## 6. Result Display

The figure below shows the temperature prediction results of the model. More indicators can be viewed using ncvue.

<figure markdown>
  ![result](https://paddle-org.bj.bcebos.com/paddlescience/docs/pangu-weather/temperature.png){ loading=lazy style="margin:0 auto;"}
  <figcaption>Temperature prediction result</figcaption>
</figure>

## 7. References

- [Pangu-Weather: A 3D High-Resolution System for Fast and Accurate Global Weather Forecast](https://arxiv.org/pdf/2211.02556)
