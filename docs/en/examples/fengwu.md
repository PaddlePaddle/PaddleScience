# FengWu

=== "Model Training Command"

    None

=== "Model Evaluation Command"

    None

=== "Model Export Command"

    None

=== "Model Inference Command"

    ``` sh
    # Download sample input data
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Fengwu/input1.npy -P ./data
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Fengwu/input2.npy -P ./data

    # Download pretrain model weight
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/Fengwu/fengwu_v2.onnx -P ./inference

    # inference
    python predict.py
    ```

## 1. Background Introduction

With the intensification of global climate change and the frequent occurrence of extreme weather in recent years, the expectations of all sectors for the timeliness and accuracy of weather forecasts are increasing day by day. How to improve the timeliness and accuracy of weather forecasts has always been a key topic in the industry. The AI large model "FengWu" is built based on multi-modal and multi-task deep learning methods, achieving effective forecasting of core atmospheric variables for more than 10 days at high resolution, and surpassing GraphCast, a model released by DeepMind, on 80% of the evaluation indicators. At the same time, "FengWu" can generate high-precision global forecast results for the next 10 days in just 30 seconds, which is significantly better than traditional models in efficiency.

## 2. Model Principle

This chapter only briefly introduces the principle of the FengWu meteorological large model. For detailed theoretical derivation, please read [FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead](https://arxiv.org/pdf/2304.02948).

The overall structure of the model is shown in the figure:

<figure markdown>
  ![result](https://paddle-org.bj.bcebos.com/paddlescience/docs/fengwu/model_architecture.png){ loading=lazy style="margin:0 auto;"}
  <figcaption>Model Structure</figcaption>
</figure>

The model takes climate variables as multi-modal inputs. The features of multiple modalities are encoded in the `Modal-Customized Encoder`, and the encoded features are fused using the Transformer-based `Cross-modal Fuser` to obtain a joint representation. Finally, climate variables are predicted separately from the joint representation in the `Modal-Customized Decoder`.

The model uses pre-trained weights for inference. Next, the inference process of the model will be introduced.

## 3. Model Construction

In this case, FengWuPredictor is implemented for inference of the ONNX model:

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

Among them, `input_file` and `input_next_file` represent the meteorological data at the start time and the meteorological data 6 hours later input to the network model respectively.

## 4. Result Visualization

The model inference result contains 56 npy files, representing meteorological data every 6 hours for the next 14 days starting from the prediction time point. Result visualization requires first converting the data from npy to NetCDF format, and then using ncvue for viewing.

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

``` py linenums="1" title="examples/fengwu/predict.py"
--8<--
examples/fengwu/predict.py
--8<--
```

## 6. Result Display

The figure below shows the model's prediction result of the average sea level pressure for the next 6 hours. More indicators can be viewed using ncvue.

<figure markdown>
  ![result](https://paddle-org.bj.bcebos.com/paddlescience/docs/fengwu/image.png){ loading=lazy style="margin:0 auto;"}
  <figcaption>Average sea level pressure in the next 6 hours</figcaption>
</figure>

## 7. References

- [FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead](https://arxiv.org/pdf/2304.02948)
