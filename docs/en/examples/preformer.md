# Preformer

Before starting training and evaluation, please download the [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download) dataset file.

Before starting evaluation, please download or train to generate a pre-trained model.

The dataset used for evaluation has been saved and can be downloaded and evaluated through the following links:
[rain_2016_01.h5](https://paddle-org.bj.bcebos.com/paddlescience/datasets/preformer/rain_2016_01.h5),
[ERA5_201601.tar.gz](https://paddle-org.bj.bcebos.com/paddlescience/datasets/meteoformer/ERA5_201601.tar.gz),
[mean.nc](https://paddle-org.bj.bcebos.com/paddlescience/datasets/climateformer/mean.nc),
[std.nc](https://paddle-org.bj.bcebos.com/paddlescience/datasets/climateformer/std.nc).

After downloading or decompressing, please maintain the following directory form:
ERA5/
├── mean.nc
├── std.nc
├── rain_2016_01.h5
└── 2016/
    ├── r_2016010100.npy
    ├── ...

=== "Model Training Command"

    ``` sh
    python main.py
    ```

=== "Model Evaluation Command"

    ``` sh
    python main.py mode=eval EVAL.pretrained_model_path="https://paddle-org.bj.bcebos.com/paddlescience/models/preformer/preformer.pdparams"
    ```

## 1. Background Introduction

Precipitation is a weather phenomenon closely related to human production and life. Accurate prediction of short-term precipitation not only provides key technical support for public services such as agricultural management, traffic planning, and disaster prevention, but is also a challenging academic research task. In recent years, deep learning has made major breakthroughs in the field of meteorological prediction. Taking multi-modal three-dimensional (altitude, longitude and latitude) meteorological data as the research object, researching short-term precipitation prediction methods based on deep learning has important theoretical research value and broad application prospects.

Preformer, a spatiotemporal Transformer network for short-term precipitation prediction, consists of an encoder, an evolver, and a decoder. Specifically, the encoder encodes spatial features by exploring dependencies between embeddings. Global temporal dynamics are learned from rearranged embeddings through the evolver. Finally, in the decoder, spatiotemporal representations are decoded into future precipitation.

## 2. Model Principle

This chapter briefly introduces the model principle of Preformer.

### 2.1 Encoder

This module uses two layers of Transformers to extract spatial features and update node features:

``` py linenums="243" title="ppsci/arch/preformer.py"
--8<--
ppsci/arch/preformer.py:243:277
--8<--
```

### 2.2 Evolver

This module uses two layers of Transformers to learn global temporal dynamics:

``` py linenums="280" title="ppsci/arch/preformer.py"
--8<--
ppsci/arch/preformer.py:280:325
--8<--
```

### 2.3 Decoder

This module uses two layers of convolution to decode spatiotemporal representations into future precipitation:

``` py linenums="329" title="ppsci/arch/preformer.py"
--8<--
ppsci/arch/preformer.py:329:344
--8<--
```

### 2.4 Preformer Model Structure

The overall structure of the model is shown in the figure:

<figure markdown>
  ![preformer-arch](https://paddle-org.bj.bcebos.com/paddlescience/docs/preformer/preformer.png){ loading=lazy style="margin:0 auto"}
  <figcaption>Preformer Network Model</figcaption>
</figure>

The Preformer model first uses a feature embedding layer to encode spatial features of input signals (meteorological elements of the past few hours):

``` py linenums="415" title="ppsci/arch/preformer.py"
--8<--
ppsci/arch/preformer.py:415:417
--8<--
```

Then the model uses the evolver to learn the dynamic characteristics of spatial features and predict the meteorological characteristics of the next few hours:

``` py linenums="419" title="ppsci/arch/preformer.py"
--8<--
ppsci/arch/preformer.py:419:422
--8<--
```

Finally, the model combines spatiotemporal dynamic characteristics with initial meteorological underlying features, and uses two layers of convolution to predict future short-term precipitation intensity:

``` py linenums="424" title="ppsci/arch/preformer.py"
--8<--
ppsci/arch/preformer.py:424:428
--8<--
```

## 3. Model Training

### 3.1 Dataset Introduction

The case uses the preprocessed ERA5SQ dataset, which belongs to a subset of ERA5 reanalysis data. ERA5SQ contains multiple variables of global atmosphere, land and ocean. The study area ranges from 140°E to 70°W, and from 55°N to the equator, with a spatial resolution of 0.25°. The dataset starts from 2016 to 2020, providing estimates of weather conditions every hour, which is very suitable for tasks such as precipitation prediction and analysis of total water vapor.

The dataset is saved as a T x C x H x W matrix, recording rainfall and meteorological element values at the corresponding location and time, where T is the time series length, C represents the channel dimension, the case selects meteorological information such as temperature, relative humidity, eastward wind speed, northward wind speed of 3 different pressure layers, H and W represent the height and width of the matrix divided by latitude and longitude. According to the year, the dataset is divided into training set, validation set, and test set at a ratio of 7:2:1. In the case, the mean and standard deviation of rainfall data, etc., were pre-calculated for subsequent regularization operations.

### 3.2 Model Training

#### 3.2.1 Model Construction

This case is implemented based on the Preformer model, expressed in PaddleScience code as follows:

``` py linenums="94" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py:94:95
--8<--
```

#### 3.2.2 Constraint Builder Construction

This case solves the problem based on data-driven methods, so it is necessary to use `SupervisedConstraint` built in PaddleScience to construct supervised constraint builders. Before defining the constraint builder, you need to first specify various parameters used for data loading in the constraint builder.

The code for loading training set data is as follows:

``` py linenums="23" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py:23:56
--8<--
```

The code for defining supervised constraints is as follows:

``` py linenums="58" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py:58:64
--8<--
```

#### 3.2.3 Validator Construction

In this case, the validation set is used to evaluate the training status of the current model at certain training epoch intervals during the training process, and `SupervisedValidator` is needed to construct the validator.

The code for loading validation set data is as follows:

``` py linenums="69" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py:69:80
--8<--
```

The code for defining supervised validator is as follows:

``` py linenums="82" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py:82:92
--8<--
```

#### 3.2.4 Learning Rate and Optimizer Construction

In this case, the learning rate size is set to `1e-3`, and the optimizer uses `Adam`, expressed in PaddleScience code as follows:

``` py linenums="97" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py:97:102
--8<--
```

#### 3.2.5 Model Training

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver`, and then start training.

``` py linenums="104" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py:104:121
--8<--
```

#### 3.2.6 Evaluation During Training

By setting the `eval_during_train` parameter in `ppsci.solver.Solver`, the model parameters with the best effect on the validation set can be automatically saved.

``` py linenums="113" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py:113:113
--8<--
```

### 3.3 Evaluating Model

#### 3.3.1 Validator Construction

The code for loading test set data is as follows:

``` py linenums="127" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py:127:138
--8<--
```

The code for defining supervised validator is as follows:

``` py linenums="140" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py:140:150
--8<--
```

Similar to `SupervisedValidator` of validation set, the evaluation indicators used here are `MAE` and `MSE`.

#### 3.3.2 Load Model and Evaluate

Set the loading path of pre-trained model parameters and load the model.

``` py linenums="152" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py:152:153
--8<--
```

Instantiate `ppsci.solver.Solver`, and then start evaluation.

``` py linenums="155" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py:155:166
--8<--
```

## 4. Complete Code

Dataset interface:

``` py linenums="1" title="ppsci/data/dataset/era5sq_dataset.py"
--8<--
ppsci/data/dataset/era5sq_dataset.py
--8<--
```

Model structure:

``` py linenums="1" title="ppsci/arch/preformer.py"
--8<--
ppsci/arch/preformer.py
--8<--
```

Model training:

``` py linenums="1" title="examples/preformer/main.py"
--8<--
examples/preformer/main.py
--8<--
```

Configuration file:

``` py linenums="1" title="examples/preformer/conf/preformer.yaml"
--8<--
examples/preformer/conf/preformer.yaml
--8<--
```

## 5. Result Display

The figure below shows the comparison between the prediction results of the Preformer model in the short-term precipitation prediction task and the ground truth results. The horizontal axis in the figure represents different time periods, with each time period interval being 1 hour, and the model predicts 6 frames of precipitation each time.

<figure markdown>
  ![result_precip](https://paddle-org.bj.bcebos.com/paddlescience/docs/preformer/result.png){ loading=lazy style="margin:0 auto;"}
  <figcaption>Preformer model prediction result ("Ours") vs ground truth result ("GT")</figcaption>
</figure>

## 6. References

- [Preformer: Simple and Efficient Design for Precipitation Nowcasting With Transformers](https://ieeexplore.ieee.org/document/10288072)
