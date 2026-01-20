# Meteoformer

Before starting training and evaluation, please download the [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download) dataset file.

Before starting evaluation, please download or train to generate a pre-trained model.

The dataset used for evaluation has been saved and can be downloaded and evaluated via the following links:
[ERA5_201601.tar.gz](https://paddle-org.bj.bcebos.com/paddlescience/datasets/meteoformer/ERA5_201601.tar.gz),
[mean.nc](https://paddle-org.bj.bcebos.com/paddlescience/datasets/climateformer/mean.nc),
[std.nc](https://paddle-org.bj.bcebos.com/paddlescience/datasets/climateformer/std.nc).

After downloading or unzipping, please maintain the following directory structure:
ERA5/
├── mean.nc
├── std.nc
└── 2016/
    ├── r_2016010100.npy
    ├── ...

=== "Model Training Command"

    ``` sh
    python main.py
    ```

=== "Model Evaluation Command"

    ``` sh
    python main.py mode=eval EVAL.pretrained_model_path="https://paddle-org.bj.bcebos.com/paddlescience/models/meteoformer/meteoformer.pdparams"
    ```

## 1. Background Introduction

Short-to-medium-range weather forecasting mainly involves predicting weather changes within the next few hours to days. Such forecasts typically need to cover multiple meteorological elements, such as temperature, humidity, wind speed, etc., which have complex spatiotemporal dependencies on weather changes. Accurate short-to-medium-range weather forecasting is of great significance for disaster prevention and mitigation, agricultural production, aerospace and other fields. Traditional weather forecasting models mainly rely on physical formulas and Numerical Weather Prediction (NWP), but with the rapid development of deep learning, data-driven models have gradually shown stronger predictive capabilities.

In order to effectively capture these multidimensional spatiotemporal features, Meteoformer came into being. Meteoformer is a model based on the Transformer architecture, specifically optimized for short-to-medium-range multi-meteorological element prediction tasks. This model can handle the spatiotemporal dependencies of multiple meteorological variables and uses a self-attention mechanism to capture correlations at different spatiotemporal scales, thereby achieving more accurate multi-step predictions of meteorological elements such as temperature, humidity, and wind speed. Through Meteoformer, weather forecasting can achieve more efficient and precise multi-element prediction, providing more reliable data support for meteorological services.

## 2. Model Principle

This chapter briefly introduces the model principle of Meteoformer.

### 2.1 Encoder

This module uses a two-layer Transformer to extract spatial features and update node features:

``` py linenums="243" title="ppsci/arch/meteoformer.py"
--8<--
ppsci/arch/meteoformer.py:243:277
--8<--
```

### 2.2 Evolver

This module uses a two-layer Transformer to learn global temporal dynamics:

``` py linenums="280" title="ppsci/arch/meteoformer.py"
--8<--
ppsci/arch/meteoformer.py:280:325
--8<--
```

### 2.3 Decoder

This module uses two layers of convolution to decode spatiotemporal representations into future multi-meteorological elements:

``` py linenums="329" title="ppsci/arch/meteoformer.py"
--8<--
ppsci/arch/meteoformer.py:329:344
--8<--
```

### 2.4 Meteoformer Model Structure

The overall structure of the model is shown in the figure:

<figure markdown>
  ![meteoformer-arch](https://paddle-org.bj.bcebos.com/paddlescience/docs/climateformer/climateformer.png){ loading=lazy style="margin:0 auto"}
  <figcaption>Meteoformer Model Structure</figcaption>
</figure>

The Meteoformer model first uses a feature embedding layer to encode spatial features of the input signal (multi-meteorological elements from the past few time frames):

``` py linenums="418" title="ppsci/arch/meteoformer.py"
--8<--
ppsci/arch/meteoformer.py:418:420
--8<--
```

Then the model uses the evolver to learn the dynamic characteristics of spatial features and predict the meteorological features of the next few time frames:

``` py linenums="422" title="ppsci/arch/meteoformer.py"
--8<--
ppsci/arch/meteoformer.py:422:425
--8<--
```

Finally, the model combines spatiotemporal dynamics with initial meteorological underlying features, and uses two layers of convolution to predict multi-meteorological element values in the future short-to-medium term:

``` py linenums="427" title="ppsci/arch/meteoformer.py"
--8<--
ppsci/arch/meteoformer.py:427:429
--8<--
```

## 3. Model Training

### 3.1 Dataset Introduction

The case uses the preprocessed ERA5Meteo dataset, which is a subset of the ERA5 reanalysis data. ERA5Meteo contains multiple variables of global atmosphere, land and ocean. The study area ranges from 140°E to 70°W, and from 55°N to the equator, with a spatial resolution of 0.25°. The dataset provides estimates of weather conditions every hour from 2016 to 2020, making it very suitable for tasks such as short-to-medium-range multi-meteorological element prediction. In practical applications, the time interval is selected as 1 hour.

The dataset is saved as a T x C x H x W matrix, recording the values of corresponding meteorological elements at corresponding locations and times, where T is the length of the time series, C represents the channel dimension, and the case selects meteorological information such as temperature, relative humidity, eastward wind speed, and northward wind speed at 3 different pressure levels. H and W represent the height and width of the matrix divided by latitude and longitude. According to the year, the dataset is divided into training set, validation set, and test set in a ratio of 7:2:1. The mean and standard deviation of meteorological element data are pre-calculated in the case for subsequent normalization operations.

### 3.2 Model Training

#### 3.2.1 Model Construction

This case is implemented based on the Meteoformer model, expressed in PaddleScience code as follows:

``` py linenums="94" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:94:95
--8<--
```

#### 3.2.2 Constraint Construction

This case solves the problem based on data-driven methods, so it is necessary to use `SupervisedConstraint` built in PaddleScience to construct supervised constraints. Before defining constraints, you need to first specify various parameters used for data loading in constraints.

Training set data loading code is as follows:

``` py linenums="23" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:23:56
--8<--
```

The code for defining supervised constraints is as follows:

``` py linenums="58" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:58:64
--8<--
```

#### 3.2.3 Validator Construction

During the training process of this case, the training status of the current model will be evaluated using the validation set at certain training round intervals, and `SupervisedValidator` is needed to construct the validator.

Validation set data loading code is as follows:

``` py linenums="69" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:69:80
--8<--
```

The code for defining supervised validator is as follows:

``` py linenums="82" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:82:92
--8<--
```

#### 3.2.4 Learning Rate and Optimizer Construction

The learning rate size used in this case is set to `1e-3`. The optimizer uses `Adam`, expressed in PaddleScience code as follows:

``` py linenums="97" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:97:102
--8<--
```

#### 3.2.5 Model Training

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver`, and then start training.

``` py linenums="104" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:104:120
--8<--
```

#### 3.2.6 Evaluation During Training

By setting the `eval_during_train` parameter in `ppsci.solver.Solver`, the model parameters with the best effect on the validation set can be automatically saved.

``` py linenums="113" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:113:113
--8<--
```

### 3.3 Evaluation Model

#### 3.3.1 Validator Construction

Test set data loading code is as follows:

``` py linenums="126" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:126:137
--8<--
```

The code for defining supervised validator is as follows:

``` py linenums="139" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:139:149
--8<--
```

Similar to `SupervisedValidator` for validation set, the evaluation metrics used here are `MAE` and `MSE`.

#### 3.3.2 Load Model and Evaluate

Set the loading path of pre-trained model parameters and load the model.

``` py linenums="151" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:151:152
--8<--
```

Instantiate `ppsci.solver.Solver`, and then start evaluation.

``` py linenums="154" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py:154:165
--8<--
```

## 4. Complete Code

Dataset interface:

``` py linenums="1" title="ppsci/data/dataset/era5meteo_dataset.py"
--8<--
ppsci/data/dataset/era5meteo_dataset.py
--8<--
```

Model structure:

``` py linenums="1" title="ppsci/arch/meteoformer.py"
--8<--
ppsci/arch/meteoformer.py
--8<--
```

Model training:

``` py linenums="1" title="examples/meteoformer/main.py"
--8<--
examples/meteoformer/main.py
--8<--
```

Configuration file:

``` py linenums="1" title="examples/meteoformer/conf/meteoformer.yaml"
--8<--
examples/meteoformer/conf/meteoformer.yaml
--8<--
```

## 5. Result Display

The figure below shows the comparison between the prediction results of the Meteoformer model in the 1000 hPa layer wind speed prediction task and the ground truth results. The horizontal axis represents different prediction time steps, the time interval is 1 hour, and the model can predict the future 6 time steps at a time.

<figure markdown>
  ![result_precip](https://paddle-org.bj.bcebos.com/paddlescience/docs/meteoformer/result.png){ loading=lazy style="margin:0 auto;"}
  <figcaption>Meteoformer model prediction result ("Pred") vs ground truth result ("GT")</figcaption>
</figure>
