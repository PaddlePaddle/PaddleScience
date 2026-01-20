# Climateformer

Before starting training and evaluation, please download the [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download) dataset file.

Before starting evaluation, please download or train to generate a pretrained model.

The 2018 data of the ERA5 dataset for evaluation has been saved and can be downloaded and evaluated through the following links:
[2018.h5](https://paddle-org.bj.bcebos.com/paddlescience/datasets/climateformer/2018.h5),
[mean.nc](https://paddle-org.bj.bcebos.com/paddlescience/datasets/climateformer/mean.nc),
[std.nc](https://paddle-org.bj.bcebos.com/paddlescience/datasets/climateformer/std.nc).

=== "Model Training Command"

    ``` sh
    python main.py
    ```

=== "Model Evaluation Command"

    ``` sh
    python main.py mode=eval EVAL.pretrained_model_path="https://paddle-org.bj.bcebos.com/paddlescience/models/climateformer/climateformer.pdparams"
    ```

## 1. Background Introduction

Long-term climate prediction mainly involves predicting weather changes in the next few weeks or even months. Such predictions usually need to cover multiple meteorological elements, such as temperature, humidity, wind speed, etc., which have complex spatiotemporal dependencies on meteorological changes. Accurate climate prediction is of great significance for disaster prevention and mitigation, agricultural production, aerospace and other fields. Traditional meteorological prediction models mainly rely on physical formulas and Numerical Weather Prediction (NWP), but with the rapid development of deep learning, data-driven models are gradually showing stronger prediction capabilities.

Climateformer is a spatiotemporal deep learning framework for long-term climate prediction. The design goal of this model is to learn and simulate the evolution paradigm of meteorological systems over long periods. Its architecture usually contains three modules: the encoder encodes multi-source and multi-sphere climate variables such as sea temperature, air pressure, and wind field into a global vector that can represent the current "climate state". The core evolution module (based on Transformer structure) is dedicated to capturing the long-range time dependence of these climate states across weeks or even months. Finally, the decoder predicts the average key climate indices for multiple future cycles based on the evolved state vector. Through Climateformer, climate forecasting can achieve more efficient and accurate multi-element prediction, providing more reliable data support for meteorological services.

## 2. Model Principle

This chapter briefly introduces the model principle of Climateformer.

### 2.1 Encoder

This module uses two layers of Transformer to extract spatial features and update node features:

``` py linenums="243" title="ppsci/arch/climateformer.py"
--8<--
ppsci/arch/climateformer.py:243:277
--8<--
```

### 2.2 Evolver

This module uses two layers of Transformer to learn global temporal dynamic characteristics:

``` py linenums="280" title="ppsci/arch/climateformer.py"
--8<--
ppsci/arch/climateformer.py:280:325
--8<--
```

### 2.3 Decoder

This module uses two layers of convolution to decode spatiotemporal representations into future multi-meteorological elements:

``` py linenums="329" title="ppsci/arch/climateformer.py"
--8<--
ppsci/arch/climateformer.py:329:344
--8<--
```

### 2.4 Climateformer Model Structure

The overall structure of the model is shown in the figure:

<figure markdown>
  ![climateformer-arch](https://paddle-org.bj.bcebos.com/paddlescience/docs/climateformer/climateformer.png){ loading=lazy style="margin:0 auto"}
  <figcaption>Climateformer Network Model</figcaption>
</figure>

The Climateformer model first uses a feature embedding layer to encode spatial features of input signals (average time frames of multiple meteorological elements over the past few weeks):

``` py linenums="418" title="ppsci/arch/climateformer.py"
--8<--
ppsci/arch/climateformer.py:418:420
--8<--
```

Then the model uses the evolver to learn the dynamic characteristics of spatial features and predict the meteorological characteristics of the average time frames of the next few weeks:

``` py linenums="422" title="ppsci/arch/climateformer.py"
--8<--
ppsci/arch/climateformer.py:422:425
--8<--
```

Finally, the model combines spatiotemporal dynamic characteristics with initial underlying meteorological features, and uses two layers of convolution to predict the weekly average values of multiple meteorological elements for the next few weeks to months:

``` py linenums="427" title="ppsci/arch/climateformer.py"
--8<--
ppsci/arch/climateformer.py:427:429
--8<--
```

## 3. Model Training

### 3.1 Dataset Introduction

The case uses the preprocessed ERA5Climate dataset, which is a subset of the ERA5 reanalysis data. ERA5Climate contains multiple variables of global atmosphere, land and ocean. The study area ranges from 140°E to 70°W, from 55°N to the equator, with a spatial resolution of 0.25°. The dataset runs from 2016 to 2020, providing hourly estimates of weather conditions, which is very suitable for tasks such as short- and medium-term multi-meteorological element prediction. In practical applications, the time interval is one week, and each frame is selected as the weekly average value within 7*24 hours.

The dataset is saved as a T x C x H x W matrix, recording the values of corresponding meteorological elements at the corresponding location and time, where T is the time series length, C represents the channel dimension (in the case, meteorological information such as temperature, relative humidity, eastward wind speed, northward wind speed at 3 different pressure levels are selected), H and W represent the height and width of the matrix divided by latitude and longitude. According to the year, the dataset is divided into training set, validation set, and test set at a ratio of 7:2:1. In the case, the mean and standard deviation of meteorological element data are pre-calculated for subsequent regularization operations.

### 3.2 Model Training

#### 3.2.1 Model Construction

This case is implemented based on the Climateformer model, expressed in PaddleScience code as follows:

``` py linenums="97" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py:97:98
--8<--
```

#### 3.2.2 Constraint Construction

This case solves the problem based on data-driven methods, so it is necessary to use `SupervisedConstraint` built in PaddleScience to construct a supervised constraint. Before defining the constraint, you need to first specify each parameter used for data loading in the constraint.

The code for loading training set data is as follows:

``` py linenums="23" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py:23:58
--8<--
```

The code for defining supervised constraint is as follows:

``` py linenums="60" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py:60:66
--8<--
```

#### 3.2.3 Validator Construction

In this case, the training process will use the validation set to evaluate the training status of the current model at a certain training epoch interval, and `SupervisedValidator` needs to be used to construct the validator.

The code for loading validation set data is as follows:

``` py linenums="71" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py:71:83
--8<--
```

The code for defining supervised validator is as follows:

``` py linenums="85" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py:85:95
--8<--
```

#### 3.2.4 Learning Rate and Optimizer Construction

In this case, the learning rate is set to `1e-3`, and the optimizer uses `Adam`, expressed in PaddleScience code as follows:

``` py linenums="100" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py:100:105
--8<--
```

#### 3.2.5 Model Training

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training.

``` py linenums="107" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py:107:123
--8<--
```

#### 3.2.6 Evaluation During Training

By setting the `eval_during_train` parameter in `ppsci.solver.Solver`, the model parameters with the best effect on the validation set can be automatically saved.

``` py linenums="116" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py:116:116
--8<--
```

### 3.3 Evaluation Model

#### 3.3.1 Validator Construction

The code for loading test set data is as follows:

``` py linenums="129" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py:129:141
--8<--
```

The code for defining supervised validator is as follows:

``` py linenums="143" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py:143:153
--8<--
```

Similar to `SupervisedValidator` for validation set, the evaluation metrics used here are `MAE` and `MSE`.

#### 3.3.2 Load Model and Evaluate

Set the loading path of pretrained model parameters and load the model.

``` py linenums="155" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py:155:156
--8<--
```

Instantiate `ppsci.solver.Solver`, and then start evaluation.

``` py linenums="158" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py:158:169
--8<--
```

## 4. Complete Code

Dataset Interface:

``` py linenums="1" title="ppsci/data/dataset/era5climate_dataset.py"
--8<--
ppsci/data/dataset/era5climate_dataset.py
--8<--
```

Model Structure:

``` py linenums="1" title="ppsci/arch/climateformer.py"
--8<--
ppsci/arch/climateformer.py
--8<--
```

Model Training:

``` py linenums="1" title="examples/climateformer/main.py"
--8<--
examples/climateformer/main.py
--8<--
```

Configuration File:

``` py linenums="1" title="examples/climateformer/conf/climateformer.yaml"
--8<--
examples/climateformer/conf/climateformer.yaml
--8<--
```

## 5. Result Display

The figure below shows the comparison between the prediction results of the Climateformer model and the ground truth in the temperature prediction task at 1000 hPa isobaric layer. The horizontal axis represents different prediction time steps, with a time interval of 1 week, and each time the model predicts the weekly average value for the next 6 weeks.

<figure markdown>
  ![result](https://paddle-org.bj.bcebos.com/paddlescience/docs/climateformer/result.png){ loading=lazy style="margin:0 auto;"}
  <figcaption>Climateformer model prediction results ("Pred") vs ground truth results ("GT")</figcaption>
</figure>
