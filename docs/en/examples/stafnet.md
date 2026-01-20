# STAFNet: Spatiotemporal-Aware Fusion Network for Air Quality Prediction

| Pretrained Model                                                   | Metric                 |
| ------------------------------------------------------------ | -------------------- |
| [stafnet.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/stafnet/stafnet.pdparams) | MAE(1-48h) : 8.70933 |

=== "Model Training Command"

    ``` sh
    python stafnet.py DATASET.data_dir="Your train dataset path" EVAL.eval_data_path="Your evaluate dataset path"
    ```

=== "Model Evaluation Command"

    ``` sh
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/stafnet/val_data.pkl -P ./dataset/
    python stafnet.py mode=eval EVAL.pretrained_model_path="https://paddle-org.bj.bcebos.com/paddlescience/models/stafnet/stafnet.pdparams"
    ```

## 1. Background Introduction

In recent years, global urbanization and industrialization have inevitably led to serious air pollution problems. The high incidence of non-communicable diseases such as heart disease, asthma and lung cancer is directly related to exposure to air pollution. Therefore, air quality prediction has become a research hotspot in public health, national economy and urban management. A large number of monitoring stations have been established to monitor air quality, and their geographical locations and historical observation data are merged into spatiotemporal data. However, due to the high complexity of air pollution formation and dispersion, air quality prediction still faces some challenges.

First, the emission and dispersion of pollutants in the air will cause rapid deterioration of air quality in neighboring areas. This phenomenon is described as spatial dependence in Tobler's First Law of Geography. Modeling spatial relationships is crucial for predicting air quality. However, due to the sparse geographical distribution of air monitoring stations, it is challenging to capture the intrinsic spatial correlations in the data. Second, air quality is affected by complex multi-source factors, especially meteorological conditions. For example, long-term light wind or calm wind will inhibit the diffusion of air pollutants, while natural rainfall plays a role in removing and washing air pollutants. However, air quality stations and meteorological stations are located in different areas, resulting in misalignment of multi-modal features. Fusing misaligned multi-modal features and obtaining complementary information to accurately predict air quality is another challenge. Last but not least, changes in air quality have obvious multi-period characteristics. Utilizing this feature is very important to improve the accuracy of air quality prediction, but it is also challenging.

Many studies have been proposed for air quality prediction. Early methods focused on learning the temporal patterns of observation data from a single observation station, while abandoning the spatial relationship between observation stations. Recently, due to the effectiveness of Graph Neural Networks (GNN) in dealing with non-Euclidean graph structures, more and more methods have adopted GNN to simulate spatial dependencies. These methods use station locations as context features, implicitly model spatial dependencies, and do not make full use of the valuable spatial information contained in station locations and relationships between stations. In addition, existing spatiotemporal GNNs lack the ability to fuse multiple features in misaligned maps. Therefore, most methods require additional interpolation algorithms to align and connect meteorological features with AQ features at an early stage. This method eliminates the spatial and structural information between air quality stations and meteorological stations, and may also introduce noise leading to error accumulation. In addition, the problem of utilizing multi-periodicity in air quality prediction remains unexplored.

This case studies the application of spatiotemporal graph networks in the direction of air quality prediction.

## 2. Model Principle

STAFNet is a novel multi-modal forecasting framework--Spatiotemporal-Aware Fusion Network to predict air quality. STAFNet consists of three main parts: Spatially Aware GNN, Cross-Graph Fusion Attention Mechanism and TimesNet. Specifically, in order to capture the spatial relationship between stations, we first introduced Spatially Aware GNN to explicitly incorporate spatial information into information transmission and node representation. To comprehensively represent meteorological impacts, we subsequently proposed a multi-modal fusion strategy based on cross-graph fusion attention mechanism to integrate meteorological data into AQ data when the number and location of different types of stations are inconsistent. Inspired by multi-period analysis, we use TimesNet to decompose time series data into periodic signals of different frequencies and extract time features separately.

This chapter only briefly introduces the model principle of STAFNet. For detailed theoretical derivation, please read STAFNet: Spatiotemporal-Aware Fusion Network for Air Quality Prediction.

The overall structure of the model is shown in the figure:

![image-20240530165151443](https://paddle-org.bj.bcebos.com/paddlescience/docs/stafnet/model.jpg)

<div align = "center">STAFNet Network Model</div>

STAFNet contains three modules, which fuse spatial information, meteorological information and historical information into air quality feature representation respectively. First, the input of the model: **air quality** data and **meteorological** data at the past T moments, use two Spatially Aware GNNs (SAGNN) to extract air quality and meteorological information respectively using the spatial relationship between monitoring stations. Then, Cross-Graph Fusion Attention (CGF) fuses meteorological information into air quality representation. Finally, we use the TimesNet model to describe the temporal dynamics of the air quality sequence and generate multi-step predictions. This inference process can be expressed as follows,

![image-20240531173333183](https://paddle-org.bj.bcebos.com/paddlescience/docs/stafnet/Equation.jpg)

## 3. Model Construction

### 3.1 Dataset Introduction

The dataset uses the Beijing air quality dataset processed by STAFNet. The dataset includes:

(1) Air quality observations (i.e. PM2.5, PM10, O3, NO2, SO2 and CO);

(2) Meteorological observations (i.e. temperature, pressure, humidity, wind speed and wind direction);

(3) Station location (i.e. longitude and latitude).

All air quality and meteorological observation data are recorded every hour. The data collection time is from January 24, 2021 to January 19, 2023. The data is divided into training set and test set at a ratio of 9:1. Air quality observation data comes from the National Urban Air Quality Real-time Release Platform, and meteorological observation data comes from the China Meteorological Administration. Specific details of the dataset are shown in the table below:

<div>            <!--Block level encapsulation-->     <center>    <!--Center image and text-->     <img src="https://paddle-org.bj.bcebos.com/paddlescience/docs/stafnet/dataset.jpg" alt="image-20240530104042194" style="zoom: 25%;" />     <br>        <!--Line break-->     Beijing Air Quality Dataset    <!--Title-->     </center> </div>

Specific datasets can be downloaded from https://quotsoft.net/air/.

Before running the code for this problem, please download the [dataset](https://paddle-org.bj.bcebos.com/paddlescience/datasets/stafnet/val_data.pkl), and store it in the path after downloading:

```
./dataset
```

### 3.2 Model Building

In the STAFNet model, input the air quality data of 35 stations in the past 72 hours to predict the air quality of these 35 stations in the future 48 hours. In this problem, we use the neural network `stafnet` as the model, which receives graph structure data and outputs prediction results.

```py linenums="10" title="examples/stafnet/stafnet.py"
--8<--
examples/stafnet/stafnet.py:10:10
--8<--
```

### 3.3 Parameter and Hyperparameter Setting

The default settings of hyperparameters `cfg.MODEL.gat_hidden_dim`, `cfg.MODEL.e_layers`, `cfg.MODEL.d_model`, `cfg.MODEL.top_k` etc. are as follows:

``` yaml linenums="35" title="examples/stafnet/conf/stafnet.yaml"
--8<--
examples/stafnet/conf/stafnet.yaml:35:59
--8<--
```

### 3.4 Optimizer Construction

The training process calls the optimizer to update model parameters. The commonly used `Adam` optimizer is selected here.

``` py linenums="62" title="examples/stafnet/stafnet.py"
--8<--
examples/stafnet/stafnet.py:62:62
--8<--
```

The settings related to learning rate are as follows:

``` yaml linenums="70" title="examples/stafnet/conf/stafnet.yaml"
--8<--
examples/stafnet/conf/stafnet.yaml:70:75
--8<--
```

### 3.5 Constraint Construction

In this case, we use a supervised dataset to train the model, so we need to construct supervised constraints.

Before defining constraints, we need to specify relevant configurations such as the dataset path and store this information in the corresponding YAML file, as shown below.

``` yaml linenums="31" title="examples/stafnet/conf/stafnet.yaml"
--8<--
examples/stafnet/conf/stafnet.yaml:31:34
--8<--
```

Finally, construct the supervised constraint as shown below.

``` py linenums="46" title="examples/stafnet/stafnet.py"
--8<--
examples/stafnet/stafnet.py:46:51
--8<--
```

### 3.6 Validator Construction

During the training process, the training status of the current model is usually evaluated using the validation set (test set) at a certain epoch interval. Therefore, `ppsci.validate.SupervisedValidator` is used to construct the validator. The construction process is similar to [Constraint Construction 3.5](https://github.com/PaddlePaddle/PaddleScience/blob/develop/docs/zh/examples/stafnet.md#36), just change the data directory to the directory of the test set, and set `EVAL.batch_size=1` in the configuration file.

``` py linenums="52" title="examples/stafnet/stafnet.py"
--8<--
examples/stafnet/stafnet.py:52:58
--8<--
```

The evaluation metric is the MAE value of the predicted result and the real result, so `ppsci.metric.MAE()` built in PaddleScience is used, as shown below.

``` py linenums="55" title="examples/stafnet/stafnet.py"
--8<--
examples/stafnet/stafnet.py:55:55
--8<--
```

### 3.7 Model Training

Since this problem is a time series prediction problem, `psci.loss.MAELoss('mean')` built in PaddleScience can be used as the loss function for the training process. At the same time, stochastic gradient descent is selected to optimize the network. After completing the above settings, just pass the above instantiated objects to `ppsci.solver.Solver` in order, and then start training. Specific code is as follows:

``` py linenums="66" title="examples/stafnet/stafnet.py"
--8<--
examples/stafnet/stafnet.py:66:82
--8<--
```

## 4. Complete Code

```py linenums="1" title="examples/stafnet/stafnet.py"
--8<--
examples/stafnet/stafnet.py
--8<--
```

## 5. References

- [STAFNet: Spatiotemporal-Aware Fusion Network for Air Quality Prediction](https://link.springer.com/chapter/10.1007/978-3-031-78186-5_22)
