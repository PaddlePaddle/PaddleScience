# Extformer-MoE

!!! note

    1. Before starting training and evaluation, please download the [ICAR-ENSO dataset](https://tianchi.aliyun.com/dataset/98942) and modify `FILE_PATH` in the yaml configuration file to the path of the decompressed dataset.
    2. Before starting training and evaluation, please install `xarray` and `h5netcdf`: `pip install requirements.txt`
    3. If video memory is insufficient during training, you can specify `MODEL.checkpoint_level` as `1` or `2`, then run in recompute mode to trade training time for video memory.

=== "Model Training Command"

    ``` sh
    # ICAR-ENSO data pre-trained model: Extformer-MoE
    python extformer_moe_enso_train.py
    # python extformer_moe_enso_train.py MODEL.checkpoint_level=1 # using recompute to run in device with small GPU memory
    # python extformer_moe_enso_train.py MODEL.checkpoint_level=2 # using recompute to run in device with small GPU memory
    ```

=== "Model Evaluation Command"

    ``` sh
    # ICAR-ENSO model evaluation: Extformer-MoE
    python extformer_moe_enso_train.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/extformer-moe/extformer_moe_pretrained.pdparams
    ```

| Model | Variable Name | C-Nino3.4-M | C-Nino3.4-WM | MSE(1E-4) | MAE(1E-1) | RMSE |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| [Extformer-MoE](https://paddle-org.bj.bcebos.com/paddlescience/models/extformer-moe/extformer_moe_pretrained.pdparams) | sst | 0.7651 | 2.39771 | 3.0000 | 0.1291 | 0.50243 |

## 1. Background Introduction

The Earth is a complex system. Variations in the Earth system, ranging from routine events like temperature fluctuations to extreme events like droughts, hail, and El Niño/Southern Oscillation (ENSO), affect our daily lives. Among all consequences, Earth system changes affect crop yields, flight delays, trigger floods and forest fires. Accurate and timely forecasting of these changes can help people take necessary precautions to avoid crises or make better use of natural resources such as wind and solar energy. Therefore, improving prediction models for Earth changes (such as weather and climate) has huge socio-economic impact.

In recent years, deep learning models have shown great potential in weather and climate forecasting tasks. Compared with traditional numerical simulation methods, deep learning methods achieve significant improvements in prediction efficiency and accuracy by utilizing emerging technologies such as visual neural networks (ViT) or graph neural networks (GNN) to learn complex mapping relationships between current and future weather or climate states directly from massive reanalysis data. However, extreme events occurring in Earth changes often present characteristics such as long-range spatiotemporal synchronous correlation, diverse spatiotemporal distribution patterns, and sparse extreme value observation signals, which bring many new technical challenges to the construction of deep learning-based Earth system extreme event prediction models.

### 1.1 Long-range Spatiotemporal Synchronous Correlation

Facing the complex coupled Earth change system, existing technologies based on visual and graph deep learning have many deficiencies in modeling the long-range spatiotemporal correlation presented by extreme weather. Specifically, intelligent forecasting models based on visual deep learning (such as Huawei's Pangu weather model) are limited to calculating information interaction within local regions and cannot efficiently utilize global information from distant regions. In contrast, weather forecasting methods based on graph neural networks (such as Google's GraphCast) can disseminate long-range information through predefined graph structures. However, prior graph structures are difficult to effectively identify key long-range information affecting extreme weather and are susceptible to noise, leading to biased or even incorrect prediction results by the model. In addition, meteorological data of the Earth system generally has massive grid points. While mining global long-range spatiotemporal correlation information, it may lead to a surge in model complexity. How to efficiently model long-range correlations in spatiotemporal data has become a major challenge for Earth system extreme event prediction.

Earthformer, a space-time transformer for Earth system prediction. To better explore the design of space-time attention, Cuboid Attention is designed, which is a generic building block for efficient space-time attention. The idea is to decompose the input tensor into non-overlapping cuboids and apply cuboid-level self-attention in parallel. Since we restrict the O(N<sup>2</sup>) self-attention to local cuboids, the overall complexity of the model is greatly reduced. Different types of correlations can be captured by different cuboid decompositions. At the same time, Earthformer introduces a set of global vectors that attend to all local cuboids, thereby gathering the overall state of the system. By attending to global vectors, local cuboids can grasp the overall dynamics of the system and share information with each other, thereby capturing long-range correlation information of the Earth system.

### 1.2 Diverse Spatiotemporal Distribution Patterns

Accurately modeling the diversity of spatiotemporal distribution patterns is the key to improving the prediction of extreme events in the Earth system. Existing methods use shared parameters in both time and space domains, and cannot effectively capture extreme weather feature patterns unique to specific time periods and geographical locations.

Mixture-of-Experts (MoE) network contains a set of expert networks and a gating network. Each expert network is an independent neural network with independent parameters, and the gating network adaptively selects a unique subset of expert networks for each input unit. During training and inference, each input unit only needs to utilize a small subset of expert networks, so the total number of expert networks can be expanded, enhancing the model's expressive power while maintaining relatively small computational complexity. In the Earth system, MoE can enhance the model's ability to capture spatiotemporal distribution differences by learning unique parameter sets related to time, geographical location, and model input.

### 1.3 Sparse Extreme Value Observation Signals

The uneven distribution of meteorological data will lead to the model being biased towards predicting frequent normal meteorological conditions, while underestimating extreme conditions with few observations, because regression loss functions commonly used in model training, such as mean square error (MSE) loss, will lead to over-smoothing of prediction results. Unlike imbalanced classification problems with discrete label spaces, imbalanced regression problems have continuous label spaces, posing greater challenges for extreme prediction problems.

Rank-N-Contrast (RNC) is a representation learning method designed to learn a regression-aware sample representation that sorts the distance between samples in the embedding space based on the distance in the continuous label space, and then uses it to predict the final continuous label. In the Earth system extreme prediction problem, RNC can regulate the representation of meteorological data so that it satisfies the continuity of the embedding space and aligns with the label space, ultimately alleviating the over-smoothing problem of extreme event prediction results.

## 2. Model Principle

### 2.1 Earthformer

This chapter only briefly introduces the model principle of EarthFormer. For detailed theoretical derivation, please read [Earthformer: Exploring Space-Time Transformers for Earth System Forecasting](https://arxiv.org/abs/2207.05833).

The Earthformer network model uses a hierarchical Encoder-Decoder architecture Transformer based on Cuboid Attention, which decomposes data into cuboids and applies cuboid-level self-attention in parallel. These cuboids further interact with a collection of global vectors to capture global information.

The overall structure of Earthformer is shown in the figure:

<center class ='img'>
<img title="Earthformer" src="https://paddle-org.bj.bcebos.com/paddlescience/docs/extformer-moe/Earthformer.png" width="60%">
</center>

### 2.2 Mixture-of-Experts

This chapter only briefly introduces the principle of Mixture-of-Experts. For detailed theoretical derivation, please read [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538).

Mixture-of-Experts (MoE) network contains a set of expert networks $E_1, E_2, ..., E_n$ with independent parameters and a gating network $G$. Given input $x$, the output of the MoE network is $y=\sum_{i=1}^n G(x)_iE_i(x)$.

The overall structure of MoE is shown in the figure:

<center class ='img'>
<img title="MoE" src="https://paddle-org.bj.bcebos.com/paddlescience/docs/extformer-moe/MoE.png" width="60%">
</center>

### 2.3 Rank-N-Contrast

Rank-N-Contrast (RNC) is a regression method that learns continuous representations through contrast based on the ranking of samples relative to each other in the label space. A simple example of RNC is shown in the figure:

<center class ='img'>
<img title="RNC" src="https://paddle-org.bj.bcebos.com/paddlescience/docs/extformer-moe/RNC.png" width="70%">
</center>

### 2.4 Training and Inference Process of Extformer-MoE Model

The model pre-training phase trains the model based on randomly initialized network weights, as shown in the figure below, where $[x_{i}]_{i=1}^{T}$ represents input meteorological data of a spatiotemporal sequence of length $T$, $[y_{i}]_{i=1}^{K}$ represents predicted meteorological data for future $K$ steps, and $[y_{i_True}]_{i=1}^{K}$ represents true data for future $K$ steps, such as sea surface temperature data and vertically integrated liquid data. Finally, the mse loss function is calculated for the network model prediction output and the ground truth. In the inference phase, given data of sequence length $T$, obtain prediction results of sequence length $K$.

## 3. Implementation of Sea Surface Temperature Model

Next, we will explain how to implement Extformer-MoE model training and inference based on PaddleScience code. For other details in this case, please refer to [API Documentation](../api/arch.md).

### 3.1 Dataset Introduction

The dataset uses the ICAR-ENSO dataset processed by [EarthFormer](https://github.com/amazon-science/earth-forecasting-transformer/tree/main).

This dataset is provided by the Institute for Climate and Application Research (ICAR). The data includes historical simulation data from CMIP5/6 models and nearly 100 years of historical observation assimilation data reconstructed by the US SODA model. Each sample contains the following meteorological and spatiotemporal variables: Sea Surface Temperature anomaly (SST), Heat Content anomaly (T300), Zonal Wind anomaly (Ua), Meridional Wind anomaly (Va), data dimension is (year, month, lat, lon). Training data provides Nino3.4 index label data for the corresponding month. The initial field data used for testing are n segments of 12 time series randomly extracted from multiple international ocean data assimilation results, and the data format is saved in NPY format.

**Training Data:**

The first dimension (year) of each data sample represents the starting year corresponding to the data. For CMIP data, there are a total of 291 years, of which 1-2265 are 151 years of historical simulation data provided by 15 models in CMIP6 (Total: 151 years * 15 models = 2265); 2266-4645 are 140 years of historical simulation data provided by 17 models in CMIP5 (Total: 140 years * 17 models = 2380). For historical observation assimilation data, it is SODA data provided by the United States.

**Training Data Label**

The label data is the Nino3.4 SST anomaly index, data dimension is (year, month).

The label data corresponding to CMIP(SODA)_train.nc is the three-month moving average of the Nino3.4 SST anomaly index at the current moment, so the data dimension and dimension introduction are consistent with the training data.

Note: The three-month moving average is the average of the current month and the next two months.

**Test Data**

The initial field (input) data used for testing are n segments of 12 time series randomly extracted from multiple international ocean data assimilation results. The data format is saved in NPY format, with dimensions (12, lat, lon, 4), 12 is time t and past 11 moments, 4 are predictors, stored in the order of SST, T300, Ua, Va.

In the training of the EarthFormer model for the ICAR-ENSO dataset, only Sea Surface Temperature (SST) is trained and predicted. Training SST anomaly observations for 12 steps (one year), predicting SST anomalies for up to 14 steps.

### 3.2 Model Pretraining

#### 3.2.1 Constraint Construction

This case solves the problem based on data-driven methods, so it is necessary to use `SupervisedConstraint` built in PaddleScience to construct supervised constraints. Before defining constraints, you need to first specify various parameters used for data loading in supervised constraints.

Data loading code is as follows:

``` py linenums="25" title="examples/extformer_moe/extformer_moe_enso_train.py"
--8<--
examples/extformer_moe/extformer_moe_enso_train.py:25:47
--8<--
```

Among them, the "dataset" field defines the `Dataset` class name used as `ExtMoEENSODataset`, the "sampler" field defines the `Sampler` class name used as `BatchSampler`, `batch_size` is set to 16, and `num_works` is 8.

The code for defining supervised constraints is as follows:

``` py linenums="49" title="examples/extformer_moe/extformer_moe_enso_train.py"
--8<--
examples/extformer_moe/extformer_moe_enso_train.py:49:55
--8<--
```

The first parameter of `SupervisedConstraint` is the data loading method, here `train_dataloader_cfg` defined above is used;

The second parameter is the definition of loss function, here a custom loss function is used;

The third parameter is the name of the constraint condition, which is convenient for subsequent indexing. Here it is named `Sup`.

#### 3.2.2 Model Construction

In this case, the sea surface temperature model is implemented based on the ExtFormerMoECuboid network model, expressed in PaddleScience code as follows:

``` py linenums="88" title="examples/extformer_moe/extformer_moe_enso_train.py"
--8<--
examples/extformer_moe/extformer_moe_enso_train.py:88:92
--8<--
```

The parameters of the network model are set through the configuration file as follows:

``` yaml linenums="47" title="examples/earthformer/conf/earthformer_enso_pretrain.yaml"
--8<--
examples/extformer_moe/conf/extformer_moe_enso_pretrain.yaml:47:129
--8<--
```

Among them, `input_keys` and `output_keys` represent the names of input and output variables of the network model respectively.

#### 3.2.3 Learning Rate and Optimizer Construction

The learning rate method used in this case is `Cosine`, and the learning rate size is set to `2e-4`. The optimizer uses `AdamW`, and groups parameters to use different `weight_decay`, expressed in PaddleScience code as follows:

``` py linenums="94" title="examples/extformer_moe/extformer_moe_enso_train.py"
--8<--
examples/extformer_moe/extformer_moe_enso_train.py:94:119
--8<--
```

#### 3.2.4 Validator Construction

During the training process of this case, the training status of the current model will be evaluated using the validation set at certain training round intervals, and `SupervisedValidator` is needed to construct the validator. The code is as follows:

``` py linenums="59" title="examples/extformer_moe/extformer_moe_enso_train.py"
--8<--
examples/extformer_moe/extformer_moe_enso_train.py:59:86
--8<--
```

The `SupervisedValidator` validator is quite similar to `SupervisedConstraint`, the difference is that the validator needs to set evaluation metric `metric`, here custom evaluation metrics `MAE`, `MSE`, `RMSE`, `corr_nino3.4_epoch` and `corr_nino3.4_weighted_epoch` are used.

#### 3.2.5 Model Training and Evaluation

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver`, and then start training and evaluation.

``` py linenums="121" title="examples/extformer_moe/extformer_moe_enso_train.py"
--8<--
examples/extformer_moe/extformer_moe_enso_train.py:121:137
--8<--
```

### 3.3 Model Evaluation

The code for building the model is:

``` py linenums="138" title="examples/extformer_moe/extformer_moe_enso_train.py"
--8<--
examples/extformer_moe/extformer_moe_enso_train.py:138:139
--8<--
```

The code for building the validator is:

``` py linenums="142" title="examples/extformer_moe/extformer_moe_enso_train.py"
--8<--
examples/extformer_moe/extformer_moe_enso_train.py:142:182
--8<--
```

## 4. Complete Code

``` py linenums="1" title="examples/extformer_moe/extformer_moe_enso_train.py"
--8<--
examples/extformer_moe/extformer_moe_enso_train.py
--8<--
```
