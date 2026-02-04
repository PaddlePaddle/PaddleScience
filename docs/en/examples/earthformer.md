# EarthFormer

Before starting training and evaluation, please download the following datasets

[ICAR-ENSO Dataset](https://tianchi.aliyun.com/dataset/98942)

[SEVIR Dataset](https://nbviewer.org/github/MIT-AI-Accelerator/eie-sevir/blob/master/examples/SEVIR_Tutorial.ipynb#download)

And install required dependencies:

``` py
pip install -r requirements.txt
```

=== "Model Training Command"

    ``` sh
    # ICAR-ENSO data model training
    python earthformer_enso_train.py
    # SEVIR data model training
    python earthformer_sevir_train.py
    ```

=== "Model Evaluation Command"

    ``` sh
    # ICAR-ENSO model evaluation
    python earthformer_enso_train.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/earthformer/earthformer_enso.pdparams
    # SEVIR model evaluation
    python earthformer_sevir_train.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/earthformer/earthformer_sevir.pdparams
    ```

=== "Model Export Command"

    ``` sh
    # ICAR-ENSO model inference
    python earthformer_enso_train.py mode=export
    # SEVIR model inference
    python earthformer_sevir_train.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    # ICAR-ENSO model inference
    python earthformer_enso_train.py mode=infer
    # SEVIR model inference
    python earthformer_sevir_train.py mode=infer
    ```

| Model | Variable Name | C-Nino3.4-M | C-Nino3.4-WM | MSE(1E-4) |
| :-- | :-- | :-- | :-- | :-- |
| [ENSO Model](https://paddle-org.bj.bcebos.com/paddlescience/models/earthformer/earthformer_enso.pdparams) | sst | 0.74130 | 2.28990 | 2.5000 |

| Model | Variable Name | CSI-M | CSI-219 | CSI-181 | CSI-160 | CSI-133 | CSI-74 | CSI-16 | MSE(1E-4) |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| [SEVIR Model](https://paddle-org.bj.bcebos.com/paddlescience/models/earthformer/earthformer_sevir.pdparams) | vil | 0.4419 | 0.1791 | 0.2848 | 0.3232 | 0.4271 | 0.6860 | 0.7513 | 3.6957 |

## 1. Background Introduction

The Earth is a complex system. Variations in the Earth system, ranging from routine events like temperature fluctuations to extreme events like droughts, hail, and El Niño/Southern Oscillation (ENSO), affect our daily lives. Among all consequences, Earth system changes affect crop yields, flight delays, trigger floods and forest fires. Accurate and timely forecasting of these changes can help people take necessary precautions to avoid crises or make better use of natural resources such as wind and solar energy. Therefore, improving prediction models for Earth changes (such as weather and climate) has huge socio-economic impact.

Earthformer, a space-time transformer for Earth system forecasting. To better explore the design of space-time attention, the paper proposes Cuboid Attention, a generic building block for efficient space-time attention. The idea is to decompose the input tensor into non-overlapping cuboids and apply cuboid-level self-attention in parallel. Since we restrict the O(N<sup>2</sup>) self-attention to local cuboids, the overall complexity is greatly reduced. Different types of correlations can be captured by different cuboid decompositions. At the same time, the paper introduces a set of global vectors that attend to all local cuboids, thereby gathering the overall state of the system. By attending to global vectors, local cuboids can grasp the overall dynamics of the system and share information with each other.

## 2. Model Principle

This chapter only briefly introduces the model principle of EarthFormer. For detailed theoretical derivation, please read [Earthformer: Exploring Space-Time Transformers for Earth System Forecasting](https://arxiv.org/abs/2207.05833).

The Earthformer network model uses a hierarchical Transformer encoder-decoder based on Cuboid Attention. The idea is to decompose data into cuboids and apply cuboid-level self-attention in parallel. These cuboids are further connected to a collection of global vectors.

The overall structure of the model is shown in the figure:

<figure markdown>
  ![Earthformer-arch](https://paddle-org.bj.bcebos.com/paddlescience/docs/earthformer/earthformer_arch.png){ loading=lazy style="margin:0 auto;height:150%;width:150%"}
  <figcaption>EarthFormer Network Model</figcaption>
</figure>

The original EarthFormer code trained estimation models for sea surface temperature (sst) in the ICAR-ENSO dataset and vertically integrated liquid (vil) in the SEVIR dataset. Next, the training and inference processes of these two models will be introduced.

### 2.1 Training and Inference Process of ICAR-ENSO and SEVIR Models

The model pre-training phase trains the model based on randomly initialized network weights, as shown in the figure below, where $[x_{i}]_{i=1}^{T}$ represents input meteorological data of a spatiotemporal sequence of length $T$, $[y_{T+i}]_{i=1}^{K}$ represents predicted meteorological data for future $K$ steps, and $[y_{T+i_true}]_{i=1}^{K}$ represents true data for future $K$ steps, such as sea surface temperature data and vertically integrated liquid data. Finally, the mse loss function is calculated for the network model prediction output and the ground truth.

<figure markdown>
  ![earthformer-pretraining](https://paddle-org.bj.bcebos.com/paddlescience/docs/earthformer/earthformer-pretrain.png){ loading=lazy style="margin:0 auto;height:70%;width:70%"}
  <figcaption>earthformer model pretraining</figcaption>
</figure>

In the inference phase, given data of sequence length $T$, obtain prediction results of sequence length $K$.

<figure markdown>
  ![earthformer-pretraining](https://paddle-org.bj.bcebos.com/paddlescience/docs/earthformer/earthformer-infer.png){ loading=lazy style="margin:0 auto;height:60%;width:60%"}
  <figcaption>earthformer model inference</figcaption>
</figure>

## 3. Implementation of Sea Surface Temperature Model

Next, we will explain how to implement EarthFormer model training and inference based on PaddleScience code. For other details in this case, please refer to [API Documentation](../api/arch.md).

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

``` py linenums="35" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:35:56
--8<--
```

Among them, the "dataset" field defines the `Dataset` class name used as `ENSODataset`, the "sampler" field defines the `Sampler` class name used as `BatchSampler`, `batch_size` is set to 16, and `num_works` is 8.

The code for defining supervised constraints is as follows:

``` py linenums="58" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:58:64
--8<--
```

The first parameter of `SupervisedConstraint` is the data loading method, here `train_dataloader_cfg` defined above is used;

The second parameter is the definition of loss function, here the custom loss function `mse_loss` is used;

The third parameter is the name of the constraint condition, which is convenient for subsequent indexing. Here it is named `Sup`.

#### 3.2.2 Model Construction

In this case, the sea surface temperature model is implemented based on the CuboidTransformer network model, expressed in PaddleScience code as follows:

``` py linenums="97" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:97:99
--8<--
```

The parameters of the network model are set through the configuration file as follows:

``` yaml linenums="46" title="examples/earthformer/conf/earthformer_enso_pretrain.yaml"
--8<--
examples/earthformer/conf/earthformer_enso_pretrain.yaml:46:105
--8<--
```

Among them, `input_keys` and `output_keys` represent the names of input and output variables of the network model respectively.

#### 3.2.3 Learning Rate and Optimizer Construction

The learning rate method used in this case is `Cosine`, and the learning rate size is set to `2e-4`. The optimizer uses `AdamW`, and groups parameters to use different `weight_decay`, expressed in PaddleScience code as follows:

``` py linenums="101" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:101:126
--8<--
```

#### 3.2.4 Validator Construction

During the training process of this case, the training status of the current model will be evaluated using the validation set at certain training round intervals, and `SupervisedValidator` is needed to construct the validator. The code is as follows:

``` py linenums="68" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:68:95
--8<--
```

The `SupervisedValidator` validator is quite similar to `SupervisedConstraint`, the difference is that the validator needs to set evaluation metric `metric`, here custom evaluation metrics `MAE`, `MSE`, `RMSE`, `corr_nino3.4_epoch` and `corr_nino3.4_weighted_epoch` are used.

#### 3.2.5 Model Training and Evaluation

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver`, and then start training and evaluation.

``` py linenums="128" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:128:146
--8<--
```

### 3.3 Model Evaluation Visualization

#### 3.3.1 Evaluate Model on Test Set

The code for building the model is:

``` py linenums="179" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:179:181
--8<--
```

The code for building the validator is:

``` py linenums="150" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:150:177
--8<--
```

#### 3.3.2 Model Export

The code for building the model is:

``` py linenums="199" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:199:202
--8<--
```

Instantiate `ppsci.solver.Solver`:

``` py linenums="204" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:204:208
--8<--
```

Construct model input format and export static model:

``` py linenums="212" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:212:218
--8<--
```

In `InputSpec` function, the first sets model input size, the second parameter sets input data type, and the third sets input data `Key`.

#### 3.3.3 Model Inference

Create predictor:

``` py linenums="222" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:222:224
--8<--
```

Prepare prediction data:

``` py linenums="226" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:226:249
--8<--
```

Perform model prediction and save predicted values:

``` py linenums="253" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py:253:258
--8<--
```

## 4. Implementation of Vertically Integrated Liquid (vil) Model

### 4.1 Dataset Introduction

The dataset uses the SEVIR dataset processed by [EarthFormer](https://github.com/amazon-science/earth-forecasting-transformer/tree/main).

The Storm Event ImagRy (SEVIR) dataset was collected and provided by MIT Lincoln Laboratory and Amazon. SEVIR is an annotated, curated, and spatiotemporally aligned dataset containing over 10,000 weather events, each consisting of a sequence of 384 km x 384 km images spanning 4 hours. Images in SEVIR are sampled and aligned via five different data types: three channels of the GOES-16 Advanced Baseline Imager (C02, C09, C13), NEXRAD Vertically Integrated Liquid (vil), and GOES-16 Geostationary Lightning Mapper (GLM) flashes.

The structure of the SEVIR dataset consists of two parts: Catalog and Data File. The catalog is a CSV file containing rows describing event metadata. Data files are a set of HDF5 files containing events for specific sensor types. Data in these files is stored as 4D tensors with shape N x L x W x T, where N is the number of events in the file, LxW is image size, and T is the number of time steps in the image sequence.

<figure markdown>
  ![SEVIR](https://paddle-org.bj.bcebos.com/paddlescience/docs/earthformer/sevir.png){ loading=lazy style="margin:0 auto;height:100%;width:100%"}
  <figcaption>SEVIR Sensor Type Description</figcaption>
</figure>

EarthFormer uses NEXRAD Vertically Integrated Liquid (VIL) in SEVIR as a benchmark for precipitation forecasting, that is, predicting VIL for the future 60 minutes given a context of 65 minutes of VIL. Therefore, the resolution is 13x384x384&rarr;12x384x384.

### 4.2 Model Pretraining

#### 4.2.1 Constraint Construction

This case solves the problem based on data-driven methods, so it is necessary to use `SupervisedConstraint` built in PaddleScience to construct supervised constraints. Before defining constraints, you need to first specify various parameters used for data loading in supervised constraints.

Data loading code is as follows:

``` py linenums="27" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:27:59
--8<--
```

Among them, the "dataset" field defines the `Dataset` class name used as `ENSODataset`, the "sampler" field defines the `Sampler` class name used as `BatchSampler`, `batch_size` is set to 1, and `num_works` is 8.

The code for defining supervised constraints is as follows:

``` py linenums="61" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:61:67
--8<--
```

The first parameter of `SupervisedConstraint` is the data loading method, here `train_dataloader_cfg` defined above is used;

The second parameter is the definition of loss function, here the custom loss function `mse_loss` is used;

The third parameter is the name of the constraint condition, which is convenient for subsequent indexing. Here it is named `Sup`.

### 4.2.2 Model Construction

In this case, the vertically integrated liquid model is implemented based on the CuboidTransformer network model, expressed in PaddleScience code as follows:

``` py linenums="117" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:117:119
--8<--
```

Defining model parameters is set through configuration, as follows:

``` yaml linenums="58" title="examples/earthformer/conf/earthformer_sevir_pretrain.yaml"
--8<--
examples/earthformer/conf/earthformer_sevir_pretrain.yaml:58:117
--8<--
```

Among them, `input_keys` and `output_keys` represent the names of input and output variables of the network model respectively.

#### 4.2.3 Learning Rate and Optimizer Construction

The learning rate method used in this case is `Cosine`, and the learning rate size is set to `1e-3`. The optimizer uses `AdamW`, and groups parameters to use different `weight_decay`, expressed in PaddleScience code as follows:

``` py linenums="121" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:121:146
--8<--
```

#### 4.2.4 Validator Construction

During the training process of this case, the training status of the current model will be evaluated using the validation set at certain training round intervals, and `SupervisedValidator` is needed to construct the validator. The code is as follows:

``` py linenums="71" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:71:115
--8<--
```

The `SupervisedValidator` validator is quite similar to `SupervisedConstraint`, the difference is that the validator needs to set evaluation metric `metric`, here custom evaluation metrics `MAE`, `MSE`, `csi`, `pod`, `sucr` and `bias` are used, and the last four evaluation metrics use different thresholds `[16,74,133,160,181,219]` respectively.

#### 4.2.5 Model Training

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver`, and then start training.

``` py linenums="148" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:148:164
--8<--
```

#### 4.2.6 Model Evaluation

Since the validation strategy in `paddlescience` is currently divided into two categories, one is to directly concatenate model outputs for the validation dataset and then calculate evaluation metrics. The other is to calculate evaluation metrics for each batch_size, then concatenate, and finally average all results. This method assumes that there is no correlation between data. However, there is correlation between data in the `SEVIR` dataset, so the second method is not applicable; and because the `SEVIR` dataset is large, using the first method for validation requires large video memory, so the method used to validate the `SEVIR` dataset is as follows:

- 1. Calculate `hits`, `misses` and `fas` three data for a batch size
- 2. Save the cumulative sum of the three values of all `batch` for all data in the dataset.
- 3. Calculate `csi`, `pod`, `sucr` and `bias` four indicators for the cumulative sum of the three values.

``` py linenums="165" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:165:181
--8<--
```

### 4.3 Model Evaluation Visualization

#### 4.3.1 Evaluate Model on Test Set

The code for building the model is:

``` py linenums="231" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:231:233
--8<--
```

The code for building the validator is:

``` py linenums="185" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:185:229
--8<--
```

Model evaluation:

``` py linenums="246" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:246:262
--8<--
```

#### 4.3.2 Model Export

The code for building the model is:

``` py linenums="266" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:266:269
--8<--
```

Instantiate `ppsci.solver.Solver`:

``` py linenums="271" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:271:275
--8<--
```

Construct model input format and export static model:

``` py linenums="279" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:279:285
--8<--
```

In `InputSpec` function, the first sets model input size, the second parameter sets input data type, and the third sets input data `Key`.

#### 4.3.3 Model Inference

Create predictor:

``` py linenums="293" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:293:294
--8<--
```

Prepare prediction data and perform corresponding mode data preprocessing:

``` py linenums="295" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:295:314
--8<--
```

Perform model prediction and visualize:

``` py linenums="318" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py:318:330
--8<--
```

## 5. Complete Code

``` py linenums="1" title="examples/earthformer/earthformer_enso_train.py"
--8<--
examples/earthformer/earthformer_enso_train.py
--8<--
```

``` py linenums="1" title="examples/earthformer/earthformer_sevir_train.py"
--8<--
examples/earthformer/earthformer_sevir_train.py
--8<--
```

## 6. Result Display

The figure below shows the prediction results and ground truth results of the vertically integrated liquid model obtained at 60-minute intervals based on 65 minutes of input data.

<figure markdown>
  ![SEVIR-predict](https://paddle-org.bj.bcebos.com/paddlescience/docs/earthformer/sevir-predict.png){ loading=lazy style="margin:0 auto;height:100%;width:100%"}
  <figcaption>Prediction results ("prediction") vs ground truth ("target") of vil in SEVIR</figcaption>
</figure>

Description:

Hit:TP, Miss:FN, False Alarm:FP

First row: Input data;

Second row: Ground truth results;

Third row: Prediction results;

Fourth row: TP, FN, FP markers under threshold `74`

Fifth row: TP, FN, FP markers under all threshold cases
