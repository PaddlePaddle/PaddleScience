# FourCastNet

<a href="https://aistudio.baidu.com/aistudio/projectdetail/6213922?contributionType=1&sUid=455441&shared=1&ts=1684585396793" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

Before starting training and evaluation, please download the [dataset](https://app.globus.org/file-manager?origin_id=945b3c9e-0f8c-11ed-8daf-9f359c660fbd&origin_path=%2F~%2Fdata%2F).

=== "Model Training Command"

    ``` sh
    # Wind speed pretrain model
    python train_pretrain.py
    # Wind speed finetune model
    python train_finetune.py
    # Precipitation model training
    python train_precip.py
    ```

=== "Model Evaluation Command"

    ``` sh
    # Wind speed pretrain model evaluation
    python train_pretrain.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/fourcastnet/pretrain.pdparams
    # Wind speed finetune model evaluation
    python train_finetune.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/fourcastnet/finetune.pdparams
    # Precipitation model evaluation
    python train_precip.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/fourcastnet/precip.pdparams WIND_MODEL_PATH=https://paddle-org.bj.bcebos.com/paddlescience/models/fourcastnet/finetune.pdparams
    ```

=== "Model Export Command"

    ``` sh
    # Wind speed pretrain model export
    python train_pretrain.py mode=export
    # Wind speed finetune model export
    python train_finetune.py mode=export
    # Precipitation model export
    python train_precip.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    # Download wind speed prediction small sample data
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/FourcastNet/global_stds.npy -P ./datasets/era5/stat/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/FourcastNet/global_means.npy -P ./datasets/era5/stat/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/FourcastNet/2018-04-04_n6_precip.npy -P ./datasets/era5/test/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/FourcastNet/2018-04-04_n6.npy -P ./datasets/era5/test/
    # Download precipitation prediction small sample data
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/FourcastNet/2018-09-08_n32.npy -P ./datasets/era5/test/
    # Wind speed pretrain model inference
    python train_pretrain.py mode=infer
    # Wind speed finetune model inference
    python train_finetune.py mode=infer
    # Precipitation model inference
    python train_precip.py mode=infer
    ```


| Model | Variable Name | ACC/RMSE(6h) | ACC/RMSE(30h) | ACC/RMSE(60h) | ACC/RMSE(120h) | ACC/RMSE(192h) |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| [Wind Speed Model](https://paddle-org.bj.bcebos.com/paddlescience/models/fourcastnet/finetune.pdparams) | U10 | 0.991/0.567 | 0.963/1.130 | 0.891/1.930 | 0.645/3.438 | 0.371/4.915 |

| Model | Variable Name | ACC/RMSE(6h) | ACC/RMSE(12h) | ACC/RMSE(24h) | ACC/RMSE(36h) |
| :-- | :-- | :-- | :-- | :-- | :-- |
| [Precipitation Model](https://paddle-org.bj.bcebos.com/paddlescience/models/fourcastnet/precip.pdparams) | TP | 0.808/1.390 | 0.760/1.540 | 0.668/1.690 | 0.590/1.920 |

## 1. Background Introduction

In weather forecasting tasks, there are two methods based on physical information drive and data drive to achieve weather forecasting. Methods based on physical information drive often rely on physical equations to achieve weather forecasting by modeling the physical relationships between atmospheric variables. For example, in the IFS model, more than 150 atmospheric variables distributed at more than 50 vertical heights are used to achieve weather forecasting. Methods based on data drive do not rely on physical equations, but require a large amount of training data. Neural networks are generally regarded as a black box structure, and the network is trained to learn the functional relationship between input data and output data to achieve prediction of output data under given input conditions. FourCastNet is a weather forecasting algorithm based on data-driven methods, which uses Adaptive Fourier Neural Operators (AFNO) for training and prediction. The algorithm focuses on predicting two major meteorological variables: wind speed at 10 meters from the earth's surface and 6-hour total precipitation, in order to provide early warning of extreme weather, natural disasters, etc. Compared with the IFS model, it only uses a total of 20 atmospheric variables at 5 vertical heights, and has the characteristics of fewer atmospheric variable inputs and fast reasoning speed.

## 2. Model Principle

This chapter only briefly introduces the model principle of FourCastNet. For detailed theoretical derivation, please read [FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators](https://arxiv.org/abs/2202.11214).

The network model of FourCastNet uses the AFNO network, which was previously commonly used in image segmentation tasks. This network makes up for the shortcomings of the ViT network through FNO, using Fourier transform to complete the interaction of different token information, significantly reducing the amount of calculation of self-attention in ViT under high resolution. For related principles of [AFNO](https://openreview.net/pdf?id=EXHG-A3jlM), [FNO](https://arxiv.org/abs/2010.08895), [VIT](https://arxiv.org/pdf/2010.11929.pdf), please also read the corresponding papers.

The overall structure of the model is shown in the figure:

<figure markdown>
  ![fourcastnet-arch](https://paddle-org.bj.bcebos.com/paddlescience/docs/fourcastnet/fourcastnet_arch.png){ loading=lazy style="margin:0 auto"}
  <figcaption>FourCastNet Network Model</figcaption>
</figure>

The FourCastNet paper trained a wind speed model and a precipitation model. Next, the training and inference processes of these two models will be introduced.

### 2.1 Training and Inference Process of Wind Speed Model

The training process of the model is mainly divided into two steps: model pre-training and model fine-tuning.

The model pre-training stage is to train the model based on randomly initialized network weights, as shown in the figure below, where $X(k)$ represents the atmospheric data at time $k$, $X(k+1)$ represents the atmospheric data predicted by the model at time $k+1$, and $X_{true}(k+1)$ represents the real atmospheric data at time $k+1$. Finally, the output predicted by the network model and the ground truth calculate the L2 loss function.

<figure markdown>
  ![fourcastnet-pretraining](https://paddle-org.bj.bcebos.com/paddlescience/docs/fourcastnet/pretraining.png){ loading=lazy style="margin:0 auto;height:40%;width:40%"}
  <figcaption>Wind speed model pre-training</figcaption>
</figure>

The second stage of model training is model fine-tuning. The training in this stage is mainly to improve the accuracy of the model in medium- and long-term weather forecasting. Specifically, when the model inputs data at time $k$ and predicts data at time $k+1$, it is then re-used as input to predict data at time $k+2$. By continuously predicting data at two moments, the long-term prediction ability of the model is improved.

<figure markdown>
  ![fourcastnet-finetuning](https://paddle-org.bj.bcebos.com/paddlescience/docs/fourcastnet/finetuning.png){ loading=lazy style="margin:0 auto;height:40%;width:40%"}
  <figcaption>Wind speed model fine-tuning</figcaption>
</figure>

In the inference stage, given data at time $k$, prediction results at times $k+1$, $k+2$, $k+3$, etc. can be obtained through continuous iteration.

<figure markdown>
  ![fourcastnet-inference](https://paddle-org.bj.bcebos.com/paddlescience/docs/fourcastnet/wind_inference.png){ loading=lazy style="margin:0 auto;height:40%;width:40%"}
  <figcaption>Wind speed model inference</figcaption>
</figure>

### 2.2 Training and Inference Process of Precipitation Model

The training of the precipitation model depends on the wind speed model. As shown in the figure below, using atmospheric variable data $X(k)$ at time $k$ as input to the trained wind speed model, the predicted atmospheric variable data $X(k+1)$ at time $k+1$ is obtained. The precipitation model takes $X(k+1)$ as input and outputs the precipitation prediction result $p(k+1)$ at time $k+1$. During model training, the L2 loss function is calculated between $p(k+1)$ and the ground truth data $p_{true}(k+1)$ to constrain network training.

<figure markdown>
  ![precip-training](https://paddle-org.bj.bcebos.com/paddlescience/docs/fourcastnet/precip_training.png){ loading=lazy style="margin:0 auto;height:40%;width:40%"}
  <figcaption>Precipitation model training</figcaption>
</figure>

It should be noted that during the training process of the precipitation model, the parameters of the wind speed model are in a frozen state and do not participate in the optimizer parameter update process.

In the inference stage, given data at time $k$, atmospheric variable prediction results at times $k+1$, $k+2$, $k+3$, etc. can be obtained through continuous iteration using the wind speed model, and used as input to the precipitation model to predict precipitation at corresponding times.

<figure markdown>
  ![precip-inference](https://paddle-org.bj.bcebos.com/paddlescience/docs/fourcastnet/precip_inference.png){ loading=lazy style="margin:0 auto;height:40%;width:40%"}
  <figcaption>Precipitation model inference</figcaption>
</figure>

## 3. Wind Speed Model Implementation

Next, we will explain how to implement the training and inference of the FourCastNet wind speed model based on PaddleScience code. For other details in this case, please refer to [API Documentation](../api/arch.md).

???+ Info

    Since complete reproduction requires 5+TB of storage space and 64-card training resources, if it is only for learning the algorithm principle of FourCastNet, it is recommended to train on a small part of the training dataset to reduce learning costs.

### 3.1 Dataset Introduction

The dataset uses the ERA5 dataset processed in [FourCastNet](https://github.com/NVlabs/FourCastNet). The resolution of this dataset is 0.25 degrees, and the data size of each variable is $720 \times 1440$, where the actual distance represented by a single data point is about 30km. FourCastNet uses data from 1979-2018, and divides it into training set, validation set, and test set according to the year. The division results are as follows:

|Dataset |Year        |
|:----:|:---------:|
|Training set |1979-2015  |
|Validation set |2016-2017  |
|Test set |2018       |

The dataset can be downloaded from [here](https://app.globus.org/file-manager?origin_id=945b3c9e-0f8c-11ed-8daf-9f359c660fbd&origin_path=%2F~%2Fdata%2F).

The model training uses 20 atmospheric variables distributed on 5 pressure layers, as shown in the table below,

<figure markdown>
  ![fourcastnet-vars](https://paddle-org.bj.bcebos.com/paddlescience/docs/fourcastnet/era5-vars.png){ loading=lazy style="margin:0 auto;height:60%;width:60%"}
  <figcaption>20 atmospheric variables</figcaption>
</figure>

Among them, $T$, $U$, $V$, $Z$, $RH$ represent temperature, zonal wind speed, meridional wind speed, geopotential and relative humidity at specified vertical heights respectively; $U_{10}$, $V_{10}$, $T_{2m}$ represent zonal wind speed at 10 meters from the ground, meridional wind speed and temperature at 2 meters from the ground. $sp$ represents surface pressure, and $mslp$ represents mean sea level pressure. $TCWV$ represents total column water vapor.

Data for 24 hours a day is sampled at 6-hour intervals to obtain data of 20 atmospheric variables globally at 0.00h/6.00h/12.00h/18.00h moments, and such data is used for model training and inference. That is, input data of 20 atmospheric variables at 0.00h moment, and the model outputs predicted data of 20 atmospheric variables at 6.00h moment.

### 3.2 Model Pre-training

First, the various parameter variables defined in the code are displayed. The specific meaning of each parameter will be explained when used below.

``` yaml linenums="28" title="examples/fourcastnet/conf/fourcastnet_pretrain.yaml"
--8<--
examples/fourcastnet/conf/fourcastnet_pretrain.yaml:28:46
--8<--
```

#### 3.2.1 Constraint Construction

This case solves the problem based on data-driven methods, so it is necessary to use `SupervisedConstraint` built in PaddleScience to construct supervised constraints. Before defining constraints, you need to first specify various parameters used for data loading in supervised constraints. First introduce the data preprocessing part, the code is as follows:

``` py linenums="46" title="examples/fourcastnet/train_pretrain.py"
--8<--
examples/fourcastnet/train_pretrain.py:46:60
--8<--
```

The data preprocessing part contains a total of 3 preprocessing methods, namely:

1. `SqueezeData`: Compress the dimensions of training data. If the dimension of input data is 4, compress data of 0th dimension and 1st dimension together, and finally transform the dimension of input data to 3.
2. `CropData`: Crop data at specified position from training data. Because the original data size in ERA5 dataset is $721 \times 1440$, this case crops the training data to $720 \times 1440$ according to the original paper setting.
3. `Normalize`: Normalize data according to mean and variance on the training dataset.

Since complete reproduction of FourCastNet requires 5TB+ of storage space and 64-card GPU resources, and requires a lot of storage resources, there are two training methods (experiments prove that the loss function convergence curves of the two training methods are basically consistent. When storage resources are relatively limited, method b can be used).

Method a: When storage resources are sufficient, data can be undivided, each node has a complete 5TB+ training data, and then directly start the training program for training. At this time, the data on each node is randomly selected from the complete training data. The loading of training data in this method is carried out using global shuffle, as shown in the figure below.

<figure markdown>
  ![fourcastnet-vars](https://paddle-org.bj.bcebos.com/paddlescience/docs/fourcastnet/fourcastnet_global_shuffle.png){ loading=lazy style="margin:0 auto;height:60%;width:60%"}
  <figcaption>Global shuffle</figcaption>
</figure>

In this method, the code for data loading is as follows:

``` py linenums="64" title="examples/fourcastnet/train_pretrain.py"
--8<--
examples/fourcastnet/train_pretrain.py:64:80
--8<--
```

Among them, the "dataset" field defines the used `Dataset` class name as `ERA5Dataset`, the "sampler" field defines the used `Sampler` class name as `BatchSampler`, setting `batch_size` to 1 and `num_works` to 8.

Method b: When storage resources are limited, the dataset needs to be evenly split to each node. This case provides a program for randomly sampling data, which can execute `ppsci/fourcastnet/sample_data.py` and can be modified as needed. This case uses method a by default, so when using method b for model training, you need to manually set `USE_SAMPLED_DATA` to `True`. The loading of training data in this method is carried out using local shuffle, as shown in the figure below. First, the training data is evenly split to 8 nodes. During training, the data of each node is randomly selected from the split data. In this case, each node needs about 1.2TB of storage space. Compared with method a, method b greatly reduces the dependence on storage space.

<figure markdown>
  ![fourcastnet-vars](https://paddle-org.bj.bcebos.com/paddlescience/docs/fourcastnet/fourcastnet_local_shuffle.png){ loading=lazy style="margin:0 auto;height:60%;width:60%"}
  <figcaption>Local shuffle</figcaption>
</figure>

In this method, the code for data loading is as follows:

``` py linenums="82" title="examples/fourcastnet/train_pretrain.py"
--8<--
examples/fourcastnet/train_pretrain.py:82:99
--8<--
```

Among them, the "dataset" field defines the used `Dataset` class name as `ERA5SampledDataset`, the "sampler" field defines the used `Sampler` class name as `DistributedBatchSampler`, setting `batch_size` to 1 and `num_works` to 8.

When complete reproduction of FourCastNet is not required, simply use the default setting of this case (method a).

The code for defining supervised constraints is as follows:

``` py linenums="100" title="examples/fourcastnet/train_pretrain.py"
--8<--
examples/fourcastnet/train_pretrain.py:100:106
--8<--
```

The first parameter of `SupervisedConstraint` is the data loading method, here `train_dataloader_cfg` defined above is used;

The second parameter is the definition of loss function, here `L2RelLoss` is used;

The third parameter is the name of the constraint condition, which is convenient for subsequent indexing. Here it is named "Sup".

#### 3.2.2 Model Construction

In this case, the wind speed model is based on the AFNONet network model, expressed in PaddleScience code as follows:

``` py linenums="153" title="examples/fourcastnet/train_pretrain.py"
--8<--
examples/fourcastnet/train_pretrain.py:153:154
--8<--
```

The parameters of the network model are set through the configuration file as follows:

``` yaml linenums="48" title="examples/fourcastnet/conf/fourcastnet_pretrain.yaml"
--8<--
examples/fourcastnet/conf/fourcastnet_pretrain.yaml:48:52
--8<--
```

Among them, `input_keys` and `output_keys` represent the names of input and output variables of the network model respectively.

#### 3.2.3 Learning Rate and Optimizer Construction

The learning rate method used in this case is `Cosine`, and the learning rate size is set to 5e-4. The optimizer uses `Adam`, expressed in PaddleScience code as follows:

``` py linenums="156" title="examples/fourcastnet/train_pretrain.py"
--8<--
examples/fourcastnet/train_pretrain.py:156:161
--8<--
```

#### 3.2.4 Validator Construction

In this case, the validation set is used to evaluate the training status of the current model at certain training epoch intervals during the training process, and `SupervisedValidator` is needed to construct the validator. The code is as follows:

``` py linenums="111" title="examples/fourcastnet/train_pretrain.py"
--8<--
examples/fourcastnet/train_pretrain.py:111:151
--8<--
```

The `SupervisedValidator` validator is similar to `SupervisedConstraint`, the difference is that the validator needs to set the evaluation metric `metric`, here 3 evaluation metrics are used, namely `MAE`, `LatitudeWeightedRMSE` and `LatitudeWeightedACC`.

#### 3.2.5 Model Training and Evaluation

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training and evaluation.

``` py linenums="163" title="examples/fourcastnet/train_pretrain.py"
--8<--
examples/fourcastnet/train_pretrain.py:163:181
--8<--
```

### 3.3 Model Fine-tuning

The previous section introduced how to pre-train the wind speed model. This section will introduce how to use the pre-trained model for fine-tuning. Since the steps for pre-training and fine-tuning the wind speed model are basically similar, this section will not introduce the repeated parts of the two, but only introduce the unique parts of model fine-tuning. First, the various parameter variables defined in the code are displayed as follows. The specific meaning of each parameter will be explained when used below.

``` yaml linenums="28" title="examples/fourcastnet/conf/fourcastnet_finetune.yaml"
--8<--
examples/fourcastnet/conf/fourcastnet_finetune.yaml:28:48
--8<--
```

The fine-tuning model program adds a `num_timestamps` parameter to control the number of time steps iterated during model fine-tuning training. This parameter will first be used in the data loading setting to set the time step size of the ground truth generated by the dataset. The code is as follows:

``` py linenums="84" title="examples/fourcastnet/train_finetune.py"
--8<--
examples/fourcastnet/train_finetune.py:84:102
--8<--
```

The `num_timestamps` parameter is set through the configuration file as follows:

``` yaml linenums="66" title="examples/fourcastnet/conf/fourcastnet_finetune.yaml"
--8<--
examples/fourcastnet/conf/fourcastnet_finetune.yaml:66:66
--8<--
```

In addition, unlike pre-training, fine-tuning model construction also requires setting the `num_timestamps` parameter to control the time step size of the prediction results output by the model. The code is as follows:

``` py linenums="160" title="examples/fourcastnet/train_finetune.py"
--8<--
examples/fourcastnet/train_finetune.py:160:164
--8<--
```

The code for evaluating model performance on the test set and visualization code have been added to the program for training fine-tuning models. Next, these two parts will be introduced in detail.

#### 3.3.1 Evaluating Model on Test Set

According to the settings in the paper, when evaluating the model on the test set, `num_timestamps` is set to 32 through the configuration file, and the interval between two adjacent test samples is 8.

``` yaml linenums="70" title="examples/fourcastnet/conf/fourcastnet_finetune.yaml"
--8<--
examples/fourcastnet/conf/fourcastnet_finetune.yaml:70:72
--8<--
```

The code for constructing the model is:

``` py linenums="221" title="examples/fourcastnet/train_finetune.py"
--8<--
examples/fourcastnet/train_finetune.py:221:226
--8<--
```

The code for constructing the validator is:

``` py linenums="228" title="examples/fourcastnet/train_finetune.py"
--8<--
examples/fourcastnet/train_finetune.py:228:273
--8<--
```

#### 3.3.2 Visualizer Construction

The wind speed model uses autoregressive method for inference, and the input data for model inference needs to be set first. The code is as follows:

``` py linenums="275" title="examples/fourcastnet/train_finetune.py"
--8<--
examples/fourcastnet/train_finetune.py:275:285
--8<--
```

``` py linenums="30" title="examples/fourcastnet/train_finetune.py"
--8<--
examples/fourcastnet/train_finetune.py:30:55
--8<--
```

In the above code, the corresponding data is read for model input based on the set time parameter `DATE_STRINGS`. In addition, the `get_vis_datas` function also reads the ground truth data at the corresponding time. These data will also be visualized for comparison with the model prediction results.

Since the model predicts zonal and meridional wind speeds separately, it is necessary to synthesize wind speeds in these two directions into real wind speed. The code is as follows:

``` py linenums="287" title="examples/fourcastnet/train_finetune.py"
--8<--
examples/fourcastnet/train_finetune.py:287:303
--8<--
```

Finally, the code for constructing the visualizer is as follows:

``` py linenums="304" title="examples/fourcastnet/train_finetune.py"
--8<--
examples/fourcastnet/train_finetune.py:304:320
--8<--
```

The constructed model, validator, and visualizer above will be passed to `ppsci.solver.Solver` for evaluating performance on the test set and visualization.

``` py linenums="322" title="examples/fourcastnet/train_finetune.py"
--8<--
examples/fourcastnet/train_finetune.py:322:333
--8<--
```

## 4. Precipitation Model Implementation

First, the various parameter variables defined in the code are displayed. The specific meaning of each parameter will be explained when used below.

``` yaml linenums="28" title="examples/fourcastnet/conf/fourcastnet_precip.yaml"
--8<--
examples/fourcastnet/conf/fourcastnet_precip.yaml:28:56
--8<--
```

### 4.1 Constraint Construction

This case solves the problem based on data-driven methods, so it is necessary to use `SupervisedConstraint` built in PaddleScience to construct supervised constraints. Before defining constraints, you need to first specify various parameters used for data loading in supervised constraints. First introduce the data preprocessing part, the code is as follows:

``` py linenums="66" title="examples/fourcastnet/train_precip.py"
--8<--
examples/fourcastnet/train_precip.py:66:85
--8<--
```

The data preprocessing part contains a total of 4 preprocessing methods, namely:

1. `SqueezeData`: Compress the dimensions of training data. If the dimension of input data is 4, compress data of 0th dimension and 1st dimension together, and finally transform the dimension of input data to 3.
2. `CropData`: Crop data at specified position from training data. Because the original data size in ERA5 dataset is $721 \times 1440$, this case crops the training data size to $720 \times 1440$ according to the original paper setting.
3. `Normalize`: Normalize data according to mean and variance on the training dataset. Here, the `apply_keys` field sets this preprocessing method to be applied only to input data.
4. `Log1p`: Map data to logarithmic space. Here, the `apply_keys` field sets this preprocessing method to be applied only to ground truth data.

The code for data loading is as follows:

``` py linenums="87" title="examples/fourcastnet/train_precip.py"
--8<--
examples/fourcastnet/train_precip.py:87:105
--8<--
```

Among them, the "dataset" field defines the used `Dataset` class name as `ERA5Dataset`, the "sampler" field defines the used `Sampler` class name as `BatchSampler`, setting `batch_size` to 1 and `num_works` to 8.

The code for defining supervised constraints is as follows:

``` py linenums="106" title="examples/fourcastnet/train_precip.py"
--8<--
examples/fourcastnet/train_precip.py:106:112
--8<--
```

The first parameter of `SupervisedConstraint` is the data loading method, here `train_dataloader_cfg` defined above is used;

The second parameter is the definition of loss function, here `L2RelLoss` is used;

The third parameter is the name of the constraint condition, which is convenient for subsequent indexing. Here it is named "Sup".

### 4.2 Model Construction

In this case, it is necessary to first define the network structure of the wind speed model and load the trained parameters, and then define the precipitation model, expressed in PaddleScience code as follows:

``` py linenums="157" title="examples/fourcastnet/train_precip.py"
--8<--
examples/fourcastnet/train_precip.py:157:162
--8<--
```

The parameters for defining the model are set through configuration as follows:

``` yaml linenums="58" title="examples/fourcastnet/conf/fourcastnet_precip.yaml"
--8<--
examples/fourcastnet/conf/fourcastnet_precip.yaml:58:65
--8<--
```

Among them, `input_keys` and `output_keys` represent the names of input and output variables of the network model respectively.

### 4.3 Learning Rate and Optimizer Construction

The learning rate method used in this case is `Cosine`, and the learning rate size is set to 2.5e-4. The optimizer uses `Adam`, expressed in PaddleScience code as follows:

``` py linenums="164" title="examples/fourcastnet/train_precip.py"
--8<--
examples/fourcastnet/train_precip.py:164:168
--8<--
```

### 4.4 Validator Construction

In this case, the validation set is used to evaluate the training status of the current model at certain training epoch intervals during the training process, and `SupervisedValidator` is needed to construct the validator. The code is as follows:

``` py linenums="117" title="examples/fourcastnet/train_precip.py"
--8<--
examples/fourcastnet/train_precip.py:117:155
--8<--
```

The `SupervisedValidator` validator is similar to `SupervisedConstraint`, the difference is that the validator needs to set the evaluation metric `metric`, here 3 evaluation metrics are used, namely `MAE`, `LatitudeWeightedRMSE` and `LatitudeWeightedACC`.

### 4.5 Model Training and Evaluation

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training and evaluation.

``` py linenums="170" title="examples/fourcastnet/train_precip.py"
--8<--
examples/fourcastnet/train_precip.py:170:187
--8<--
```

### 4.6 Evaluating Model on Test Set

According to the settings in the paper, when evaluating the model on the test set, `num_timestamps` is set to 6, and the interval between two adjacent test samples is 8.

The code for constructing the model is:

``` py linenums="199" title="examples/fourcastnet/train_precip.py"
--8<--
examples/fourcastnet/train_precip.py:199:210
--8<--
```

The code for constructing the validator is:

``` py linenums="233" title="examples/fourcastnet/train_precip.py"
--8<--
examples/fourcastnet/train_precip.py:233:271
--8<--
```

### 4.7 Visualizer Construction

The precipitation model uses autoregressive method for inference, and the input data for model inference needs to be set first. The code is as follows:

``` py linenums="273" title="examples/fourcastnet/train_precip.py"
--8<--
examples/fourcastnet/train_precip.py:273:284
--8<--
```

``` py linenums="30" title="examples/fourcastnet/train_precip.py"
--8<--
examples/fourcastnet/train_precip.py:30:57
--8<--
```

In the above code, the corresponding data is read for model input based on the set time parameter `DATE_STRINGS`. In addition, the `get_vis_datas` function also reads the ground truth data at the corresponding time. These data will also be visualized for comparison with the model prediction results.

Since the model performs logarithmic processing on precipitation, it is necessary to remap the model results back to linear space. The code is as follows:

``` py linenums="286" title="examples/fourcastnet/train_precip.py"
--8<--
examples/fourcastnet/train_precip.py:286:299
--8<--
```

Finally, the code for constructing the visualizer is as follows:

``` py linenums="300" title="examples/fourcastnet/train_precip.py"
--8<--
examples/fourcastnet/train_precip.py:300:317
--8<--
```

The constructed model, validator, and visualizer above will be passed to `ppsci.solver.Solver` for evaluating performance on the test set and visualization.

``` py linenums="319" title="examples/fourcastnet/train_precip.py"
--8<--
examples/fourcastnet/train_precip.py:319:330
--8<--
```

## 5. Complete Code

``` py linenums="1" title="examples/fourcastnet/train_pretrain.py"
--8<--
examples/fourcastnet/train_pretrain.py
--8<--
```

``` py linenums="1" title="examples/fourcastnet/train_finetune.py"
--8<--
examples/fourcastnet/train_finetune.py
--8<--
```

``` py linenums="1" title="examples/fourcastnet/train_precip.py"
--8<--
examples/fourcastnet/train_precip.py
--8<--
```

## 6. Result Display

The figure below shows the prediction results and ground truth results of the wind speed model at 6-hour intervals.

<figure markdown>
  ![result_wind](https://paddle-org.bj.bcebos.com/paddlescience/docs/fourcastnet/result_wind.gif){ loading=lazy style="margin:0 auto;"}
  <figcaption>Wind speed model prediction result ("output") vs ground truth result ("target")</figcaption>
</figure>

The figure below shows the prediction results and ground truth results of the precipitation model at 6-hour intervals.

<figure markdown>
  ![result_precip](https://paddle-org.bj.bcebos.com/paddlescience/docs/fourcastnet/result_precip.gif){ loading=lazy style="margin:0 auto;"}
  <figcaption>Precipitation model prediction result ("output") vs ground truth result ("target")</figcaption>
</figure>
