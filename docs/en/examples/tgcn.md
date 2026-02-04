# TGCN

=== "Model Training Command"

    ``` sh
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/tgcn/tgcn_data.zip
    unzip tgcn_data.zip
    python run.py data_name=PEMSD8
    # python run.py data_name=PEMSD4
    ```

=== "Model Evaluation Command"

    ``` sh
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/tgcn/tgcn_data.zip
    unzip tgcn_data.zip
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/tgcn/PEMSD8_pretrained_model.pdparams
    python run.py data_name=PEMSD8 mode=eval EVAL.pretrained_model_path=PEMSD8_pretrained_model.pdparams
    # wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/tgcn/PEMSD4_pretrained_model.pdparams
    # python run.py data_name=PEMSD4 mode=eval EVAL.pretrained_model_path=PEMSD4_pretrained_model.pdparams
    ```

| Pretrained Model                                                   | Metric                    |
| ------------------------------------------------------------ | ----------------------- |
| [PEMSD4_pretrained_model.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/tgcn/PEMSD4_pretrained_model.pdparams) | MAE: 21.48; RMSE: 34.06 |
| [PEMSD8_pretrained_model.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/tgcn/PEMSD8_pretrained_model.pdparams) | MAE: 15.57; RMSE: 24.52 |


## 1. Background Introduction

Traffic prediction aims to predict future traffic time series conditions (such as traffic flow or traffic speed) by analyzing historical observation data (such as sensor records on traffic networks). As an important component of the Intelligent Transportation System (ITS), the traffic prediction task is the core foundation for realizing smart cities, including active dynamic traffic control and intelligent route guidance, which helps reduce road safety hazards and improve the operational efficiency of urban transportation systems.

TGCN, a Temporal Graph Convolutional Network for traffic flow prediction. Specifically, by modeling the traffic network as graph structure data, the Graph Convolutional Network (GCN) module is used to extract spatial features; by modeling the traffic signal as time series information, the Temporal Convolutional Network (TCN) module is used to capture temporal features. TGCN finally completes the traffic flow prediction task by iteratively executing two modules.

## 2. Model Principle

This chapter briefly introduces the model principle of TGCN.

### 2.1 Graph Convolutional Network Module

This module uses a two-layer message passing network to extract spatial features and update node features:

``` py linenums="12" title="ppsci/arch/tgcn.py"
--8<--
ppsci/arch/tgcn.py:12:35
--8<--
```

### 2.2 Temporal Convolutional Network Module

This module uses a three-layer one-dimensional convolutional network to extract temporal features and update node features:

``` py linenums="38" title="ppsci/arch/tgcn.py"
--8<--
ppsci/arch/tgcn.py:38:71
--8<--
```

### 2.3 TGCN Model Structure

The TGCN model first uses a feature embedding layer to encode the input signal (i.e., traffic flow data of traffic nodes in the past period):

``` py linenums="140" title="ppsci/arch/tgcn.py"
--8<--
ppsci/arch/tgcn.py:140:145
--8<--
```

``` py linenums="173" title="ppsci/arch/tgcn.py"
--8<--
ppsci/arch/tgcn.py:173:176
--8<--
```

Then the model alternately stacks the aforementioned TCN module and GCN module to update node features:

``` py linenums="147" title="ppsci/arch/tgcn.py"
--8<--
ppsci/arch/tgcn.py:147:157
--8<--
```

``` py linenums="178" title="ppsci/arch/tgcn.py"
--8<--
ppsci/arch/tgcn.py:178:192
--8<--
```

Finally, the model concatenates the initial node features with the inputs of the two GCN modules, and uses a two-layer MLP to obtain the target output (i.e., traffic flow prediction of traffic nodes in the future period):

``` py linenums="159" title="ppsci/arch/tgcn.py"
--8<--
ppsci/arch/tgcn.py:159:170
--8<--
```

``` py linenums="194" title="ppsci/arch/tgcn.py"
--8<--
ppsci/arch/tgcn.py:194:198
--8<--
```

## 3. Model Training

### 3.1 Dataset Introduction

The case uses preprocessed PEMSD4 and PEMSD8 datasets. PEMSD4 is traffic data from the San Francisco Bay Area, selecting traffic data recorded by 307 sensors on 29 roads from January to February 2018. PEMSD8 is traffic data collected by 170 detectors on 8 roads in San Bernardino from July to August 2016.

Both datasets are saved as N x T x 1 matrices, recording traffic data of corresponding traffic nodes and times, where N is the number of traffic nodes and T is the length of the time series. The two datasets are divided into training set, validation set, and test set according to 7:2:1 respectively. The mean and standard deviation of traffic data are pre-calculated in the case for subsequent normalization operations.

### 3.2 Model Training

#### 3.2.1 Model Construction

This case is implemented based on the TGCN model, expressed in PaddleScience code as follows:

``` py linenums="67" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py:67:82
--8<--
```

#### 3.2.2 Constraint Construction

This case solves the problem based on data-driven methods, so it is necessary to use `SupervisedConstraint` built in PaddleScience to construct supervised constraints. Before defining constraints, you need to first specify various parameters used for data loading in constraints.

Training set data loading code is as follows:

``` py linenums="10" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py:10:29
--8<--
```

The code for defining supervised constraints is as follows:

``` py linenums="31" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py:31:35
--8<--
```

The first parameter of `SupervisedConstraint` is the data loading method, here `train_dataloader_cfg` defined above is used;

The second parameter is the definition of loss function, here the custom loss function `L1_loss` is used;

The third parameter is the name of the constraint condition, which is convenient for subsequent indexing. Here it is named `train`.

#### 3.2.3 Validator Construction

During the training process of this case, the training status of the current model will be evaluated using the validation set at certain training round intervals, and `SupervisedValidator` is needed to construct the validator.

Validation set data loading code is as follows:

``` py linenums="37" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py:37:54
--8<--
```

The code for defining supervised validator is as follows:

``` py linenums="56" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py:56:63
--8<--
```

The `SupervisedValidator` validator is similar to `SupervisedConstraint` constraint, the difference is that the validator needs to set evaluation metric `metric`, here the evaluation metrics used are `MAE` and `RMSE`.

#### 3.2.4 Learning Rate and Optimizer Construction

The learning rate size used in this case is set to `1e-2`. The optimizer uses `Adam`, expressed in PaddleScience code as follows:

``` py linenums="83" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py:83:84
--8<--
```

#### 3.2.5 Model Training

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver`, and then start training.

``` py linenums="88" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py:88:104
--8<--
```

#### 3.2.6 Model Export

By setting the `eval_during_train` parameter in `ppsci.solver.Solver`, the model parameters with the best effect on the validation set can be automatically saved.

``` py linenums="97" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py:97:97
--8<--
```

### 3.3 Evaluation Model

#### 3.3.1 Validator Construction

Test set data loading code is as follows:

``` py linenums="108" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py:108:125
--8<--
```

The code for defining supervised validator is as follows:

``` py linenums="127" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py:127:134
--8<--
```

Similar to `SupervisedValidator` for validation set, the evaluation metrics used here are `MAE` and `RMSE`.

#### 3.3.2 Load Model and Evaluate

Set the loading path of pre-trained model parameters and load the model.

``` py linenums="138" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py:138:153
--8<--
```

Instantiate `ppsci.solver.Solver`, and then start evaluation.

``` py linenums="155" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py:155:166
--8<--
```

## 4. Complete Code

Dataset interface:

``` py linenums="1" title="ppsci/data/dataset/pems_dataset.py"
--8<--
ppsci/data/dataset/pems_dataset.py
--8<--
```

Model structure:

``` py linenums="1" title="ppsci/arch/tgcn.py"
--8<--
ppsci/arch/tgcn.py
--8<--
```

Model training:

``` py linenums="1" title="examples/tgcn/run.py"
--8<--
examples/tgcn/run.py
--8<--
```

Configuration file:

``` py linenums="1" title="examples/tgcn/conf/run.yaml"
--8<--
examples/tgcn/conf/run.yaml
--8<--
```

## 5. Result Display

The table below shows the evaluation results of TGCN on PEMSD4 and PEMSD8 datasets.

| Dataset | MAE   | RMSE  |
| :----- | :---- | :---- |
| PEMSD4 | 21.48 | 34.06 |
| PEMSD8 | 15.57 | 24.52 |
