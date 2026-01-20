# Rossler System

<a href="https://aistudio.baidu.com/aistudio/projectdetail/6209280?sUid=455441&shared=1&ts=1684495132419" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

=== "Model Training Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_training.hdf5 -P ./datasets/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_valid.hdf5 -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_training.hdf5 --create-dirs -o ./datasets/rossler_training.hdf5
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_valid.hdf5 --create-dirs -o ./datasets/rossler_valid.hdf5
    python train_enn.py
    python train_transformer.py
    ```

=== "Model Evaluation Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_training.hdf5 -P ./datasets/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_valid.hdf5 -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_training.hdf5 --create-dirs -o ./datasets/rossler_training.hdf5
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_valid.hdf5 --create-dirs -o ./datasets/rossler_valid.hdf5
    python train_enn.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/rossler/rossler_pretrained.pdparams
    python train_transformer.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/rossler/rossler_transformer_pretrained.pdparams EMBEDDING_MODEL_PATH=https://paddle-org.bj.bcebos.com/paddlescience/models/rossler/rossler_pretrained.pdparams
    ```

=== "Model Export Command"

    ``` sh
    python train_transformer.py mode=export EMBEDDING_MODEL_PATH=https://paddle-org.bj.bcebos.com/paddlescience/models/rossler/rossler_pretrained.pdparams
    ```

=== "Model Inference Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_training.hdf5 -P ./datasets/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_valid.hdf5 -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_training.hdf5 --create-dirs -o ./datasets/rossler_training.hdf5
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_valid.hdf5 --create-dirs -o ./datasets/rossler_valid.hdf5
    python train_transformer.py mode=infer
    ```

| Model | MSE |
| :-- | :-- |
| [rossler_transformer_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/rossler/rossler_transformer_pretrained.pdparams) | 0.022 |

## 1. Background Introduction

The Rossler System, first proposed by German scientist Rossler, is also a common chaotic system. This system plays an important role in the study of chaos theory, providing a mathematical description and understanding method for chaotic phenomena. At the same time, because the system is extremely sensitive to numerical disturbances, it is also a good benchmark for evaluating the accuracy of machine learning (deep learning) models.

## 2. Problem Definition

The state equations of the Rossler system:

$$
\begin{cases}
  \dfrac{\partial x}{\partial t} = -\omega y - z, & \\
  \dfrac{\partial y}{\partial t} = \omega x + \alpha y, & \\
  \dfrac{\partial z}{\partial t} = \beta + z(x - \gamma)
\end{cases}
$$

When the parameters take the following values, the system exhibits classic chaotic characteristics:

$$\omega = 1.0, \alpha = 0.165, \beta = 0.2, \gamma = 10$$

In this case, given the coordinates of the point at the initial time, predict the movement trajectory of the point in the future period.

## 3. Problem Solving

Next, we will explain how to solve this problem using deep learning methods based on PaddleScience code. This case is solved based on the method in the paper [Transformers for Modeling Physical Systems](https://arxiv.org/abs/2010.03957). For the theoretical part of this method, please refer to [this document](lorenz.md#31) or [original paper](https://arxiv.org/abs/2010.03957). Next, the dataset used will be introduced first, and then the supervised constraint construction and model construction of the two training steps of this method (Embedding model training, Transformer model training) will be explained. For other details, please refer to [API Documentation](../api/arch.md).

### 3.1 Dataset Introduction

The dataset uses data provided in [Transformer-Physx](https://github.com/zabaras/transformer-physx). This dataset is obtained using the traditional Runge-Kutta numerical solution method. The dataset is divided as follows:

|Dataset |Number of Time Series|Number of Time Steps|Download Address|
|:----:|:---------:|:--------:|:--------:|
|Training Set |256        |1025      |[rossler_training.hdf5](https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_training.hdf5)|
|Validation Set |32         |1025      |[rossler_valid.hdf5](https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_valid.hdf5)|

Dataset official website: <https://zenodo.org/record/5148524#.ZDe77-xByrc>

### 3.2 Embedding Model

First, show the various parameter variables defined in the code. The specific meaning of each parameter will be explained when used below.

``` yaml linenums="22" title="examples/rossler/conf/enn.yaml"
--8<--
examples/rossler/conf/enn.yaml:22:34
--8<--
```

#### 3.2.1 Constraint Construction

This case solves the problem based on data-driven methods, so it is necessary to use `SupervisedConstraint` built in PaddleScience to construct supervised constraints. Before defining constraints, you need to first specify various parameters used for data loading in supervised constraints. The code is as follows:

``` py linenums="55" title="examples/rossler/train_enn.py"
--8<--
examples/rossler/train_enn.py:55:74
--8<--
```

Among them, the "dataset" field defines the used `Dataset` class name as `RosslerDataset`, and also specifies the values of parameters when initializing this class:

1. `file_path`: Represents the file path of the training dataset, specified as the value of variable `train_file_path`;
2. `input_keys`: Represents the variable name of model input data, here fill in variable `input_keys`;
3. `label_keys`: Represents the variable name of true label, here fill in variable `output_keys`;
4. `block_size`: Represents how long the time step is used for training, specified as the value of variable `train_block_size`;
5. `stride`: Represents the time step interval between two consecutive training samples, specified as 16;
6. `weight_dict`: Represents the weight of the loss function between model output variables and true labels, generated here using `output_keys` and `weights`.

The "sampler" field defines the used `Sampler` class name as `BatchSampler`, and also specifies that the parameters `drop_last` and `shuffle` are both `True` when initializing this class.

`train_dataloader_cfg` also defines the values of `batch_size` and `num_workers`.

The code for defining supervised constraints is as follows:

``` py linenums="76" title="examples/rossler/train_enn.py"
--8<--
examples/rossler/train_enn.py:76:86
--8<--
```

The first parameter of `SupervisedConstraint` is the data loading method, here `train_dataloader_cfg` defined above is used;

The second parameter is the definition of loss function, here MSELoss with L2Decay is used, class name is `MSELossWithL2Decay`, `regularization_dict` sets the variable name and corresponding weight of regularization;

The third parameter indicates how to calculate the intermediate variables that need to be constrained during training. Here, the variable we constrain is the output of the network;

The fourth parameter is the name of the constraint condition, which is convenient for subsequent indexing. Here it is named "Sup".

#### 3.2.2 Model Construction

In this case, the input and output of the Embedding model are the position coordinates $(x, y, z)$ of points in physical space. A fully connected layer is used to implement the Embedding model, as shown in the figure below.

<figure markdown>
  ![rossler_embedding](https://paddle-org.bj.bcebos.com/paddlescience/docs/rossler/rossler_embedding.png){ loading=lazy }
  <figcaption>Embedding Network Model</figcaption>
</figure>

Expressed in PaddleScience code as follows:

``` py linenums="92" title="examples/rossler/train_enn.py"
--8<--
examples/rossler/train_enn.py:92:99
--8<--
```

Among them, the first two parameters of `RosslerEmbedding` have been described above and will not be repeated here. The third and fourth parameters of the network model are the mean and variance of the training dataset, which are used to normalize the input data. The code for calculating mean and variance is expressed as follows:

``` py linenums="32" title="examples/rossler/train_enn.py"
--8<--
examples/rossler/train_enn.py:32:43
--8<--
```

#### 3.2.3 Learning Rate and Optimizer Construction

The learning rate method used in this case is `ExponentialDecay`, and the learning rate size is set to 0.001. The optimizer uses `Adam`, and gradient clipping uses Paddle's built-in `ClipGradByGlobalNorm` method. Expressed in PaddleScience code as follows

``` py linenums="101" title="examples/rossler/train_enn.py"
--8<--
examples/rossler/train_enn.py:101:110
--8<--
```

#### 3.2.4 Validator Construction

During the training process of this case, the training status of the current model will be evaluated using the validation set at certain training round intervals, and `SupervisedValidator` is needed to construct the validator. The code is as follows:

``` py linenums="114" title="examples/rossler/train_enn.py"
--8<--
examples/rossler/train_enn.py:114:141
--8<--
```

The `SupervisedValidator` validator is similar to `SupervisedConstraint`, the difference is that the validator needs to set the evaluation metric `metric`, here `ppsci.metric.MSE` is used.

#### 3.2.5 Model Training and Evaluation

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver`, and then start training and evaluation.

``` py linenums="143" title="examples/rossler/train_enn.py"
--8<--
examples/rossler/train_enn.py:143:157
--8<--
```

### 3.3 Transformer Model

The previous section introduced how to construct the training and evaluation of the Embedding model. In this section, we will introduce how to use the trained Embedding model to train the Transformer model. Because the steps for training the Transformer model are basically similar to the steps for training the Embedding model, the various parameters in the repeated parts of the two are not introduced in detail in this section. First, the various parameter variables defined in the code are shown below, and the specific meaning of each parameter will be explained when used below.

``` yaml linenums="23" title="examples/rossler/conf/transformer.yaml"
--8<--
examples/rossler/conf/transformer.yaml:23:34
--8<--
```

#### 3.3.1 Constraint Construction

The Transformer model also solves problems based on data-driven methods, so it is necessary to use `SupervisedConstraint` built in PaddleScience to construct supervised constraints. Before defining constraints, you need to first specify various parameters used for data loading in supervised constraints. The code is as follows:

``` py linenums="64" title="examples/rossler/train_transformer.py"
--8<--
examples/rossler/train_transformer.py:64:82
--8<--
```

The various parameters for data loading are basically consistent with those in the Embedding model and will not be repeated. It should be noted that since the input data for Transformer model training is the output data of the Encoder module of the Embedding model, we use the trained Embedding model as a parameter of `RosslerDataset`, and first map the training data to the coding space during initialization.

The code for defining supervised constraints is as follows:

``` py linenums="84" title="examples/rossler/train_transformer.py"
--8<--
examples/rossler/train_transformer.py:84:89
--8<--
```

#### 3.3.2 Model Construction

In this case, the input and output of the Transformer model are vectors in the coding space. The Transformer structure used is as follows:

<figure markdown>
  ![rossler_transformer](https://paddle-org.bj.bcebos.com/paddlescience/docs/rossler/rossler_transformer.png){ loading=lazy }
  <figcaption>Transformer Network Model</figcaption>
</figure>

Expressed in PaddleScience code as follows:

``` py linenums="95" title="examples/rossler/train_transformer.py"
--8<--
examples/rossler/train_transformer.py:95:95
--8<--
```

In addition to filling in `input_keys` and `output_keys`, the class `PhysformerGPT2` also needs to set the number of layers of the Transformer model `num_layers`, the size of the context `num_ctx`, the length of the input Embedding vector `embed_size`, and the parameter of the multi-head attention mechanism `num_heads`. The values filled in here are 4, 64, 32, 4.

#### 3.3.3 Learning Rate and Optimizer Construction

The learning rate method used in this case is `CosineWarmRestarts`, and the learning rate size is set to 0.001. The optimizer uses `Adam`, and gradient clipping uses Paddle's built-in `ClipGradByGlobalNorm` method. Expressed in PaddleScience code as follows:

``` py linenums="97" title="examples/rossler/train_transformer.py"
--8<--
examples/rossler/train_transformer.py:97:104
--8<--
```

#### 3.3.4 Validator Construction

During the training process, the training status of the current model will be evaluated using the validation set at certain training round intervals, and `SupervisedValidator` is needed to construct the validator. Expressed in PaddleScience code as follows:

``` py linenums="107" title="examples/rossler/train_transformer.py"
--8<--
examples/rossler/train_transformer.py:107:132
--8<--
```

#### 3.3.5 Visualizer Construction

In this case, the visualizer can be constructed to visualize the evaluation results during model evaluation. Since the output data of the Transformer model is the predicted data in the coding space and cannot be directly visualized, it is necessary to additionally transform the output data to the physical state space using the Decoder module of the Embedding network.

In this paper, the code for transforming the output data of the Transformer model to the physical state space is defined first:

``` py linenums="34" title="examples/rossler/train_transformer.py"
--8<--
examples/rossler/train_transformer.py:34:52
--8<--
```

``` py linenums="63" title="examples/rossler/train_transformer.py"
--8<--
examples/rossler/train_transformer.py:63:64
--8<--
```

It can be seen that the program first loads the trained Embedding model, and then implements the transformation from the encoding vector to the physical state space in the `__call__` function of `OutputTransform`.

After defining the above code, you can implement the construction of the visualizer code:

``` py linenums="134" title="examples/rossler/train_transformer.py"
--8<--
examples/rossler/train_transformer.py:134:152
--8<--
```

First use the dataset in `mse_validator` above for visualization, and also introduce the `vis_data_nums` variable to control the number of samples to be visualized. Finally, build the visualizer through `VisualizerScatter3D`.

#### 3.3.6 Model Training, Evaluation and Visualization

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver`, and then start training and evaluation.

``` py linenums="154" title="examples/rossler/train_transformer.py"
--8<--
examples/rossler/train_transformer.py:154:172
--8<--
```

## 4. Complete Code

``` py linenums="1" title="rossler/train_enn.py"
--8<--
examples/rossler/train_enn.py
--8<--
```

``` py linenums="1" title="rossler/train_transformer.py"
--8<--
examples/rossler/train_transformer.py
--8<--
```

## 5. Result Display

The figure below shows the model prediction results and traditional numerical differentiation prediction results under two different initial conditions.

<figure markdown>
  ![result_states0](https://paddle-org.bj.bcebos.com/paddlescience/docs/rossler/result_states0.png){ loading=lazy }
  <figcaption>Model prediction result ("pred_states") vs traditional numerical differentiation result ("states")</figcaption>
</figure>

<figure markdown>
  ![result_states1](https://paddle-org.bj.bcebos.com/paddlescience/docs/rossler/result_states1.png){ loading=lazy }
  <figcaption>Model prediction result ("pred_states") vs traditional numerical differentiation result ("states")</figcaption>
</figure>
