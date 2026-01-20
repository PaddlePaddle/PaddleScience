# Lorenz System

<a href="https://aistudio.baidu.com/aistudio/projectdetail/6206798?contributionType=1&sUid=455441&shared=1&ts=1684477535039" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

=== "Model Training Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_training_rk.hdf5 -P ./datasets/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_valid_rk.hdf5 -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_training_rk.hdf5 --create-dirs -o ./datasets/lorenz_training_rk.hdf5
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_valid_rk.hdf5 --create-dirs -o ./datasets/lorenz_valid_rk.hdf5
    python train_enn.py
    python train_transformer.py
    ```

=== "Model Evaluation Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_training_rk.hdf5 -P ./datasets/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_valid_rk.hdf5 -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_training_rk.hdf5 --create-dirs -o ./datasets/lorenz_training_rk.hdf5
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_valid_rk.hdf5 --create-dirs -o ./datasets/lorenz_valid_rk.hdf5
    python train_enn.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/lorenz/lorenz_pretrained.pdparams
    python train_transformer.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/lorenz/lorenz_transformer_pretrained.pdparams EMBEDDING_MODEL_PATH=https://paddle-org.bj.bcebos.com/paddlescience/models/lorenz/lorenz_pretrained.pdparams
    ```

=== "Model Export Command"

    ``` sh
    python train_transformer.py mode=export EMBEDDING_MODEL_PATH=https://paddle-org.bj.bcebos.com/paddlescience/models/lorenz/lorenz_pretrained.pdparams
    ```

=== "Model Inference Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_training_rk.hdf5 -P ./datasets/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_valid_rk.hdf5 -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_training_rk.hdf5 --create-dirs -o ./datasets/lorenz_training_rk.hdf5
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_valid_rk.hdf5 --create-dirs -o ./datasets/lorenz_valid_rk.hdf5
    python train_transformer.py mode=infer
    ```

| Model | MSE |
| :-- | :-- |
| [lorenz_transformer_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/lorenz/lorenz_transformer_pretrained.pdparams) | 0.054 |

## 1. Background Introduction

The Lorenz System, named after the American meteorologist Edward N. Lorenz who first proposed it in an article in 1963, is also known as the "Lorenz chaotic system". The famous "Butterfly Effect", that is, "a butterfly in the tropical rainforest of the Amazon River basin in South America occasionally flapping its wings can cause a tornado in Texas, USA two weeks later", also originated from this article. The Lorenz system is characterized by complex and uncertain dynamic behaviors under certain parameter conditions, including sensitivity to initial conditions and unpredictability of long-term behavior. This chaotic behavior exists in nature and many practical application fields, such as climate change, stock market fluctuations, etc. The Lorenz system is extremely sensitive to numerical disturbances and is a good benchmark for evaluating the accuracy of machine learning (deep learning) models.

## 2. Problem Definition

State equations of the Lorenz system:

$$
\begin{cases}
  \dfrac{\partial x}{\partial t} = \sigma(y - x), & \\
  \dfrac{\partial y}{\partial t} = x(\rho - z) - y, & \\
  \dfrac{\partial z}{\partial t} = xy - \beta z
\end{cases}
$$

When the parameters take the following values, the system exhibits classic chaotic characteristics:

$$\rho = 28, \sigma = 10, \beta = \frac{8}{3}$$

In this case, it is required to predict the trajectory of the point in the future period given the coordinates of the point at the initial moment.

## 3. Problem Solving

Next, we will explain how to solve this problem using deep learning methods based on PaddleScience code. This case is based on the method of the paper [Transformers for Modeling Physical Systems](https://arxiv.org/abs/2010.03957). Next, we will first briefly introduce the theoretical method of this paper, then introduce the dataset used, and finally explain the construction of supervised constraints and model construction for the two training steps of this method (Embedding model training, Transformer model training), while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Method Introduction

The Transformer structure has achieved great success in the fields of NLP and CV, but it has not been explored more in modeling physical systems. In the article [Transformers for Modeling Physical Systems](https://arxiv.org/abs/2010.03957), the authors proposed a Transformer-based network structure for modeling physical systems. Experimental results show that the proposed method can accurately model different dynamic systems and is better than other traditional methods.

As shown in the figure below, the method mainly includes two network models: Embedding model and Transformer model. Among them, the Encoder module of the Embedding model is responsible for encoding physical state variables into encoding vectors, and the Decoder module is responsible for mapping encoding vectors to physical state variables; the Transformer model acts on the encoding space, its input is the output of the Encoder module of the Embedding model, using the encoding vector at the current moment to predict the encoding vector at the next moment, and the predicted encoding vector can be decoded by the Decoder module of the Embedding model to obtain the corresponding physical state variable. During model training, the Embedding model is trained first, and then the parameters of the Embedding model are frozen to train the Transformer model. For details of this method, please refer to the paper [Transformers for Modeling Physical Systems](https://arxiv.org/abs/2010.03957).

<figure markdown>
  ![trphysx-arch](https://paddle-org.bj.bcebos.com/paddlescience/docs/lorenz/trphysx-arch.png){ loading=lazy }
  <figcaption>Left: Embedding network structure, Right: Transformer network structure</figcaption>
</figure>

### 3.2 Dataset Introduction

The dataset uses data provided in [Transformer-Physx](https://github.com/zabaras/transformer-physx). This dataset is obtained using the Runge-Kutta traditional numerical solution method, with a time step size of 0.01, and the initial position is randomly selected from the following range:

$$x_{0} \sim(-20, 20), y_{0} \sim(-20, 20), z_{0} \sim(10, 40)$$

The division of the dataset is as follows:

|Dataset |Number of Time Series|Number of Time Steps|Download Link|
|:----:|:---------:|:--------:|:--------:|
|Training Set |2048       |256       |[lorenz_training_rk.hdf5](https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_training_rk.hdf5)|
|Validation Set |64         |1024      |[lorenz_valid_rk.hdf5](https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_valid_rk.hdf5)|

The official website of the dataset is: <https://zenodo.org/record/5148524#.ZDe77-xByrc>

### 3.3 Embedding Model

First, the parameter variables defined in the code are displayed. The specific meaning of each parameter will be explained when used below.

``` yaml linenums="26" title="examples/conf/enn.yaml"
--8<--
examples/lorenz/conf/enn.yaml:26:34
--8<--
```

#### 3.3.1 Constraint Construction

This case solves the problem based on data-driven methods, so it is necessary to use `SupervisedConstraint` built in PaddleScience to construct supervised constraints. Before defining constraints, you need to first specify various parameters used for data loading in supervised constraints. The code is as follows:

``` py linenums="51" title="examples/lorenz/train_enn.py"
--8<--
examples/lorenz/train_enn.py:51:70
--8<--
```

Among them, the "dataset" field defines the `Dataset` class name used as `LorenzDataset`, and also specifies the values of parameters when the class is initialized:

1. `file_path`: represents the file path of the training dataset, specified as the value of variable `train_file_path`;
2. `input_keys`: represents the variable name of the model input data, here fill in the variable `input_keys`;
3. `label_keys`: represents the variable name of the real label, here fill in the variable `output_keys`;
4. `block_size`: represents how long the time step is used for training, specified as the value of variable `train_block_size`;
5. `stride`: represents the time step interval between two continuous training samples, specified as 16;
6. `weight_dict`: represents the weight of the loss function of each variable output by the model and the real label, generated here using `output_keys` and `weights`.

The "sampler" field defines the `Sampler` class name used as `BatchSampler`, and also specifies that the parameters `drop_last` and `shuffle` are both `True` when the class is initialized.

`train_dataloader_cfg` also defines the values of `batch_size` and `num_workers`.

The code for defining supervised constraints is as follows:

``` py linenums="72" title="examples/lorenz/train_enn.py"
--8<--
examples/lorenz/train_enn.py:72:85
--8<--
```

The first parameter of `SupervisedConstraint` is the data loading method, here `train_dataloader_cfg` defined above is used;

The second parameter is the definition of loss function, here `MSELossWithL2Decay` with L2Decay is used, and `regularization_dict` sets the variable name and corresponding weight of regularization;

The third parameter indicates how to calculate the intermediate variables that need to be constrained during training. Here the variable we constrain is the output of the network;

The fourth parameter is the name of the constraint condition, which is convenient for subsequent indexing. Here it is named "Sup".

#### 3.3.2 Model Construction

In this case, the input and output of the Embedding model are the position coordinates $(x, y, z)$ of points in physical space, and fully connected layers are used to implement the Embedding model, as shown in the figure below.

<figure markdown>
  ![lorenz_embedding](https://paddle-org.bj.bcebos.com/paddlescience/docs/lorenz/lorenz_embedding.png){ loading=lazy }
  <figcaption>Embedding Network Model</figcaption>
</figure>

Expressed in PaddleScience code as follows:

``` py linenums="91" title="examples/lorenz/train_enn.py"
--8<--
examples/lorenz/train_enn.py:91:97
--8<--
```

Among them, the first two parameters of `LorenzEmbedding` have been described above and will not be repeated here. The third and fourth parameters of the network model are the mean and variance of the training dataset, used to normalize the input data. The code for calculating the mean and variance is expressed as follows:

``` py linenums="32" title="examples/lorenz/train_enn.py"
--8<--
examples/lorenz/train_enn.py:32:39
--8<--
```

#### 3.3.3 Learning Rate and Optimizer Construction

The learning rate method used in this case is `ExponentialDecay`, and the learning rate size is set to 0.001. The optimizer uses `Adam`, and gradient clipping uses the `ClipGradByGlobalNorm` method built in Paddle. Expressed in PaddleScience code as follows:

``` py linenums="99" title="examples/lorenz/train_enn.py"
--8<--
examples/lorenz/train_enn.py:99:108
--8<--
```

#### 3.3.4 Validator Construction

During the training process of this case, the training status of the current model will be evaluated using the validation set at certain training round intervals, and `SupervisedValidator` is needed to construct the validator. The code is as follows:

``` py linenums="112" title="examples/lorenz/train_enn.py"
--8<--
examples/lorenz/train_enn.py:112:139
--8<--
```

The `SupervisedValidator` validator is quite similar to `SupervisedConstraint`, the difference is that the validator needs to set evaluation metric `metric`, here `ppsci.metric.MSE` is used.

#### 3.3.5 Model Training and Evaluation

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver`, and then start training and evaluation.

``` py linenums="142" title="examples/lorenz/train_enn.py"
--8<--
examples/lorenz/train_enn.py:142:156
--8<--
```

### 3.4 Transformer Model

The above introduced how to construct the training and evaluation of the Embedding model. This section will introduce how to use the trained Embedding model to train the Transformer model. Because the steps for training the Transformer model are basically similar to the steps for training the Embedding model, the parameters in the overlapping parts of the two will not be described in detail in this section. First, the parameter variables defined in the code are displayed as follows, and the specific meaning of each parameter will be explained when used below.

``` yaml linenums="36" title="examples/lorenz/conf/transformer.yaml"
--8<--
examples/lorenz/conf/transformer.yaml:36:43
--8<--
```

#### 3.4.1 Constraint Construction

The Transformer model also solves the problem based on data-driven methods, so it is necessary to use `SupervisedConstraint` built in PaddleScience to construct supervised constraints. Before defining constraints, you need to first specify various parameters used for data loading in supervised constraints. The code is as follows:

``` py linenums="68" title="examples/lorenz/train_transformer.py"
--8<--
examples/lorenz/train_transformer.py:68:85
--8<--
```

The various parameters for data loading are basically consistent with those in the Embedding model and will not be repeated. It should be noted that since the input data for Transformer model training is the output data of the Encoder module of the Embedding model, we pass the trained Embedding model as a parameter to `LorenzDataset`, and map the training data to the encoding space first during initialization.

The code for defining supervised constraints is as follows:

``` py linenums="87" title="examples/lorenz/train_transformer.py"
--8<--
examples/lorenz/train_transformer.py:87:92
--8<--
```

#### 3.4.2 Model Construction

In this case, the input and output of the Transformer model are both vectors in the encoding space. The Transformer structure used is as follows:

<figure markdown>
  ![lorenz_transformer](https://paddle-org.bj.bcebos.com/paddlescience/docs/lorenz/lorenz_transformer.png){ loading=lazy }
  <figcaption>Transformer Network Model</figcaption>
</figure>

Expressed in PaddleScience code as follows:

``` py linenums="98" title="examples/lorenz/train_transformer.py"
--8<--
examples/lorenz/train_transformer.py:98:98
--8<--
```

In addition to filling in `input_keys` and `output_keys`, the class `PhysformerGPT2` also needs to set the number of layers of the Transformer model `num_layers`, the context size `num_ctx`, the length of the input Embedding vector `embed_size`, and the parameter `num_heads` of the multi-head attention mechanism. The values filled in here are 4, 64, 32, 4.

#### 3.4.3 Learning Rate and Optimizer Construction

The learning rate method used in this case is `CosineWarmRestarts`, and the learning rate size is set to 0.001. The optimizer uses `Adam`, and gradient clipping uses the `ClipGradByGlobalNorm` method built in Paddle. Expressed in PaddleScience code as follows:

``` py linenums="101" title="examples/lorenz/train_transformer.py"
--8<--
examples/lorenz/train_transformer.py:101:107
--8<--
```

#### 3.4.4 Validator Construction

During the training process, the training status of the current model will be evaluated using the validation set at certain training round intervals, and `SupervisedValidator` is needed to construct the validator. Expressed in PaddleScience code as follows:

``` py linenums="110" title="examples/lorenz/train_transformer.py"
--8<--
examples/lorenz/train_transformer.py:110:135
--8<--
```

#### 3.4.5 Visualizer Construction

In this case, a visualizer can be constructed to visualize the evaluation results during model evaluation. Since the output data of the Transformer model is predicted data in the encoding space and cannot be directly visualized, the output data needs to be additionally transformed to the physical state space using the Decoder module of the Embedding network.

In this article, the code for transforming the output data of the Transformer model to the physical state space is defined first:

``` py linenums="34" title="examples/lorenz/train_transformer.py"
--8<--
examples/lorenz/train_transformer.py:34:52
--8<--
```

``` py linenums="64" title="examples/lorenz/train_transformer.py"
--8<--
examples/lorenz/train_transformer.py:64:65
--8<--
```

It can be seen that the program first loads the trained Embedding model, and then implements the transformation from the encoding vector to the physical state space in the `__call__` function of `OutputTransform`.

After defining the above code, the construction of the visualizer code can be implemented:

``` py linenums="138" title="examples/lorenz/train_transformer.py"
--8<--
examples/lorenz/train_transformer.py:138:155
--8<--
```

First, use the dataset in `mse_validator` above for visualization, and introduce the `vis_data_nums` variable to control the number of samples to be visualized. Finally, construct the visualizer through `VisualizerScatter3D`.

#### 3.4.6 Model Training, Evaluation and Visualization

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver`, and then start training and evaluation.

``` py linenums="157" title="examples/lorenz/train_transformer.py"
--8<--
examples/lorenz/train_transformer.py:157:175
--8<--
```

## 4. Complete Code

``` py linenums="1" title="lorenz/train_enn.py"
--8<--
examples/lorenz/train_enn.py
--8<--
```

``` py linenums="1" title="lorenz/train_transformer.py"
--8<--
examples/lorenz/train_transformer.py
--8<--
```

## 5. Result Display

The following figures show the model prediction results and traditional numerical differentiation prediction results under two different initial conditions.

<figure markdown>
  ![result_states0](https://paddle-org.bj.bcebos.com/paddlescience/docs/lorenz/result_states0.png){ loading=lazy }
  <figcaption>Model prediction results ("pred_states") vs traditional numerical differentiation results ("states")</figcaption>
</figure>

<figure markdown>
  ![result_states1](https://paddle-org.bj.bcebos.com/paddlescience/docs/lorenz/result_states1.png){ loading=lazy }
  <figcaption>Model prediction results ("pred_states") vs traditional numerical differentiation results ("states")</figcaption>
</figure>
