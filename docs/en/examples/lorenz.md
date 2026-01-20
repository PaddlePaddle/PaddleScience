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

The Lorenz system, proposed by meteorologist Edward N. Lorenz in 1963, is a seminal model in chaos theory. It famously illustrates the "Butterfly Effect," where small changes in initial conditions can lead to vastly different outcomes—metaphorically, a butterfly flapping its wings in Brazil causing a tornado in Texas.

Mathematically, the Lorenz system describes atmospheric convection using a set of three ordinary differential equations. It exhibits chaotic behavior for certain parameter values, characterized by extreme sensitivity to initial conditions and long-term unpredictability. Due to this sensitivity, the Lorenz system serves as an excellent benchmark for evaluating the precision and stability of machine learning models in capturing complex dynamics.

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

While Transformers have revolutionized NLP and CV, their application to physical system modeling is relatively new. This example implements the method from [Transformers for Modeling Physical Systems](https://arxiv.org/abs/2010.03957), which adapts the Transformer architecture for dynamical systems.

The approach involves two key components:
1.  **Embedding Model**: An autoencoder structure.
    -   **Encoder**: Maps physical state variables to a latent embedding space.
    -   **Decoder**: Reconstructs physical states from the latent vectors.
2.  **Transformer Model**: Operates within the latent space. It predicts the future latent state based on the current latent state (output of the Encoder).

**Training Strategy**:
1.  Train the Embedding model to minimize reconstruction error.
2.  Freeze the Embedding model and train the Transformer to predict dynamics in the latent space.

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

We first define the key hyperparameters in the configuration file:

``` yaml linenums="26" title="examples/conf/enn.yaml"
--8<--
examples/lorenz/conf/enn.yaml:26:34
--8<--
```

#### 3.3.1 Constraint Construction

Since this is a data-driven task, we use `SupervisedConstraint`. First, we configure the data loader:

``` py linenums="51" title="examples/lorenz/train_enn.py"
--8<--
examples/lorenz/train_enn.py:51:70
--8<--
```

- **Dataset**: `LorenzDataset` handles loading the HDF5 data.
    - `block_size`: Length of time sequence for training.
    - `stride`: Step interval between samples.
- **Sampler**: `BatchSampler` with shuffling enabled.

The code for defining supervised constraints is as follows:

``` py linenums="72" title="examples/lorenz/train_enn.py"
--8<--
examples/lorenz/train_enn.py:72:85
--8<--
```

- **Dataloader**: Uses `train_dataloader_cfg`.
- **Loss**: `MSELossWithL2Decay` (MSE with L2 regularization).
- **Target**: Model output.
- **Name**: "Sup".

#### 3.3.2 Model Construction

The Embedding model uses fully connected layers to map physical coordinates $(x, y, z)$ to and from the latent space.

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

We use `ExponentialDecay` for the learning rate (initial lr=0.001) and the `Adam` optimizer with `ClipGradByGlobalNorm` for gradient clipping.

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

Having trained the Embedding model, we now train the Transformer model using the fixed Embedding model. The process is similar, so we focus on the differences.

``` yaml linenums="36" title="examples/lorenz/conf/transformer.yaml"
--8<--
examples/lorenz/conf/transformer.yaml:36:43
--8<--
```

#### 3.4.1 Constraint Construction

We again use `SupervisedConstraint`. The data loading configuration is:

``` py linenums="68" title="examples/lorenz/train_transformer.py"
--8<--
examples/lorenz/train_transformer.py:68:85
--8<--
```

**Note**: The Transformer trains on data in the latent (encoding) space. We pass the pre-trained Embedding model to `LorenzDataset` to map the physical data to the encoding space during initialization.

The code for defining supervised constraints is as follows:

``` py linenums="87" title="examples/lorenz/train_transformer.py"
--8<--
examples/lorenz/train_transformer.py:87:92
--8<--
```

#### 3.4.2 Model Construction

The Transformer operates entirely within the latent encoding space.

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

To visualize results, we must map the Transformer's latent output back to physical space using the Decoder. We define an `OutputTransform` for this purpose:

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

`OutputTransform` loads the Embedding model and decodes the latent vectors. We then construct the visualizer:

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

The plots below compare the model's predictions with the numerical ground truth for two different initial conditions.

<figure markdown>
  ![result_states0](https://paddle-org.bj.bcebos.com/paddlescience/docs/lorenz/result_states0.png){ loading=lazy }
  <figcaption>Model prediction results ("pred_states") vs traditional numerical differentiation results ("states")</figcaption>
</figure>

<figure markdown>
  ![result_states1](https://paddle-org.bj.bcebos.com/paddlescience/docs/lorenz/result_states1.png){ loading=lazy }
  <figcaption>Model prediction results ("pred_states") vs traditional numerical differentiation results ("states")</figcaption>
</figure>
