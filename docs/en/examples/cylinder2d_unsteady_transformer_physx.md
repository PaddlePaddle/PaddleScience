# 2D-Cylinder (2D Flow Around a Cylinder)

<a href="https://aistudio.baidu.com/aistudio/projectdetail/6178818?sUid=455441&shared=1&ts=1684397945680" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

=== "Model Training Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/cylinder_training.hdf5 -P ./datasets/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/cylinder_valid.hdf5 -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/cylinder_training.hdf5 --create-dirs -o ./datasets/cylinder_training.hdf5
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/cylinder_valid.hdf5 --create-dirs -o ./datasets/cylinder_valid.hdf5
    python train_enn.py
    python train_transformer.py
    ```

=== "Model Evaluation Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/cylinder_training.hdf5 -P ./datasets/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/cylinder_valid.hdf5 -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/cylinder_training.hdf5 --create-dirs -o ./datasets/cylinder_training.hdf5
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/cylinder_valid.hdf5 --create-dirs -o ./datasets/cylinder_valid.hdf5
    python train_enn.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/cylinder/cylinder_pretrained.pdparams
    python train_transformer.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/cylinder/cylinder_transformer_pretrained.pdparams EMBEDDING_MODEL_PATH=https://paddle-org.bj.bcebos.com/paddlescience/models/cylinder/cylinder_pretrained.pdparams
    ```

=== "Model Export Command"

    ``` sh
    python train_transformer.py mode=export EMBEDDING_MODEL_PATH=https://paddle-org.bj.bcebos.com/paddlescience/models/cylinder/cylinder_pretrained.pdparams
    ```

=== "Model Inference Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/cylinder_training.hdf5 -P ./datasets/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/cylinder_valid.hdf5 -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/cylinder_training.hdf5 --create-dirs -o ./datasets/cylinder_training.hdf5
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/cylinder_valid.hdf5 --create-dirs -o ./datasets/cylinder_valid.hdf5
    python train_transformer.py mode=infer
    ```

| Model | MSE |
| :-- | :-- |
| [cylinder_transformer_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/cylinder/cylinder_transformer_pretrained.pdparams) | 1.093 |

## 1. Background Introduction

The problem of flow around a cylinder can be applied to many fields. For example, in industrial design, it can be used to simulate and optimize fluid flow in various equipment, such as wind turbines, hydrodynamic performance of cars and aircraft, etc. In the field of environmental protection, the problem of flow around a cylinder also has applications, such as predicting and controlling river floods, studying the diffusion of pollutants, etc. In addition, in engineering practice, such as fluid dynamics, hydrostatics, heat exchange, aerodynamics and other fields, the problem of flow around a cylinder also has practical significance.

2D Flow Around a Cylinder refers to the flow pattern of low-speed steady flow around a two-dimensional cylinder, which is only related to the $Re$ number. When $Re \le 1$, the inertial force in the flow field occupies a secondary position compared with the viscous force, the streamlines upstream and downstream of the cylinder are symmetrical, and the drag coefficient is approximately inversely proportional to $Re$ (drag coefficient is 10~60). The flow around this $Re$ number range is called the Stokes region; as $Re$ increases, the streamlines upstream and downstream of the cylinder gradually lose symmetry.

## 2. Problem Definition

Mass conservation:

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

$x$ momentum conservation:

$$
\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} = -\frac{1}{\rho}\frac{\partial p}{\partial x} + \nu(\frac{\partial ^2 u}{\partial x ^2} + \frac{\partial ^2 u}{\partial y ^2})
$$

$y$ momentum conservation:

$$
\frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} = -\frac{1}{\rho}\frac{\partial p}{\partial y} + \nu(\frac{\partial ^2 v}{\partial x ^2} + \frac{\partial ^2 v}{\partial y ^2})
$$

**Let:**

$t^* = \frac{L}{U_0}$

$x^*=y^* = L$

$u^*=v^* = U_0$

$p^* = \rho {U_0}^2$

**Define:**

Dimensionless time $\tau = \frac{t}{t^*}$

Dimensionless coordinate $x：X = \frac{x}{x^*}$; Dimensionless coordinate $y：Y = \frac{y}{y^*}$

Dimensionless velocity $x：U = \frac{u}{u^*}$; Dimensionless velocity $y：V = \frac{v}{u^*}$

Dimensionless pressure $P = \frac{p}{p^*}$

Reynolds number $Re = \frac{L U_0}{\nu}$

The following dimensionless Navier-Stokes equations can be obtained and applied to the interior of the fluid domain:

Mass conservation:

$$
\frac{\partial U}{\partial X} + \frac{\partial U}{\partial Y} = 0
$$

$x$ momentum conservation:

$$
\frac{\partial U}{\partial \tau} + U\frac{\partial U}{\partial X} + V\frac{\partial U}{\partial Y} = -\frac{\partial P}{\partial X} + \frac{1}{Re}(\frac{\partial ^2 U}{\partial X^2} + \frac{\partial ^2 U}{\partial Y^2})
$$

$y$ momentum conservation:

$$
\frac{\partial V}{\partial \tau} + U\frac{\partial V}{\partial X} + V\frac{\partial V}{\partial Y} = -\frac{\partial P}{\partial Y} + \frac{1}{Re}(\frac{\partial ^2 V}{\partial X^2} + \frac{\partial ^2 V}{\partial Y^2})
$$

For the fluid domain boundary and the inner circumference boundary of the fluid domain, Dirichlet boundary conditions need to be applied:

Fluid domain inlet boundary:

$$
u=1, v=0
$$

Circumference boundary:

$$
u=0, v=0
$$

Fluid domain outlet boundary:

$$
p=0
$$

## 3. Problem Solving

Next, we will explain how to solve this problem using deep learning methods based on PaddleScience code. This case is solved based on the method of the paper [Transformers for Modeling Physical Systems](https://arxiv.org/abs/2010.03957). For the theoretical part of this method, please refer to [this document](lorenz.md#31) or [original paper](https://arxiv.org/abs/2010.03957). Next, the dataset used will be introduced first, and then the supervised constraint construction and model construction of the two training steps of this method (Embedding model training, Transformer model training) will be explained. For other details, please refer to [API Documentation](../api/arch.md).

### 3.1 Dataset Introduction

The dataset uses the data provided in [Transformer-Physx](https://github.com/zabaras/transformer-physx). The data in this dataset is calculated using OpenFOAM, each time step size is 0.5, and $Re$ is randomly selected from the following range:

$$Re \sim(100, 750)$$

The division of the dataset is as follows:

|Dataset |Number of flow field simulations|Number of time steps|Download address|
|:----:|:---------:|:--------:|:--------:|
|Training set |27         |400       |[cylinder_training.hdf5](https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/cylinder_training.hdf5)|
|Validation set |6          |400       |[cylinder_valid.hdf5](https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/cylinder_valid.hdf5)|

The official website of the dataset is: <https://zenodo.org/record/5148524#.ZDe77-xByrc>

### 3.2 Embedding Model

First, the various parameter variables defined in the code are displayed. The specific meaning of each parameter will be explained when used below.

``` py linenums="58" title="examples/cylinder/2d_unsteady/transformer_physx/train_enn.py"
--8<--
examples/cylinder/2d_unsteady/transformer_physx/train_enn.py:58:59
--8<--
```

#### 3.2.1 Constraint Construction

This case solves the problem based on data-driven methods, so it is necessary to use `SupervisedConstraint` built in PaddleScience to construct supervised constraints. Before defining constraints, you need to first specify various parameters used for data loading in supervised constraints. The code is as follows:

``` py linenums="61" title="examples/cylinder/2d_unsteady/transformer_physx/train_enn.py"
--8<--
examples/cylinder/2d_unsteady/transformer_physx/train_enn.py:61:80
--8<--
```

Among them, the "dataset" field defines the used `Dataset` class name as `CylinderDataset`, and also specifies the value of parameters when initializing this class:

1. `file_path`: represents the file path of the training dataset, specified as the value of variable `train_file_path`;
2. `input_keys`: represents the variable name of model input data, fill in variable `input_keys` here;
3. `label_keys`: represents the variable name of true label, fill in variable `output_keys` here;
4. `block_size`: represents how long time steps are used for training, specified as the value of variable `train_block_size`;
5. `stride`: represents the time step interval between two consecutive training samples, specified as 16;
6. `weight_dict`: represents the weight of the loss function of each variable output by the model and the true label, generated here using `output_keys` and `weights`.

The "sampler" field defines the used `Sampler` class name as `BatchSampler`, and also specifies that the parameters `drop_last` and `shuffle` are both `True` when initializing this class.

`train_dataloader_cfg` also defines the values of `batch_size` and `num_workers`.

The code for defining supervised constraints is as follows:

``` py linenums="82" title="examples/cylinder/2d_unsteady/transformer_physx/train_enn.py"
--8<--
examples/cylinder/2d_unsteady/transformer_physx/train_enn.py:82:94
--8<--
```

The first parameter of `SupervisedConstraint` is the data loading method, here `train_dataloader_cfg` defined above is used;

The second parameter is the definition of loss function, here `MSELossWithL2Decay` with L2Decay is used, and `regularization_dict` sets the regularization variable name and corresponding weight;

The third parameter indicates how to calculate the intermediate variables that need to be constrained during training. Here the variable we constrain is the output of the network;

The fourth parameter is the name of the constraint condition, which is convenient for subsequent indexing. Here it is named "Sup".

#### 3.2.2 Model Construction

In this case, the Embedding model uses a convolutional neural network to implement the Embedding model, as shown in the figure below.

<figure markdown>
  ![cylinder-embedding](https://paddle-org.bj.bcebos.com/paddlescience/docs/cylinder2d_unsteady_transformer_physx/cylinder_embedding.png){ loading=lazy }
  <figcaption>Embedding Network Model</figcaption>
</figure>

Expressed in PaddleScience code as follows:

``` py linenums="104" title="examples/cylinder/2d_unsteady/transformer_physx/train_enn.py"
--8<--
examples/cylinder/2d_unsteady/transformer_physx/train_enn.py:104:109
--8<--
```

Among them, the first two parameters of `CylinderEmbedding` have been described in the previous text, so they will not be repeated here. The third and fourth parameters of the network model are the mean and variance of the training dataset, which are used to normalize input data. The code for calculating mean and variance is expressed as follows:

``` py linenums="32" title="examples/cylinder/2d_unsteady/transformer_physx/train_enn.py"
--8<--
examples/cylinder/2d_unsteady/transformer_physx/train_enn.py:32:49
--8<--
```

#### 3.2.3 Learning Rate and Optimizer Construction

The learning rate method used in this case is `ExponentialDecay`, and the learning rate size is set to 0.001. The optimizer uses `Adam`, and gradient clipping uses the `ClipGradByGlobalNorm` method built in Paddle. Expressed in PaddleScience code as follows:

``` py linenums="111" title="examples/cylinder/2d_unsteady/transformer_physx/train_enn.py"
--8<--
examples/cylinder/2d_unsteady/transformer_physx/train_enn.py:111:120
--8<--
```

#### 3.2.4 Validator Construction

In this case, the validation set is used to evaluate the training status of the current model at certain training epoch intervals during the training process, and `SupervisedValidator` is needed to construct the validator. The code is as follows:

``` py linenums="124" title="examples/cylinder/2d_unsteady/transformer_physx/train_enn.py"
--8<--
examples/cylinder/2d_unsteady/transformer_physx/train_enn.py:124:151
--8<--
```

The `SupervisedValidator` validator is similar to `SupervisedConstraint`, the difference is that the validator needs to set the evaluation metric `metric`, here `ppsci.metric.MSE` is used.

#### 3.2.5 Model Training and Evaluation

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training and evaluation.

``` py linenums="153" title="examples/cylinder/2d_unsteady/transformer_physx/train_enn.py"
--8<--
examples/cylinder/2d_unsteady/transformer_physx/train_enn.py:153:169
--8<--
```

### 3.3 Transformer Model

The previous section introduced how to construct the training and evaluation of the Embedding model. This section will introduce how to use the trained Embedding model to train the Transformer model. Since the steps for training the Transformer model are basically similar to those for training the Embedding model, the parameters in the repeated parts of the two will not be described in detail in this section. First, the various parameter variables defined in the code are displayed as follows. The specific meaning of each parameter will be explained when used below.

``` yaml linenums="23" title="examples/cylinder/2d_unsteady/transformer_physx/conf/transformer.yaml"
--8<--
examples/cylinder/2d_unsteady/transformer_physx/conf/transformer.yaml:23:34
--8<--
```

#### 3.3.1 Constraint Construction

The Transformer model also solves the problem based on data-driven methods, so it is necessary to use `SupervisedConstraint` built in PaddleScience to construct supervised constraints. Before defining constraints, you need to first specify various parameters used for data loading in supervised constraints. The code is as follows:

``` py linenums="68" title="examples/cylinder/2d_unsteady/transformer_physx/train_transformer.py"
--8<--
examples/cylinder/2d_unsteady/transformer_physx/train_transformer.py:68:85
--8<--
```

The parameters for data loading are basically consistent with those in the Embedding model and will not be repeated. It should be noted that since the input data for Transformer model training is the output data of the Encoder module of the Embedding model, we take the trained Embedding model as a parameter of `CylinderDataset`, and first map the training data to the encoding space during initialization.

The code for defining supervised constraints is as follows:

``` py linenums="87" title="examples/cylinder/2d_unsteady/transformer_physx/train_transformer.py"
--8<--
examples/cylinder/2d_unsteady/transformer_physx/train_transformer.py:87:92
--8<--
```

#### 3.3.2 Model Construction

In this case, the input and output of the Transformer model are vectors in the encoding space. The Transformer structure used is as follows:

<figure markdown>
  ![cylinder_transformer](https://paddle-org.bj.bcebos.com/paddlescience/docs/cylinder2d_unsteady_transformer_physx/cylinder_transformer.png){ loading=lazy }
  <figcaption>Transformer Network Model</figcaption>
</figure>

Expressed in PaddleScience code as follows:

``` py linenums="98" title="examples/cylinder/2d_unsteady/transformer_physx/train_transformer.py"
--8<--
examples/cylinder/2d_unsteady/transformer_physx/train_transformer.py:98:98
--8<--
```

In addition to filling in `input_keys` and `output_keys`, the class `PhysformerGPT2` also needs to set the number of layers of the Transformer model `num_layers`, the size of the context `num_ctx`, the length of the input Embedding vector `embed_size`, and the parameter of the multi-head attention mechanism `num_heads`. The values filled in here are 6, 16, 128, 4.

#### 3.3.3 Learning Rate and Optimizer Construction

The learning rate method used in this case is `CosineWarmRestarts`, and the learning rate size is set to 0.001. The optimizer uses `Adam`, and gradient clipping uses the `ClipGradByGlobalNorm` method built in Paddle. Expressed in PaddleScience code as follows:

``` py linenums="100" title="examples/cylinder/2d_unsteady/transformer_physx/train_transformer.py"
--8<--
examples/cylinder/2d_unsteady/transformer_physx/train_transformer.py:100:107
--8<--
```

#### 3.3.4 Validator Construction

During the training process, the validation set is used to evaluate the training status of the current model at certain training epoch intervals, and `SupervisedValidator` is needed to construct the validator. Expressed in PaddleScience code as follows:

``` py linenums="110" title="examples/cylinder/2d_unsteady/transformer_physx/train_transformer.py"
--8<--
examples/cylinder/2d_unsteady/transformer_physx/train_transformer.py:110:135
--8<--
```

#### 3.3.5 Visualizer Construction

In this case, a visualizer can be constructed to visualize the evaluation results during model evaluation. Since the output data of the Transformer model is predicted data in the encoding space and cannot be directly visualized, it is necessary to additionally transform the output data to the physical state space using the Decoder module of the Embedding network.

In this article, the code for transforming the output data of the Transformer model to the physical state space is first defined:

``` py linenums="35" title="examples/cylinder/2d_unsteady/transformer_physx/train_transformer.py"
--8<--
examples/cylinder/2d_unsteady/transformer_physx/train_transformer.py:35:56
--8<--
```

``` py linenums="64" title="examples/cylinder/2d_unsteady/transformer_physx/train_transformer.py"
--8<--
examples/cylinder/2d_unsteady/transformer_physx/train_transformer.py:64:65
--8<--
```

It can be seen that the program first loads the trained Embedding model, and then implements the transformation from encoding vector to physical state space in the `__call__` function of `OutputTransform`.

After defining the above code, the construction of the visualizer code can be implemented:

``` py linenums="146" title="examples/cylinder/2d_unsteady/transformer_physx/train_transformer.py"
--8<--
examples/cylinder/2d_unsteady/transformer_physx/train_transformer.py:146:164
--8<--
```

First, use the dataset in `mse_validator` above for visualization. In addition, the `vis_data_nums` variable is introduced to control the number of samples to be visualized. Finally, construct the visualizer through `VisualizerScatter3D`.

#### 3.3.6 Model Training, Evaluation and Visualization

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training and evaluation.

``` py linenums="166" title="examples/cylinder/2d_unsteady/transformer_physx/train_transformer.py"
--8<--
examples/cylinder/2d_unsteady/transformer_physx/train_transformer.py:166:184
--8<--
```

## 4. Complete Code

``` py linenums="1" title="examples/cylinder/2d_unsteady/transformer_physx/train_transformer.py"
--8<--
examples/cylinder/2d_unsteady/transformer_physx/train_transformer.py
--8<--
```

``` py linenums="1" title="examples/cylinder/2d_unsteady/transformer_physx/train_transformer.py"
--8<--
examples/cylinder/2d_unsteady/transformer_physx/train_transformer.py
--8<--
```

## 5. Result Display

For the problem in this case, the prediction results of the model and the results of traditional numerical differentiation are shown below, where ux and uy represent the velocity in the x and y directions respectively, and p represents the pressure.

<figure markdown>
  ![result_states0](https://paddle-org.bj.bcebos.com/paddlescience/docs/cylinder2d_unsteady_transformer_physx/result_states0.png){ loading=lazy }
  <figcaption>Model prediction results ("pred") vs traditional numerical differentiation results ("target")</figcaption>
</figure>
