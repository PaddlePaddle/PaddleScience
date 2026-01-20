# Transolver

!!! note

    Please install related dependencies before running this case: `pip install -r requirements.txt`

=== "Model Training Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlecfd/datasets/pptransformer/mlcfd_data.zip
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlecfd/datasets/pptransformer/mlcfd_data.zip -o mlcfd_data.zip
    unzip mlcfd_data.zip
    python main.py
    ```

    !!! note

        When running for the first time, mlcfd_data will be preprocessed, which takes about one hour. Please wait patiently.

=== "Model Evaluation Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlecfd/datasets/pptransformer/mlcfd_data.zip
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlecfd/datasets/pptransformer/mlcfd_data.zip -o mlcfd_data.zip
    unzip mlcfd_data.zip
    python main.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/transolver/transolver_pretrained.pdparams
    ```

    !!! note

        When running for the first time, mlcfd_data will be preprocessed, which takes about one hour. Please wait patiently.

=== "Model Export Command"

    ``` sh
    python main.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlecfd/datasets/pptransformer/mlcfd_data.zip
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlecfd/datasets/pptransformer/mlcfd_data.zip -o mlcfd_data.zip
    unzip mlcfd_data.zip
    python main.py mode=infer
    ```

    !!! note

        When running for the first time, mlcfd_data will be preprocessed, which takes about one hour. Please wait patiently.

| Pretrained Model | Metrics |
|:--| :--|
| [transolver_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/transolver/transolver_pretrained.pdparams) | rho_d:, 0.99314<br>c_d: 0.01136<br>relative l2 error of press: 0.07829<br>relative l2 error of velocity: 0.02304<br>press: 4.95888<br>velocity: [0.12163974 0.14851639 0.41583335] 0.26443 |

## 1. Background Introduction

Transolver is a neural operator model based on the Transformer architecture for learning solution operators of Partial Differential Equations (PDEs). The core innovation of this model lies in its Physics Attention mechanism, which can efficiently handle physical field prediction problems on irregular meshes.

Compared with traditional Transformer models, Transolver has the following characteristics:

- **Physics-aware Attention Mechanism**: Aggregates irregular grid points into regular representations through slice technology, which not only retains spatial physical information, but also greatly reduces computational complexity
- **Flexible Geometric Adaptability**: Able to handle geometric bodies of arbitrary shapes and unstructured meshes
- **Efficient Computational Performance**: Through the slice attention mechanism, the computational complexity is reduced from $O(N^2)$ to $O(NG)$, where $N$ is the number of grid points and $G$ is the number of slices

This case uses the Transolver model to learn the velocity field and pressure field distribution of the external flow field of a car on the ShapeNet Car dataset. This is a typical Computational Fluid Dynamics (CFD) surrogate modeling problem. By learning a large amount of car shape and corresponding flow field data, the model can quickly predict the flow field distribution of new car shapes, thereby greatly reducing the computational cost of CFD simulation.

## 2. Problem Definition

The goal of this case is to establish a mapping relationship between car geometry and its surrounding flow field (velocity field and pressure field). Specifically:

- **Input**: Grid point coordinates of the car surface and surrounding space $\mathbf{x} \in \mathbb{R}^{N \times 7}$, where $N$ is the number of grid points, and 7 dimensions include geometric information such as spatial coordinates and normal vectors
- **Output**: Velocity vector $\mathbf{v} \in \mathbb{R}^{N \times 3}$ and pressure scalar $p \in \mathbb{R}^{N \times 1}$ at each grid point

The training goal is to minimize the mean square error between the predicted flow field and the real CFD simulation results, while ensuring that the model can accurately predict key aerodynamic parameters such as the drag coefficient of the car.

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, constraint construction, and optimizer construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Model Construction

In this problem, we need to establish a mapping function $f: \mathbb{R}^{N \times 7} \to \mathbb{R}^{N \times 4}$ from grid point coordinates $\mathbf{x}$ to flow field variables $(\mathbf{v}, p)$, that is:

$$
(\mathbf{v}, p) = f(\mathbf{x})
$$

Here we use the Transolver model to represent this mapping function, expressed in PaddleScience code as follows:

``` py linenums="25"
--8<--
examples/transolver/main.py:25:27
--8<--
```

In order to accurately and quickly access the value of specific variables during calculation, we specify here that the input variable name of the network model is `["x"]`, and the output variable names are `["velo_vec", "press"]`. These names are consistent with subsequent code.

The detailed configuration of the model is as follows:

``` yaml linenums="43"
--8<--
examples/transolver/conf/shapenet_car.yaml:43:57
--8<--
```

Where:

- `space_dim`: Input space dimension, set to 7 (including spatial coordinates, normal vectors and other geometric information)
- `n_layers`: Number of Transformer layers, set to 8
- `n_hidden`: Hidden layer dimension, set to 256
- `n_head`: Number of multi-head attention heads, set to 8
- `dropout`: Dropout rate, set to 0
- `act`: Activation function, use `gelu`
- `mlp_ratio`: Hidden layer expansion ratio of MLP layer, set to 2
- `out_dim`: Output dimension list, `[3, 1]` corresponds to velocity field (3 dimensions) and pressure field (1 dimension) respectively
- `slice_num`: Number of slices in slice attention mechanism, set to 32, used to reduce computational complexity
- `ref`: Resolution of reference grid, set to 8
- `unified_pos`: Whether to use unified position encoding, set to False

### 3.2 Model Architecture Details

The core architecture of Transolver includes the following key components:

#### 3.2.1 Preprocessing

Map input geometric features to hidden space:

$$
h = \text{MLP}(\text{concat}(f, x))
$$

Where $f$ is function feature (such as initial field), and $x$ is spatial coordinate.

#### 3.2.2 Physics Attention

This is the core innovation of Transolver, which achieves efficient global information interaction through the following steps:

1. **Slice Aggregation**: Aggregate $N$ irregular grid points into $G$ slice representations

$$
S = \text{softmax}\left(\frac{W_s(h)}{\tau}\right)^T h
$$

Where $S \in \mathbb{R}^{G \times D}$ is slice representation, and $\tau$ is learnable temperature parameter.

2. **Self-Attention on Slices**: Perform standard Transformer attention calculation on slice representations

$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

Where $Q = W_q S, K = W_k S, V = W_v S$.

3. **Disaggregation**: Map slice representations back to original grid points

$$
h' = S \cdot \text{Attn}(Q, K, V) \cdot W
$$

Through this mechanism, the computational complexity is reduced from $O(N^2)$ to $O(NG + G^2)$, which can significantly improve efficiency when $G \ll N$.

#### 3.2.3 Feed-Forward Network (MLP)

Each Transformer block is followed by a feed-forward network:

$$
\text{FFN}(h) = W_2 \cdot \text{GELU}(W_1 h + b_1) + b_2
$$

#### 3.2.4 Layer Normalization and Residual Connection

Each sub-layer uses Layer Normalization (LayerNorm) and residual connection:

$$
\begin{aligned}
h^{(l+1)} &= \text{Attn}(\text{LN}(h^{(l)})) + h^{(l)} \\
h^{(l+1)} &= \text{FFN}(\text{LN}(h^{(l+1)})) + h^{(l+1)}
\end{aligned}
$$

### 3.3 Data Loading

This case uses the ShapeNet Car dataset, which contains CFD simulation results of different car shapes. The data loading code is as follows:

``` py linenums="29"
--8<--
examples/transolver/main.py:29:43
--8<--
```

Dataset configuration:

``` yaml linenums="31"
--8<--
examples/transolver/conf/shapenet_car.yaml:31:40
--8<--
```

Where:

- `data_dir`: Raw data directory
- `save_dir`: Preprocessed data saving directory
- `val_fold_id`: Fold ID used for cross-validation
- `preprocessed`: Whether to use preprocessed data
- `r`: Data downsampling ratio

### 3.4 Constraint Construction

This case uses supervised learning constraints to train the model by minimizing the error between the predicted flow field and the real flow field:

``` py linenums="45"
--8<--
examples/transolver/main.py:45:76
--8<--
```

The loss function contains two parts:

1. Mean square error of velocity field: $\mathcal{L}_{velo} = \text{MSE}(\mathbf{v}_{pred}, \mathbf{v}_{true})$
2. Mean square error of surface pressure field: $\mathcal{L}_{press} = \text{MSE}(p_{pred}|_{surf}, p_{true}|_{surf})$

Total loss is: $\mathcal{L} = \mathcal{L}_{velo} + w_{press} \cdot \mathcal{L}_{press}$, where $w_{press}$ is pressure loss weight.

### 3.5 Optimizer Construction

Use Adam optimizer with exponential decay learning rate strategy:

``` py linenums="78"
--8<--
examples/transolver/main.py:78:89
--8<--
```

Learning rate configuration:

``` yaml linenums="63"
--8<--
examples/transolver/conf/shapenet_car.yaml:63:67
--8<--
```

It includes:

- Warmup phase (`warmup_epoch`): The learning rate gradually increases from $\frac{lr_{max}}{25}$ to $lr_{max}$ in the first 60 epochs
- Decay phase: Then the learning rate decays according to $lr = lr_{max} \times \gamma^{step}$ for each step

### 3.6 Validator Construction

Use validation set to evaluate model performance during training:

``` py linenums="91"
--8<--
examples/transolver/main.py:91:129
--8<--
```

Evaluation metrics include:

- Mean square error of velocity field
- Mean square error of surface pressure field

### 3.7 Model Training and Evaluation

After completing the above settings, pass the instantiated objects to `ppsci.solver.Solver`, and then start training and evaluation:

``` py linenums="140"
--8<--
examples/transolver/main.py:140:142
--8<--
```

### 3.8 Model Evaluation

In the evaluation phase, in addition to calculating conventional error metrics, the prediction error of drag coefficient and Spearman correlation coefficient will also be calculated:

``` py linenums="145"
--8<--
examples/transolver/main.py:145:241
--8<--
```

Evaluation metrics include:

- **Relative L2 Error**: Measure the overall deviation of the predicted field from the true field
- **Root Mean Square Error (RMSE)**: Evaluate point-wise prediction accuracy
- **Drag Coefficient Error**: Evaluate the prediction accuracy of key aerodynamic parameters
- **Spearman Correlation Coefficient ($\rho_d$)**: Evaluate the correlation of drag coefficient ranking

## 4. Complete Code

``` py linenums="1" title="main.py"
--8<--
examples/transolver/main.py
--8<--
```

## 5. Result Display

After model training is completed, the prediction performance can be evaluated on the validation set. The main evaluation metrics include:

- **Relative L2 Error of Velocity Field**: Measure the overall accuracy of velocity field prediction
- **Relative L2 Error of Pressure Field**: Measure the overall accuracy of pressure field prediction
- **Relative Error of Drag Coefficient**: Evaluate the prediction accuracy of key aerodynamic parameters
- **Spearman Correlation Coefficient**: Evaluate the accuracy of drag coefficient ranking of different car shapes

Through training, the Transolver model can predict the flow field distribution around the car with high accuracy, significantly reducing calculation time compared to traditional CFD simulation, providing a fast surrogate model for car aerodynamic shape optimization.

## 6. References

- [Transolver: A Fast Transformer Solver for PDEs on General Geometries](https://arxiv.org/abs/2402.02366)
- [Transolver GitHub Repository](https://github.com/thuml/Transolver)
