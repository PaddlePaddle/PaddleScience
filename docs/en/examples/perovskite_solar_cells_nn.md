# psc_NN (Machine Learning for Perovskite Solar Cells: An Open-Source Pipeline)

!!! note "Notes"

    1. Before starting training, please ensure that the dataset has been correctly placed in the `data/cleaned/` directory.
    2. Training and evaluation require additional dependencies, please install them using `pip install -r requirements.txt`.
    3. For optimal performance, it is recommended to use GPU for training.

=== "Model Training Command"

    ``` sh
    python psc_nn.py
    ```

=== "Model Evaluation Command"

    ``` sh
    # Use local pre-trained model
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/psc/data.zip
    unzip data.zip
    python psc_nn.py mode=eval eval.pretrained_model_path="Your pdparams path"
    ```

    ``` sh
    # Or use provided pre-trained model
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/psc/data.zip
    unzip data.zip
    python psc_nn.py mode=eval eval.pretrained_model_path="https://paddle-org.bj.bcebos.com/paddlescience/models/PerovskiteSolarCells/solar_cell_pretrained.pdparams"
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [solar_cell_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/PerovskiteSolarCells/solar_cell_pretrained.pdparams) | RMSE: 3.91798 |

## 1. Background Introduction

Solar cells are key energy devices that directly convert light energy into electrical energy through the photovoltaic effect. Performance prediction is an important part of optimizing and designing solar cells. However, traditional performance prediction methods often rely on complex physical simulations and a large number of experimental tests, which are not only costly but also time-consuming, restricting the efficiency of research and development.

In recent years, the rapid development of deep learning and machine learning technologies has provided innovative methods for solar cell performance prediction. Through machine learning technology, development speed can be significantly accelerated while achieving prediction accuracy comparable to experimental results. Especially in the research of perovskite solar cells, the chemical composition and structural diversity of materials bring new challenges to model training. To solve this problem, researchers usually convert material properties into fixed-length feature vectors to adapt to machine learning models. Nevertheless, the feature representation design for different performance indicators still needs continuous optimization, and the interpretability requirements for model prediction results are also stricter.

In this study, by utilizing a comprehensive database (PDP) containing information on the properties of perovskite solar cells, we constructed and evaluated a variety of machine learning models including XGBoost and psc_nn, focusing on predicting short-circuit current density (Jsc). The results show that combining deep learning with hyperparameter optimization tools (such as Optuna) can significantly improve the efficiency of solar cell design, providing a more accurate and efficient solution for the research and development of new solar cells.

## 2. Model Principle

This chapter only briefly introduces the principle of the solar cell performance prediction model. For detailed theoretical derivation, please read [Machine Learning for Perovskite Solar Cells: An Open-Source Pipeline](https://onlinelibrary.wiley.com/doi/10.1002/apxr.202400060).

The main idea of this method is to establish a nonlinear mapping relationship between spectral response data and short-circuit current density (Jsc) through an artificial neural network. The overall structure of the artificial neural network model is shown in the figure below:

![psc_nn_overview](https://paddle-org.bj.bcebos.com/paddlescience/docs/psc_nn/psc_nn_overview.png)

This case uses a Multi-Layer Perceptron (MLP) as the basic model architecture, mainly including the following parts:

1. Input layer: Receives 2808-dimensional spectral response data
2. Hidden layer: 4-6 fully connected layers, the number of neurons in each layer is optimized by Optuna
3. Activation function: Uses ReLU activation function to introduce nonlinear characteristics
4. Output layer: Outputs the predicted Jsc value

In this way, we can automatically find the model configuration best suited for the current task and improve the prediction performance of the model.

## 3. Model Implementation

In this chapter, we explain how to implement the perovskite solar cell performance prediction model based on PaddleScience code. This case combines the Optuna framework for hyperparameter optimization and uses various built-in functional modules of PaddleScience. In order to quickly understand PaddleScience, only key steps such as model construction, constraint construction, and validator construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Dataset Introduction

The dataset used in this case contains [Perovskite Database Project (PDP) data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/psc/data.zip). The dataset is divided into the following parts:

1. Training set:
   - Feature data: `data/cleaned/training.csv`
   - Label data: `data/cleaned/training_labels.csv`
2. Validation set:
   - Feature data: `data/cleaned/validation.csv`
   - Label data: `data/cleaned/validation_labels.csv`

To facilitate data processing, we implemented a helper function `create_tensor_dict` to create a tensor dictionary of inputs and labels:

``` py linenums="84" title="examples/perovskite_solar_cells/psc_nn.py"
--8<--
examples/perovskite_solar_cells/psc_nn.py:84:89
--8<--
```

The data reading and preprocessing code is as follows:

``` py linenums="172" title="examples/perovskite_solar_cells/psc_nn.py"
--8<--
examples/perovskite_solar_cells/psc_nn.py:172:191
--8<--
```

For hyperparameter optimization, we further divide the training set into training set and validation set:

``` py linenums="185" title="examples/perovskite_solar_cells/psc_nn.py"
--8<--
examples/perovskite_solar_cells/psc_nn.py:185:187
--8<--
```

### 3.2 Model Construction

This case uses `ppsci.arch.MLP` built in PaddleScience to build a multi-layer perceptron model. The hyperparameters of the model are optimized through the Optuna framework, mainly including:

1. Number of network layers: 4-6 layers
2. Number of neurons per layer: 10-input_dim/2
3. Activation function: ReLU
4. Input dimension: 2808 (spectral response data dimension)
5. Output dimension: 1 (Jsc predicted value)

The model definition code is as follows:

``` py linenums="152" title="examples/perovskite_solar_cells/psc_nn.py"
--8<--
examples/perovskite_solar_cells/psc_nn.py:152:168
--8<--
```

### 3.3 Loss Function Design

Considering that different samples in the dataset may have different importance, we designed a weighted mean square error loss function. This function assigns higher weight to larger Jsc values to improve the prediction accuracy of the model on high-performance solar cells:

``` py linenums="72" title="examples/perovskite_solar_cells/psc_nn.py"
--8<--
examples/perovskite_solar_cells/psc_nn.py:72:81
--8<--
```

### 3.4 Constraint Construction

This case solves the problem based on data-driven methods, so `SupervisedConstraint` built in PaddleScience is used to construct supervised constraints. To reduce code duplication, we implemented the `create_constraint` function to create supervised constraints:

``` py linenums="92" title="examples/perovskite_solar_cells/psc_nn.py"
--8<--
examples/perovskite_solar_cells/psc_nn.py:92:111
--8<--
```

### 3.5 Validator Construction

In order to monitor the training situation of the model in real time, we implemented the `create_validator` function to create a validator:

``` py linenums="114" title="examples/perovskite_solar_cells/psc_nn.py"
--8<--
examples/perovskite_solar_cells/psc_nn.py:114:129
--8<--
```

### 3.6 Optimizer Construction

In order to unify the management of the creation of optimizer and learning rate scheduler, we implemented the `create_optimizer` function:

``` py linenums="132" title="examples/perovskite_solar_cells/psc_nn.py"
--8<--
examples/perovskite_solar_cells/psc_nn.py:132:149
--8<--
```

### 3.7 Model Training and Evaluation

During the training process, we use the functions encapsulated above to create data dictionaries, constraints, validators and optimizers:

``` py linenums="258" title="examples/perovskite_solar_cells/psc_nn.py"
--8<--
examples/perovskite_solar_cells/psc_nn.py:258:262
--8<--
```

## 4. Complete Code

``` py linenums="1" title="examples/perovskite_solar_cells/psc_nn.py"
--8<--
examples/perovskite_solar_cells/psc_nn.py
--8<--
```

## 5. References

- [Machine Learning for Perovskite Solar Cells: An Open-Source Pipeline](https://onlinelibrary.wiley.com/doi/10.1002/apxr.202400060)
