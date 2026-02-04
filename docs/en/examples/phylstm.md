# PhyLSTM

=== "Model Training Command"

    === "phylstm2"

        ``` sh
        # linux
        wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat -o data_boucwen.mat
        python phylstm2.py
        ```

    === "phylstm3"

        ``` sh
        # linux
        wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat -o data_boucwen.mat
        python phylstm3.py
        ```

=== "Model Evaluation Command"

    === "phylstm2"

        ``` sh
        # linux
        wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat -o data_boucwen.mat
        python phylstm2.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/phylstm/phylstm2_pretrained.pdparams
        ```

    === "phylstm3"

        ``` sh
        # linux
        wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat -o data_boucwen.mat
        python phylstm3.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/phylstm/phylstm3_pretrained.pdparams
        ```

| Pretrained Model | Metrics |
|:--| :--|
| [phylstm2_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/phylstm/phylstm2_pretrained.pdparams) | loss(sup_valid): 0.00799 |
| [phylstm3_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/phylstm/phylstm3_pretrained.pdparams) | loss(sup_valid): 0.03098 |

## 1. Background Introduction

We introduce an innovative physics-informed LSTM framework for metamodeling of nonlinear structural systems lacking data. The basic concept is to integrate available but incomplete physical knowledge (such as physical laws, scientific principles) into a deep Long Short-Term Memory (LSTM) network, which restricts and facilitates learning within a feasible solution space. Physical constraints are embedded in the loss function to enforce model training to accurately capture potential system non-linearities even when available training datasets are very limited. Especially for dynamic structures, physical laws of equations of motion, state dependence, and hysteresis constitutive relationships are considered to construct physical losses. Embedded physics can alleviate overfitting problems, reduce the need for large training datasets, and improve the robustness of trained models, making them capable of extrapolation, thereby making more reliable predictions. Therefore, the physics-knowledge-guided deep learning paradigm outperforms traditional non-physics-guided data-driven neural networks.

## 2. Problem Definition

Metamodeling of structural systems aims to develop low-fidelity (or low-order) models to effectively capture potential nonlinear input-output behaviors. Metamodels can be trained on datasets obtained from high-fidelity simulations or actual system sensing. To illustrate better, we consider a building-type structure and assume that seismic dynamics are governed by low-fidelity nonlinear equations of motion (EOM):

$$
\mathbf{M} \ddot{\mathbf{u}}+\underbrace{\mathbf{C} \dot{\mathbf{u}}+\lambda \mathbf{K u}+(1-\lambda) \mathbf{K r}}_{\mathbf{h}}=-\mathbf{M} \Gamma a_g
$$

Where M is the mass matrix; C is the damping matrix; K is the stiffness matrix.

The governing equation can be rewritten in a more general form:

$$
\ddot{\mathbf{u}}+\mathrm{g}=-\Gamma a_g
$$

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Model Construction

In the PhyLSTM problem, establish an LSTM network Deep LSTM network, expressed in PaddleScience code as follows

``` py linenums="102"
--8<--
examples/phylstm/phylstm2.py:102:107
--8<--
```

DeepPhyLSTM parameters input_size is input size, output_size is output size, hidden_size is hidden layer size, model_type is model type.

### 3.2 Data Construction

Before running the code for this problem, please download [data_boucwen.mat](https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat) according to the following command.

``` sh
wget -c -P ./ https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat
```

This case involves reading data construction, as shown below

``` py linenums="37"
--8<--
examples/phylstm/phylstm2.py:37:100
--8<--
```

### 3.3 Constraint Construction

Set training dataset and loss calculation function, return fields, code is as follows:

``` py linenums="120"
--8<--
examples/phylstm/phylstm2.py:120:146
--8<--
```

### 3.4 Validator Construction

Set evaluation dataset and loss calculation function, return fields, code is as follows:

``` py linenums="148"
--8<--
examples/phylstm/phylstm2.py:148:170
--8<--
```

### 3.5 Hyperparameter Setting

Next, we need to specify the number of training epochs. Here we use 100 training epochs based on experimental experience.

``` yaml linenums="38"
--8<--
examples/phylstm/conf/phylstm2.yaml:38:40
--8<--
```

### 3.6 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the `Adam` optimizer is selected and `learning_rate` is set to 1e-3.

``` py linenums="172"
--8<--
examples/phylstm/phylstm2.py:172:173
--8<--
```

### 3.7 Model Training and Evaluation

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order.

``` py linenums="174"
--8<--
examples/phylstm/phylstm2.py:174:180
--8<--
```

Finally, start training and evaluation:

``` py linenums="182"
--8<--
examples/phylstm/phylstm2.py:182:185
--8<--
```

## 4. Complete Code

=== "phylstm2"

    ``` py linenums="1" title="phylstm2.py"
    --8<--
    examples/phylstm/phylstm2.py
    --8<--
    ```

=== "phylstm3"

    ``` py linenums="1" title="phylstm3.py"
    --8<--
    examples/phylstm/phylstm3.py
    --8<--
    ```

## 5. Result Display

The PhyLSTM2 case was experimented with the parameter configuration of epoch=100 and learning\_rate=1e-3, and the result returned Loss was 0.00799.

The PhyLSTM3 case was experimented with the parameter configuration of epoch=200 and learning\_rate=1e-3, and the result returned Loss was 0.03098.

## 6. References

- [Physics-informed multi-LSTM networks for metamodeling of nonlinear structures](https://www.sciencedirect.com/science/article/abs/pii/S0045782520304114)
- [https://github.com/zhry10/PhyLSTM.git](https://github.com/zhry10/PhyLSTM.git)
