# PhyGeoNet

<a href="https://aistudio.baidu.com/projectdetail/7195983" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

=== "Model Training Command"

    ``` sh

    # heat_equation
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation.npz -P ./data/

    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation.npz --create-dirs -o ./data/heat_equation.npz

    python heat_equation.py

    # heat_equation_bc
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation_bc.npz -P ./data/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation_bc_test.npz -P ./data/

    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation_bc.npz --create-dirs -o ./data/heat_equation.npz
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation_bc_test.npz --create-dirs -o ./data/heat_equation.npz

    python heat_equation_with_bc.py
    ```

=== "Model Evaluation Command"

    ``` sh

    # heat_equation
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation.npz -P ./data/

    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation.npz --create-dirs -o ./data/heat_equation.npz

    python heat_equation.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/PhyGeoNet/heat_equation_pretrain.pdparams

    # heat_equation_bc
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation_bc.npz -P ./data/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation_bc_test.npz -P ./data/

    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation_bc.npz --create-dirs -o ./data/heat_equation.npz
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation_bc_test.npz --create-dirs -o ./data/heat_equation.npz

    python heat_equation_with_bc.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/PhyGeoNet/heat_equation_bc_pretrain.pdparams

    ```

=== "Model Export Command"

    ``` sh
    # heat_equation
    python heat_equation.py mode=export

    # heat_equation_bc
    python heat_equation_with_bc.py mode=export
    ```

=== "Model Inference Command"

    ``` sh
    # heat_equation
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation.npz -P ./data/

    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation.npz --create-dirs -o ./data/heat_equation.npz

    python heat_equation.py mode=infer

    # heat_equation_bc
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation_bc.npz -P ./data/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation_bc_test.npz -P ./data/

    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation_bc.npz --create-dirs -o ./data/heat_equation.npz
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation_bc_test.npz --create-dirs -o ./data/heat_equation.npz

    python heat_equation_with_bc.py mode=infer
    ```

| Model | mRes | ev |
| :-- | :-- | :-- |
| [heat_equation_pretrain.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/PhyGeoNet/heat_equation_pretrain.pdparams)  | 0.815 |0.095|
| [heat_equation_bc_pretrain.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/PhyGeoNet/heat_equation_bc_pretrain.pdparams)  | 992 |0.31|

## 1. Background Introduction

In recent years, deep learning has achieved remarkable achievements in many fields, especially in computer vision and natural language processing. Inspired by the rapid development of deep learning and based on the powerful function approximation ability of deep learning, neural networks have also achieved success in the field of scientific computing. Current research is mainly divided into two categories. One is to add physical information and physical constraints to the loss function to train neural networks, represented by PINN and Deep Ritz Net. The other is data-driven deep neural network operators, represented by FNO and DeepONet. These methods have been widely used in scientific practice, such as weather forecasting, quantum chemistry, biological engineering, and computational fluid dynamics. Due to the parameter sharing nature of convolutional neural networks, they can learn large-scale spatiotemporal domains, so they have received more and more attention.

## 2. Problem Definition

In actual scientific computing problems, the solution domain of many partial differential equations has complex boundaries and is non-uniform. Existing neural networks often target solution domains with regular boundaries and uniform grids, so they have no practical application effect.

Aiming at the problem that physical information neural networks perform poorly on complex boundary non-uniform grid solution domains, this paper proposes a method to transform irregular boundary non-uniform grids into regular boundary uniform grids through coordinate transformation. In addition, this paper uses the above advantages of convolutional neural networks after becoming uniform grids, and proposes corresponding physical information convolutional neural networks.

## 3. Problem Solving

To save space, `heat equation` will be used as an example to explain how to implement it using PaddleScience.

### 3.1 Model Construction

This case uses the proposed USCNN model for training. The construction method of this model is shown below.

``` py linenums="23"
--8<--
examples/phygeonet/heat_equation.py:23:23
--8<--
```

Among them, the parameters required to build the model can be obtained from the corresponding configuration file.

``` yaml linenums="34"
--8<--
examples/phygeonet/conf/heat_equation.yaml:34:43
--8<--
```

### 3.2 Data Reading

The dataset used in this case is stored in the `.npz` file, and the following code is used to read it.

``` py linenums="15"
--8<--
examples/phygeonet/heat_equation.py:15:21
--8<--
```

### 3.3 Output Transformation Function Construction

This article is a forced boundary constraint. During training, the corresponding output transformation function is used to calculate the differential of the output result of the model.

``` py linenums="50"
--8<--
examples/phygeonet/heat_equation.py:50:79
--8<--
```

### 3.4 Constraint Construction

Construct corresponding constraint conditions. Since the boundary constraint is a forced constraint, the constraint conditions are mainly residual constraints.

``` py linenums="28"
--8<--
examples/phygeonet/heat_equation.py:28:48
--8<--
```

### 3.5 Optimizer Construction

Consistent with the description in the paper, we use a constant learning rate of 0.001 to construct the Adam optimizer.

``` py linenums="25"
--8<--
examples/phygeonet/heat_equation.py:25:25
--8<--
```

### 3.6 Model Training

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver`.

``` py linenums="82"
--8<--
examples/phygeonet/heat_equation.py:82:89
--8<--
```

Finally start training:

``` py linenums="90"
--8<--
examples/phygeonet/heat_equation.py:90:90
--8<--
```

### 3.7 Model Evaluation

After the model training is completed, the evaluate() function can be used to evaluate and visualize the trained model.

``` py linenums="94"
--8<--
examples/phygeonet/heat_equation.py:94:151
--8<--
```

## 4. Complete Code

``` py linenums="1" title="heat_equation.py"
--8<--
examples/phygeonet/heat_equation.py
--8<--
```

## 5. Result Display

Heat equation result display:
![image](https://paddle-org.bj.bcebos.com/paddlescience/docs/PhyGeoNet/heat_equation.jpg)

Heat equation with boundary result display:

T=0
![image](https://paddle-org.bj.bcebos.com/paddlescience/docs/PhyGeoNet/heat_equation_bc_1.png)

T=3
![image](https://paddle-org.bj.bcebos.com/paddlescience/docs/PhyGeoNet/heat_equation_bc_2.png)

T=6
![image](https://paddle-org.bj.bcebos.com/paddlescience/docs/PhyGeoNet/heat_equation_bc_3.png)

## 6. Summary

This paper constructs a coordinate transformation function using harmonic mapping, so that the physical information network can be trained on irregular non-uniform grids. At the same time, because the mapping is performed using traditional methods, it can be embedded before and after the network without training. Through a large number of experiments, it is shown that the network can perform better than SOAT networks on various irregular grid problems.

## 7. References

[PhyGeoNet: Physics-informed geometry-adaptive convolutional neural networks for solving parameterized steady-state PDEs on irregular domain](https://www.sciencedirect.com/science/article/pii/S0021999120308536?via%3Dihub)

[Github PhyGeoNet](https://github.com/Jianxun-Wang/phygeonet/tree/master?tab=readme-ov-file)
