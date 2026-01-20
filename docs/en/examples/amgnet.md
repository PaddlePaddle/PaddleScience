# AMGNet

<!-- <a href="https://aistudio.baidu.com/aistudio/projectdetail/6184070?contributionType=1&sUid=438690&shared=1&ts=1684239806160" class="md-button md-button--primary" style>AI Studio Quick Experience</a> -->

!!! info "Note"

    Before running this case, you need to install [**P**addle **G**raph **L**earning](https://github.com/PaddlePaddle/PGL) graph learning tool and [PyAMG](https://github.com/pyamg/pyamg) algebraic multigrid tool via `pip install -r requirements.txt` command.

=== "Model Training Command"

    === "amgnet_airfoil"

        ``` sh
        # linux
        wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/AMGNet/data.zip
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/AMGNet/data.zip -o data.zip
        # unzip it
        unzip data.zip
        python amgnet_airfoil.py
        ```
    === "amgnet_cylinder"

        ``` sh
        # linux
        wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/AMGNet/data.zip
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/AMGNet/data.zip -o data.zip
        # unzip it
        unzip data.zip
        python amgnet_cylinder.py
        ```

=== "Model Evaluation Command"

    === "amgnet_airfoil"

        ``` sh
        # linux
        wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/AMGNet/data.zip
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/AMGNet/data.zip -o data.zip
        # unzip it
        unzip data.zip
        python amgnet_airfoil.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/amgnet/amgnet_airfoil_pretrained.pdparams
        ```
    === "amgnet_cylinder"

        ``` sh
        # linux
        wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/AMGNet/data.zip
        # windows
        # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/AMGNet/data.zip -o data.zip
        # unzip it
        unzip data.zip
        python amgnet_cylinder.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/amgnet/amgnet_cylinder_pretrained.pdparams
        ```

| Pretrained Model | Metrics |
|:--| :--|
| [amgnet_airfoil_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/amgnet/amgnet_airfoil_pretrained.pdparams) | loss(RMSE_validator): 0.0001 <br> RMSE.RMSE(RMSE_validator): 0.01315 |
| [amgnet_cylinder_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/amgnet/amgnet_cylinder_pretrained.pdparams) | loss(RMSE_validator): 0.00048 <br> RMSE.RMSE(RMSE_validator): 0.02197 |

## 1. Background Introduction

In recent years, the successful application of deep learning in computer vision and natural language processing has prompted people to explore the application of artificial intelligence in the field of scientific computing, especially in the field of Computational Fluid Dynamics (CFD).

Fluid is a very complex physical system, and the behavior of fluid is governed by the Navier-Stokes equations. Grid-based finite volume or finite element simulation methods are widely used numerical methods in CFD. The physical problems studied by computational fluid dynamics are often very complex and usually require a lot of computing resources to find the solution to the problem, so a trade-off between solution accuracy and computational cost is needed. In order to perform numerical simulation, the computational domain is usually discretized by grids. Since the grid has good geometric and physical problem representation capabilities and is compatible with the graph structure, the authors of this article use graph neural networks to construct a data-driven model for flow field prediction by training CFD simulation data.

## 2. Problem Definition

The authors propose a graph neural network-based CFD calculation model called AMGNET (A Multi-scale Graph neural Network), which can predict flow fields under different physical parameters. This method has the following characteristics:

- AMGNET converts the grid in CFD into a graph structure and processes and aggregates information through graph neural networks. Compared with traditional GCN methods, the prediction error of this method is significantly lower.

- AMGNET can calculate the fluid velocity in the x and y directions at the same time, and can also calculate the fluid pressure.

- AMGNET coarsens the graph through the RS algorithm (Olson and Schroder, 2018), and can predict using only a small number of nodes, further improving the prediction speed.

The figure below shows the network structure of this method. The basic principle of this model is to convert the grid structure into a graph structure, and then encode the nodes and edges in the graph through the physical information, location information and node type of the nodes in the grid. Then, the obtained graph neural network is coarsened using a coarsening layer based on the algebraic multigrid algorithm (RS) to classify all nodes into coarse node sets and fine node sets, where the coarse node set is a subset of the fine node set. The node set of the coarse graph is the coarse node set, thus completing the coarsening of the graph and reducing the scale of the graph. After coarsening is completed, the features of the graph are summarized and extracted through the designed graph neural network message passing block (GN). Afterwards, the graph restoration layer uses reverse operations and uses spatial interpolation (Qi et al., 2017) to upsample the graph. For example, to interpolate node $i$, find the $k$ nodes closest to node $i$ in the coarse graph, and then calculate the features of node $i$ through the formula. Finally, the velocity and pressure information of each node is obtained through the decoder.

![AMGNet_overview](https://paddle-org.bj.bcebos.com/paddlescience/docs/AMGNet/amgnet.png)

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Dataset Download

The airfoil dataset used in this case comes from de Avila Belbute-Peres et al., where the airfoil dataset uses NACA0012 airfoil, including train, test and corresponding grid data mesh_fine; the cylinder dataset is a CFD calculation example calculated by the original author using software.

Execute the following command to download and unzip the dataset.

``` sh
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/AMGNet/data.zip
unzip data.zip
```

### 3.2 Model Construction

In this problem, we use the graph neural network `AMGNet` as the model, which receives graph structure data and outputs prediction results.

=== "airfoil"

    ``` py linenums="61"
    --8<--
    examples/amgnet/amgnet_airfoil.py:61:62
    --8<--
    ```

=== "cylinder"

    ``` py linenums="61"
    --8<--
    examples/amgnet/amgnet_cylinder.py:61:62
    --8<--
    ```

In order to access the value of specific variables accurately and quickly during calculation, we specify the input variable name of the network model as `("input", )` and the output variable name as `("pred", )`, these names are consistent with the subsequent code.

### 3.3 Constraint Construction

In this case, we use supervised datasets to train the model, so we need to build supervised constraints.

Before defining constraints, we need to specify the path of the dataset and other related configurations, and store this information in the corresponding YAML file, as shown below.

=== "airfoil"

    ``` yaml linenums="21"
    --8<--
    examples/amgnet/conf/amgnet_airfoil.yaml:21:27
    --8<--
    ```

=== "cylinder"

    ``` yaml linenums="21"
    --8<--
    examples/amgnet/conf/amgnet_cylinder.yaml:21:27
    --8<--
    ```

Then define the calculation process of training loss function, as shown below.

=== "airfoil"

    ``` py linenums="35"
    --8<--
    examples/amgnet/amgnet_airfoil.py:35:40
    --8<--
    ```

=== "cylinder"

    ``` py linenums="35"
    --8<--
    examples/amgnet/amgnet_cylinder.py:35:40
    --8<--
    ```

Finally construct supervised constraints, as shown below.

=== "airfoil"

    ``` py linenums="82"
    --8<--
    examples/amgnet/amgnet_airfoil.py:82:90
    --8<--
    ```

=== "cylinder"

    ``` py linenums="82"
    --8<--
    examples/amgnet/amgnet_cylinder.py:82:90
    --8<--
    ```

### 3.4 Hyperparameter Setting

Set parameters such as training rounds, as shown below.

=== "airfoil"

    ``` yaml linenums="50"
    --8<--
    examples/amgnet/conf/amgnet_airfoil.yaml:50:52
    --8<--
    ```

=== "cylinder"

    ``` yaml linenums="50"
    --8<--
    examples/amgnet/conf/amgnet_cylinder.yaml:50:52
    --8<--
    ```

### 3.5 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the `Adam` optimizer is selected, and a fixed `5e-4` is used as the learning rate.

=== "airfoil"

    ``` py linenums="92"
    --8<--
    examples/amgnet/amgnet_airfoil.py:92:93
    --8<--
    ```

=== "cylinder"

    ``` py linenums="92"
    --8<--
    examples/amgnet/amgnet_cylinder.py:92:93
    --8<--
    ```

### 3.6 Validator Construction

Usually during the training process, the training status of the current model is evaluated using the validation set (test set) at a certain epoch interval, so `ppsci.validate.SupervisedValidator` is used to construct the validator. The construction process is similar to [Constraint Construction](#33), just change the data directory to the directory of the test set, and set `EVAL.batch_size=1` in the configuration file.

=== "airfoil"

    ``` py linenums="95"
    --8<--
    examples/amgnet/amgnet_airfoil.py:95:118
    --8<--
    ```
=== "cylinder"

    ``` py linenums="95"
    --8<--
    examples/amgnet/amgnet_cylinder.py:95:118
    --8<--
    ```

The evaluation metric is the RMSE value of the predicted result and the real result, so a custom metric calculation function needs to be defined, as shown below.

=== "airfoil"

    ``` py linenums="43"
    --8<--
    examples/amgnet/amgnet_airfoil.py:43:52
    --8<--
    ```
=== "cylinder"

    ``` py linenums="43"
    --8<--
    examples/amgnet/amgnet_cylinder.py:43:52
    --8<--
    ```

### 3.7 Model Training

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training.

=== "airfoil"

    ``` py linenums="120"
    --8<--
    examples/amgnet/amgnet_airfoil.py:120:136
    --8<--
    ```
=== "cylinder"

    ``` py linenums="120"
    --8<--
    examples/amgnet/amgnet_cylinder.py:120:136
    --8<--
    ```

### 3.8 Result Visualization

After training, the program will predict the data in the test set and visualize the results in the form of images, as shown below.

=== "airfoil"

    ``` py linenums="138"
    --8<--
    examples/amgnet/amgnet_airfoil.py:138:151
    --8<--
    ```
=== "cylinder"

    ``` py linenums="138"
    --8<--
    examples/amgnet/amgnet_cylinder.py:138:151
    --8<--
    ```

## 4. Complete Code

=== "airfoil"

    ``` py linenums="1" title="amgnet_airfoil.py"
    --8<--
    examples/amgnet/amgnet_airfoil.py
    --8<--
    ```
=== "cylinder"

    ``` py linenums="1" title="amgnet_airfoil.py"
    --8<--
    examples/amgnet/amgnet_cylinder.py
    --8<--
    ```

## 5. Result Display

The following shows the prediction results and reference results of the model for pressure $p(x,y)$, x (horizontal) direction velocity $u(x,y)$, and y (vertical) direction velocity $v(x,y)$ at each point in the computational domain.

=== "airfoil"

    <figure markdown>
        ![Airfoil_0_vec_x](https://paddle-org.bj.bcebos.com/paddlescience/docs/AMGNet/airfoil_0field.png0_field.png){ loading=lazy }
        <figcaption>Left: Predicted x-direction velocity p, Right: Actual x-direction velocity</figcaption>
        ![Airfoil_0_p](https://paddle-org.bj.bcebos.com/paddlescience/docs/AMGNet/airfoil_0field.png1_field.png){ loading=lazy }
        <figcaption>Left: Predicted pressure p, Right: Actual pressure p</figcaption>
        ![Airfoil_0_vec_y](https://paddle-org.bj.bcebos.com/paddlescience/docs/AMGNet/airfoil_0field.png2_field.png){ loading=lazy }
        <figcaption>Left: Predicted y-direction velocity p, Right: Actual y-direction velocity</figcaption>
    </figure>

=== "cylinder"

    <figure markdown>
        ![Cylinder_0_vec_x](https://paddle-org.bj.bcebos.com/paddlescience/docs/AMGNet/cylinder_0field.png0_field.png){ loading=lazy }
        <figcaption>Left: Predicted x-direction velocity p, Right: Actual x-direction velocity</figcaption>
        ![Cylinder_0_p](https://paddle-org.bj.bcebos.com/paddlescience/docs/AMGNet/cylinder_0field.png1_field.png){ loading=lazy }
        <figcaption>Left: Predicted pressure p, Right: Actual pressure p</figcaption>
        ![Cylinder_0_vec_y](https://paddle-org.bj.bcebos.com/paddlescience/docs/AMGNet/cylinder_0field.png2_field.png){ loading=lazy }
        <figcaption>Left: Predicted y-direction velocity p, Right: Actual y-direction velocity</figcaption>
    </figure>

It can be seen that the model prediction results are basically consistent with the real results.

## 6. References

- [AMGNET: multi-scale graph neural networks for flow field prediction](https://doi.org/10.1080/09540091.2022.2131737)
- [AMGNet - Github](https://github.com/baoshiaijhin/amgnet)
- [AMGNet - AIStudio](https://aistudio.baidu.com/projectdetail/5592458)
