# Combining Differentiable PDE Solvers and Graph Neural Networks for Fluid Flow Prediction

<a href="https://aistudio.baidu.com/projectdetail/7127446" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

=== "Model Training Command"

    ``` sh
    # only linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/CFDGCN/data.zip
    unzip data.zip
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/CFDGCN/meshes.tar
    tar -xvf meshes.tar
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/CFDGCN/SU2Bin.tgz
    tar -zxvf SU2Bin.tgz

    # set BATCH_SIZE = number of cpu cores
    export BATCH_SIZE=4

    # prediction experiments
    mpirun -np $((BATCH_SIZE+1)) python cfdgcn.py \
      TRAIN.batch_size=$((BATCH_SIZE)) > /dev/null

    # generalization experiments
    mpirun -np $((BATCH_SIZE+1)) python cfdgcn.py \
      TRAIN.batch_size=$((BATCH_SIZE)) \
      TRAIN_DATA_DIR="./data/NACA0012_machsplit_noshock/outputs_train" \
      TRAIN_MESH_GRAPH_PATH="./data/NACA0012_machsplit_noshock/mesh_fine. su2" \
      EVAL_DATA_DIR="./data/NACA0012_machsplit_noshock/outputs_test" \
      EVAL_MESH_GRAPH_PATH="./data/NACA0012_machsplit_noshock/mesh_fine.su2" \
      > /dev/null
    ```

## 1. Background Introduction

In recent years, the successful application of deep learning in computer vision and natural language processing has prompted people to explore the application of artificial intelligence in the field of scientific computing, especially in the field of Computational Fluid Dynamics (CFD).

Fluid is a very complex physical system, and the behavior of fluid is governed by the Navier-Stokes equations. Grid-based finite volume or finite element simulation methods are widely used numerical methods in CFD. The physical problems studied by computational fluid dynamics are often very complex and usually require a lot of computing resources to find the solution to the problem, so a trade-off between solution accuracy and computational cost is needed. In order to perform numerical simulation, the computational domain is usually discretized by grids. Since the grid has good geometric and physical problem representation capabilities and is compatible with the graph structure, the authors of this article use graph neural networks to construct a data-driven model for flow field prediction by training CFD simulation data.

## 2. Problem Definition

The authors propose a graph neural network-based CFD calculation model called CFD-GCN (Computational fluid dynamics - Graph convolution network). This model is a hybrid graph neural network that combines traditional graph convolution networks with coarse-resolution CFD simulators. It can not only greatly accelerate CFD prediction but also generalize well to new scenarios. At the same time, the prediction effect of the model is far better than the simulation effect of coarse-resolution CFD alone.

The figure below shows the network structure of this method. The network has two main components: GCN graph neural network and SU2 fluid simulator. The network operates on two different graphs, which are the graph of the fine grid and the graph of the coarse grid. The network first runs a CFD simulation on the coarse grid while processing the graph of the fine grid using GCN. Then, the simulation results are upsampled and concatenated with the intermediate output of GCN. Finally, the model applies additional GCN layers to these concatenated features to predict the desired output values.

![CFDGCN_overview](https://ai-studio-static-online.cdn.bcebos.com/d3c10c571f68481888cbe212b5019fce9806ef52f8bc4eeeb4c2349c6072fd4a)

## 3. Problem Solving

Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods.
In order to quickly understand PaddleScience, only key steps such as model construction, equation construction, and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

!!! info "Note"

    Before running this case, you need to install [**P**addle **G**raph **L**earning](https://github.com/PaddlePaddle/PGL) graph learning tool and [Mpi4py](https://github.com/pyamg/pyamg) MPI python interface library via `pip install pgl==2.2.6 mpi4py` command.

    Since the new version of Paddle relies on a higher python version, the installation of `pgl` and `mpi4py` may have problems. It is recommended to use [AI Studio Quick Experience](https://aistudio.baidu.com/projectdetail/7127446), where the running environment has been configured in the project.

### 3.1 Dataset Download

The airfoil dataset used in this case comes from de Avila Belbute-Peres et al., where the airfoil dataset uses NACA0012 airfoil, including train, test and corresponding grid data mesh_fine; the cylinder dataset is a CFD calculation example calculated by the original author using software.

Execute the following command to download and unzip the dataset.

``` sh
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/CFDGCN/data.zip
unzip data.zip
wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/CFDGCN/meshes.tar
tar -xvf meshes.tar
```

### 3.2 SU2 Precompiled Library Installation

The SU2 version of this case is too low (v6.2.0), so specific versions of openmpi and mpi4py need to be installed (openmpi 1.10.2 and mpi4py 3.1.4).

The SU2 fluid simulator is embedded in the network in the form of a precompiled library. We need to download and set environment variables.

Execute the following command to download and unzip the precompiled library.

``` sh
wget -c -P https://paddle-org.bj.bcebos.com/paddlescience/datasets/CFDGCN/SU2Bin.tgz
tar -zxvf SU2Bin.tgz
```

After the precompiled library is downloaded, set the environment variables of SU2.

``` sh
export SU2_RUN=/absolute_path/to/SU2Bin/
export SU2_HOME=/absolute_path/to/SU2Bin/
export PATH=$PATH:$SU2_RUN
export PYTHONPATH=$PYTHONPATH:$SU2_RUN
```

### 3.3 Model Construction

In this problem, we use the neural network `CFDGCN` as the model, which receives graph structure data and outputs prediction results.

``` py linenums="77"
--8<--
examples/cfdgcn/cfdgcn.py:77:82
--8<--
```

In order to access the value of specific variables accurately and quickly during calculation, we specify the input variable name of the network model as `("input", )` and the output variable name as `("pred", )`, these names are consistent with the subsequent code.

### 3.4 Constraint Construction

In this case, we use supervised datasets to train the model, so we need to build supervised constraints.

Before defining constraints, we need to specify the path of the dataset and other related configurations, and store this information in the corresponding YAML file, as shown below.

``` yaml linenums="28"
--8<--
examples/cfdgcn/conf/cfdgcn.yaml:28:34
--8<--
```

Then define the calculation process of training loss function, as shown below.

``` py linenums="31"
--8<--
examples/cfdgcn/cfdgcn.py:31:36
--8<--
```

Finally construct supervised constraints, as shown below.

``` py linenums="58"
--8<--
examples/cfdgcn/cfdgcn.py:58:84
--8<--
```

### 3.5 Hyperparameter Setting

Set parameters such as training rounds, as shown below.

``` yaml linenums="50"
--8<--
examples/cfdgcn/conf/cfdgcn.yaml:50:56
--8<--
```

### 3.6 Optimizer Construction

The training process will call the optimizer to update model parameters. Here, the `Adam` optimizer is selected, and a fixed `5e-4` is used as the learning rate.

``` py linenums="96"
--8<--
examples/cfdgcn/cfdgcn.py:96:97
--8<--
```

### 3.7 Validator Construction

Usually during the training process, the training status of the current model is evaluated using the validation set (test set) at a certain epoch interval, so `ppsci.validate.SupervisedValidator` is used to construct the validator. The construction process is similar to [Constraint Construction](#34), just change the data directory to the directory of the test set, and set `EVAL.batch_size=1` in the configuration file.

``` py linenums="100"
--8<--
examples/cfdgcn/cfdgcn.py:100:123
--8<--
```

The evaluation metric is the RMSE value of the predicted result and the real result, so a custom metric calculation function needs to be defined, as shown below.

``` py linenums="39"
--8<--
examples/cfdgcn/cfdgcn.py:39:48
--8<--
```

The evaluation metric is the RMSE value of the predicted result and the real result, so a custom metric calculation function needs to be defined, as shown below.

### 3.8 Model Training

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training.

``` py linenums="125"
--8<--
examples/cfdgcn/cfdgcn.py:125:140
--8<--
```

### 3.9 Result Visualization

After training, the program will predict the data in the test set and visualize the results in the form of images, as shown below.

``` py linenums="145"
--8<--
examples/cfdgcn/cfdgcn.py:145:157
--8<--
```

## 4. Complete Code

``` py linenums="1" title="cfdgcn.py"
--8<--
examples/cfdgcn/cfdgcn.py
--8<--
```

## 5. Result Display

The following shows the prediction results and reference results of the model for pressure $p(x,y)$, x (horizontal) direction velocity $u(x,y)$, and y (vertical) direction velocity $v(x,y)$ at each point in the computational domain.

=== "Prediction Experiment"

    <figure markdown>
        ![Airfoil_0_vec_x](https://ai-studio-static-online.cdn.bcebos.com/e8670d7f82124b5cbab784a6c182f19ed4d892ee95c54127879a37021dbc518d){ loading=lazy }
        <figcaption>Left: Predicted x-direction velocity p, Right: Actual x-direction velocity</figcaption>
        ![Airfoil_0_p](https://ai-studio-static-online.cdn.bcebos.com/4cbf4b4b35a54d629e9d19dbfe250a215f1c72cf25454769be81b4f9c2132577){ loading=lazy }
        <figcaption>Left: Predicted pressure p, Right: Actual pressure p</figcaption>
        ![Airfoil_0_vec_y](https://ai-studio-static-online.cdn.bcebos.com/41241506b6824de39a65a9ff2071b2b2aa425407d9d445b98cc6e0b35e0f6fcd){ loading=lazy }
        <figcaption>Left: Predicted y-direction velocity p, Right: Actual y-direction velocity</figcaption>
    </figure>

=== "Generalization Experiment"

    <figure markdown>
        ![Airfoil_0_vec_x](https://ai-studio-static-online.cdn.bcebos.com/b2f0755b34904c31a16136a2124c275f9e98734a824c4b38a87ade94e6f3f4d6){ loading=lazy }
        <figcaption>Left: Predicted x-direction velocity p, Right: Actual x-direction velocity</figcaption>
        ![Airfoil_0_p](https://ai-studio-static-online.cdn.bcebos.com/830e8908abe74380b6b438f4cf51cd4a2c16e96d330e4afc884ec6502e00a387){ loading=lazy }
        <figcaption>Left: Predicted pressure p, Right: Actual pressure p</figcaption>
        ![Airfoil_0_vec_y](https://ai-studio-static-online.cdn.bcebos.com/e7b585aaf4cd48eea6f48c907437e75fd7811a8bbc08441d858f8bf982ab1607){ loading=lazy }
        <figcaption>Left: Predicted y-direction velocity p, Right: Actual y-direction velocity</figcaption>
    </figure>

It can be seen that the model prediction results are basically consistent with the real results, and the model generalization effect is good.

## 6. References

* [Combining Differentiable PDE Solvers and Graph Neural Networks for Fluid Flow Prediction](https://arxiv.org/abs/2007.04439)
* [locuslab/cfd-gcnCFDGCN](https://github.com/locuslab/cfd-gcn)
* [CFDGCN - AIStudio](https://aistudio.baidu.com/projectdetail/5216848)
