# TopOpt

<a href="https://aistudio.baidu.com/projectdetail/6956236" class="md-button md-button--primary" style>AI Studio Quick Experience</a>

=== "Model Training Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/topopt/top_dataset.h5 -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/topopt/top_dataset.h5 --create-dirs -o ./datasets/top_dataset.h5
    python topopt.py
    ```

=== "Model Evaluation Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/topopt/top_dataset.h5 -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/topopt/top_dataset.h5 --create-dirs -o ./datasets/top_dataset.h5
    python topopt.py mode=eval 'EVAL.pretrained_model_path_dict={'Uniform': 'https://paddle-org.bj.bcebos.com/paddlescience/models/topopt/uniform_pretrained.pdparams', 'Poisson5': 'https://paddle-org.bj.bcebos.com/paddlescience/models/topopt/poisson5_pretrained.pdparams', 'Poisson10': 'https://paddle-org.bj.bcebos.com/paddlescience/models/topopt/poisson10_pretrained.pdparams', 'Poisson30': 'https://paddle-org.bj.bcebos.com/paddlescience/models/topopt/poisson30_pretrained.pdparams'}'
    ```

=== "Model Export Command"

    ``` sh
    python topopt.py mode=export INFER.pretrained_model_name=Uniform
    ```

    ``` sh
    python topopt.py mode=export INFER.pretrained_model_name=Poisson5
    ```

    ``` sh
    python topopt.py mode=export INFER.pretrained_model_name=Poisson10
    ```

    ``` sh
    python topopt.py mode=export INFER.pretrained_model_name=Poisson30
    ```

=== "Model Inference Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/topopt/top_dataset.h5 -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/topopt/top_dataset.h5 --create-dirs -o ./datasets/top_dataset.h5
    python topopt.py mode=infer INFER.pretrained_model_name=Uniform INFER.img_num=3
    ```

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/topopt/top_dataset.h5 -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/topopt/top_dataset.h5 --create-dirs -o ./datasets/top_dataset.h5
    python topopt.py mode=infer INFER.pretrained_model_name=Poisson5 INFER.img_num=3
    ```

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/topopt/top_dataset.h5 -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/topopt/top_dataset.h5 --create-dirs -o ./datasets/top_dataset.h5
    python topopt.py mode=infer INFER.pretrained_model_name=Poisson10 INFER.img_num=3
    ```

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/topopt/top_dataset.h5 -P ./datasets/
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/topopt/top_dataset.h5 --create-dirs -o ./datasets/top_dataset.h5
    python topopt.py mode=infer INFER.pretrained_model_name=Poisson30 INFER.img_num=3
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [topopt_uniform_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/topopt/uniform_pretrained.pdparams) | loss(sup_validator): [0.14336, 0.10211, 0.07927, 0.06433, 0.04970, 0.04612, 0.04201, 0.03566, 0.03623, 0.03314, 0.02929, 0.02857, 0.02498, 0.02517, 0.02523, 0.02618]<br>metric.Binary_Acc(sup_validator): [0.9410, 0.9673, 0.9718, 0.9727, 0.9818, 0.9824, 0.9826, 0.9845, 0.9856, 0.9892, 0.9892, 0.9907, 0.9890, 0.9916, 0.9914, 0.9922]<br>metric.IoU(sup_validator): [0.8887, 0.9367, 0.9452, 0.9468, 0.9644, 0.9655, 0.9659, 0.9695, 0.9717, 0.9787, 0.9787, 0.9816, 0.9784, 0.9835, 0.9831, 0.9845] |
| [topopt_poisson5_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/topopt/poisson5_pretrained.pdparams) | loss(sup_validator): [0.11926, 0.09162, 0.08014, 0.06390, 0.05839, 0.05264, 0.04921, 0.04737, 0.04872, 0.04564, 0.04226, 0.04267, 0.04407, 0.04172, 0.03939, 0.03927]<br>metric.Binary_Acc(sup_validator): [0.9471, 0.9619, 0.9702, 0.9742, 0.9782, 0.9801, 0.9803, 0.9825, 0.9824, 0.9837, 0.9850, 0.9850, 0.9870, 0.9863, 0.9870, 0.9872]<br>metric.IoU(sup_validator): [0.8995, 0.9267, 0.9421, 0.9497, 0.9574, 0.9610, 0.9614, 0.9657, 0.9655, 0.9679, 0.9704, 0.9704, 0.9743, 0.9730, 0.9744, 0.9747] |
| [topopt_poisson10_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/topopt/poisson10_pretrained.pdparams) | loss(sup_validator): [0.12886, 0.07201, 0.05946, 0.04622, 0.05072, 0.04178, 0.03823, 0.03677, 0.03623, 0.03029, 0.03398, 0.02978, 0.02861, 0.02946, 0.02831, 0.02817]<br>metric.Binary_Acc(sup_validator): [0.9457, 0.9703, 0.9745, 0.9798, 0.9827, 0.9845, 0.9859, 0.9870, 0.9882, 0.9880, 0.9893, 0.9899, 0.9882, 0.9899, 0.9905, 0.9904]<br>metric.IoU(sup_validator): [0.8969, 0.9424, 0.9502, 0.9604, 0.9660, 0.9696, 0.9722, 0.9743, 0.9767, 0.9762, 0.9789, 0.9800, 0.9768, 0.9801, 0.9813, 0.9810] |
| [topopt_poisson30_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/topopt/poisson30_pretrained.pdparams) | loss(sup_validator): [0.19111, 0.10081, 0.06930, 0.04631, 0.03821, 0.03441, 0.02738, 0.03040, 0.02787, 0.02385, 0.02037, 0.02065, 0.01840, 0.01896, 0.01970, 0.01676]<br>metric.Binary_Acc(sup_validator): [0.9257, 0.9595, 0.9737, 0.9832, 0.9828, 0.9883, 0.9885, 0.9892, 0.9901, 0.9916, 0.9924, 0.9925, 0.9926, 0.9929, 0.9937, 0.9936]<br>metric.IoU(sup_validator): [0.8617, 0.9221, 0.9488, 0.9670, 0.9662, 0.9769, 0.9773, 0.9786, 0.9803, 0.9833, 0.9850, 0.9853, 0.9855, 0.9860, 0.9875, 0.9873] |

## 1. Background Introduction

Topology Optimization is a mathematical method that optimizes the distribution of materials within a given design area to maximize system performance for a given set of loads, boundary conditions, and constraints. This problem is challenging because it requires the solution to be binary, that is, it should indicate whether material exists or does not exist in each part of the design area. A common example of this optimization is minimizing the elastic strain energy of an object given the total weight and boundary conditions. With the development of the automotive and aerospace industries in the 20th century, topology optimization has expanded its application to many other disciplines: such as fluid, acoustics, electromagnetics, optics, and their combinations. SIMP (Simplied Isotropic Material with Penalization) is currently a widespread, simple and efficient topology optimization solution method. It improves the convergence of binary solutions by penalizing intermediate values of material density.

## 2. Problem Definition

Topology Optimization Problem:

$$
\begin{aligned}
& \underset{\mathbf{x}}{\text{min}} \quad && c(\mathbf{u}(\mathbf{x}), \mathbf{x}) = \sum_{j=1}^{N} E_{j}(x_{j})\mathbf{u}_{j}^{\intercal}\mathbf{k}_{0}\mathbf{u}_{j} \\
& \text{s.t.} \quad && V(\mathbf{x})/V_{0} = f_{0} \\
& \quad && \mathbf{K}\mathbf{U} = \mathbf{F} \\
& \quad && x_{j} \in \{0, 1\}, \quad j = 1,...,N
\end{aligned}
$$

Where: $x_{j}$ is material distribution; $c$ is compliance; $\mathbf{u}_{j}$ is element displacement vector; $\mathbf{k}_{0}$ is element stiffness matrix for an element with unit Youngs modulu; $\mathbf{U}$, $\mathbf{F}$ are global displacement and force vectors; $\mathbf{K}$ is global stiffness matrix; $V(\mathbf{x})$, $V_{0}$ are material volume and design area volume; $f_{0}$ is pre-specified volume ratio.

## 3. Problem Solving

In actual solving of the above problem, for simplification, the last constraint condition will be changed to a continuous form: $x_{j} \in [0, 1], \quad j = 1,...,N$. The common optimization algorithm is the SIMP algorithm, which is a gradient-based iterative method and penalizes non-binary solutions: $E_{j}(x_{j}) = E_{\text{min}} + x_{j}^{p}(E_{0} - E_{\text{min}})$. We will not expand on the SIMP algorithm here. Since using the SIMP method, the solver only needs to perform the initial $N_{0}$ iterations to obtain a basic view very close to the final result, this case hopes to predict the optimization solution given after 100 iterations of SIMP by using the $N_{0}$-th initial iteration result of SIMP and its corresponding gradient information as the input of Unet.

### 3.1 Dataset Preparation

The downloaded dataset is processed synthetic data, and the format after processing is `"iters": shape = (10000, 100, 40, 40)`, `"target": shape = (10000, 1, 40, 40)`

- 10000 - Number of randomly generated problems

- 100 - SIMP iterations

- 40 - Image height

- 40 - Image width

Please store the dataset address in `./datasets/top_dataset.h5`

Generate training set: The original code uses all 10000 problems to generate training data.

``` py linenums="68"
--8<--
examples/topopt/functions.py:68:101
--8<--
```

``` py linenums="40"
--8<--
examples/topopt/topopt.py:40:48
--8<--
```

### 3.2 Model Construction

The image $I$ obtained after the $N_{0}$ initial iteration steps of SIMP can be seen as the final structure blurred. Since the image $I^*$ given by the final optimization solution does not contain information about the intermediate process, $I^*$ can be interpreted as the mask of image $I$. Thus, the optimization process $I \rightarrow I^*$ can be seen as a binary image segmentation or foreground-background segmentation process, so the Unet model is constructed for prediction. The specific network structure is shown in the figure:
![Unet](https://ai-studio-static-online.cdn.bcebos.com/7a0e54df9c9d48e5841423546e851f620e73ea917f9e4258aefc47c498bba85e)

``` py linenums="90"
--8<--
examples/topopt/topopt.py:90:91
--8<--
```

Detailed model code is in `examples/topopt/topoptmodel.py`.

### 3.3 Parameter Setting

Based on the paper and original code, the following training parameters are given:

``` yaml linenums="49"
--8<--
examples/topopt/conf/topopt.yaml:49:54
--8<--
```

``` py linenums="36"
--8<--
examples/topopt/topopt.py:36:38
--8<--
```

### 3.4 data transform

Based on the paper and original code, the following custom data transform code is given, including random horizontal or vertical flip and random 90 degree rotation, transform input and label simultaneously:

``` py linenums="102"
--8<--
examples/topopt/functions.py:102:133
--8<--
```

### 3.5 Constraint Construction

In this case, we use supervised learning method for training, so supervised constraint `SupervisedConstraint` is used, code as follows:

``` py linenums="50"
--8<--
examples/topopt/topopt.py:50:75
--8<--
```

The first parameter of `SupervisedConstraint` is the reading configuration of supervised constraint. The `"dataset"` field in the configuration represents the training dataset information used, and its various fields represent:

1. `name`: Dataset type, here `"NamedArrayDataset"` represents the `np.ndarray` type dataset read sequentially by batch;
2. `input`: Input variable dictionary: `{"input_name": input_dataset}`;
3. `label`: Label variable dictionary: `{"label_name": label_dataset}`;
4. `transforms`: Dataset preprocessing configuration, where `"FunctionalTransform"` is user-defined preprocessing method.

The `"batch_size"` field in the reading configuration represents the batch size specified during training, and the `"sampler"` field represents the relevant sampling configuration of dataloader.

The second parameter is the loss function. Here [custom loss](#381-loss) is used, and the value corresponding to $\beta$ in the loss formula is determined by `cfg.vol_coeff`.

The third parameter is the name of the constraint condition, which is convenient for subsequent indexing. Here it is named `"sup_constraint"`.

After the constraint construction is completed, encapsulate it into a dictionary with the name we just named as the keyword for subsequent access.

### 3.6 Sampler Construction

The second dimension of the original data has 100 channels, corresponding to the 100 iteration results of the SIMP algorithm. The goal of this case model is to directly predict the final optimization solution result after 100 steps of iteration of the SIMP algorithm using the iteration result of a certain step in the middle of SIMP. Here, a channel sampler needs to be constructed to randomly extract a channel or directly specify a channel from the second dimension of the input model data according to a certain probability distribution, and then input it into the network for training or inference. This case puts the sampling step into the forward method of the model.

``` py linenums="23"
--8<--
examples/topopt/functions.py:23:67
--8<--
```

``` py linenums="80"
--8<--
examples/topopt/topopt.py:80:81
--8<--
```

### 3.7 Optimizer Construction

The training process will call the optimizer to update model parameters. Here `Adam` optimizer is selected.

``` py linenums="93"
--8<--
examples/topopt/topopt.py:93:96
--8<--
```

### 3.8 Loss and Metric Construction

#### 3.8.1 Loss Construction

Loss function is confidence loss + beta * volume fraction constraints:

$$
\mathcal{L} = \mathcal{L}_{\text{conf}}(X_{\text{true}}, X_{\text{pred}}) + \beta * \mathcal{L}_{\text{vol}}(X_{\text{true}}, X_{\text{pred}})
$$

confidence loss is binary cross-entropy:

$$
\mathcal{L}_{\text{conf}}(X_{\text{true}}, X_{\text{pred}}) = -\frac{1}{NM}\sum_{i=1}^{N}\sum_{j=1}^{M}\left[X_{\text{true}}^{ij}\log(X_{\text{pred}}^{ij}) +  (1 - X_{\text{true}}^{ij})\log(1 - X_{\text{pred}}^{ij})\right]
$$

volume fraction constraints:

$$
\mathcal{L}_{\text{vol}}(X_{\text{true}}, X_{\text{pred}}) = (\bar{X}_{\text{pred}} - \bar{X}_{\text{true}})^2
$$

Loss construction code is as follows:

``` py linenums="263"
--8<--
examples/topopt/topopt.py:263:274
--8<--
```

#### 3.8.2 Metric Construction

The original code of this case chooses Binary Accuracy and IoU for evaluation:

$$
\text{Bin. Acc.} = \frac{w_{00}+w_{11}}{n_{0}+n_{1}}
$$

$$
\text{IoU} = \frac{1}{2}\left[\frac{w_{00}}{n_{0}+w_{10}} + \frac{w_{11}}{n_{1}+w_{01}}\right]
$$

Where $n_{0} = w_{00} + w_{01}$, $n_{1} = w_{10} + w_{11}$, $w_{tp}$ represents the number of pixel points that are actually class $t$ and predicted as class $p$
Metric construction code is as follows:

``` py linenums="277"
--8<--
examples/topopt/topopt.py:277:317
--8<--
```

### 3.9 Model Training

This case has four sub-cases according to different choices of samplers. Case parameters are as follows:

``` yaml linenums="29"
--8<--
examples/topopt/conf/topopt.yaml:29:31
--8<--
```

Training code is as follows:

``` py linenums="77"
--8<--
examples/topopt/topopt.py:77:111
--8<--
```

### 3.10 Evaluation Model

For the four trained models, different channel samplers are used respectively (the second dimension of the original data corresponds to the 100-step output result of the SIMP algorithm, uniformly taking the 5th, 10th, 15th, 20th, ..., 80th channels of the second dimension of the original data and their corresponding gradient information as new inputs to construct the evaluation dataset) for evaluation. During each evaluation, only `cfg.EVAL.num_val_step` bacth data are taken to calculate their average Binary Accuracy and IoU metrics; at the same time, the evaluation result needs to be compared with the threshold judgment result of the input data itself (0.5 as the threshold). Please refer to [Complete Code](#4) for specific code.

#### 3.10.1 Validator Construction

To apply PaddleScience API, here a validator SupervisedValidator is constructed for evaluation at each evaluation:

``` py linenums="218"
--8<--
examples/topopt/topopt.py:218:245
--8<--
```

The validator configuration is similar to the setting of [Constraint Construction](#35). In the reading configuration, `"num_workers": 0` means single-thread reading; evaluation metric `"metric"` is custom evaluation metric, including Binary Accuracy and IoU.

### 3.11 Evaluation Result Visualization

Use `ppsci.utils.misc.plot_curve()` method to directly plot the results of Binary Accuracy and IoU:

``` py linenums="185"
--8<--
examples/topopt/topopt.py:185:193
--8<--
```

## 4. Complete Code

``` py linenums="1" title="topopt.py"
--8<--
examples/topopt/topopt.py
--8<--
```

``` py linenums="1" title="functions.py"
--8<--
examples/topopt/functions.py
--8<--
```

``` py linenums="1" title="topoptmodel.py"
--8<--
examples/topopt/topoptmodel.py
--8<--
```

## 5. Result Display

The figure below shows the performance of 4 models on 16 different evaluation datasets respectively, including two metrics: Binary Accuracy and IoU. The abscissa represents different evaluation datasets, for example: abscissa $i$ represents the evaluation dataset constructed by the $5\cdot(i+1)$-th channel of the second dimension of the original data and its corresponding gradient information; the ordinate is the evaluation metric. The metric corresponding to `thresholding` can be understood as benchmark.

<figure markdown>
  ![bin_acc](https://ai-studio-static-online.cdn.bcebos.com/859ca7c5d6bb4d60b4e1a329b369c3f2bb942ba281664d1a9156397a34a9191b){ loading=lazy }
  <figcaption>Binary Accuracy Result</figcaption>
</figure>

<figure markdown>
  ![iou](https://ai-studio-static-online.cdn.bcebos.com/807ea645100447818d0bafc39e9d489eef26b7a0195d4bd59da4e68445d304cb){ loading=lazy }
  <figcaption>IoU Result</figcaption>
</figure>

The metrics in the above figure are represented in a table as:

| bin_acc | eval_dataset_ch_5 | eval_dataset_ch_10 | eval_dataset_ch_15 | eval_dataset_ch_20 | eval_dataset_ch_25 | eval_dataset_ch_30 | eval_dataset_ch_35 | eval_dataset_ch_40 | eval_dataset_ch_45 | eval_dataset_ch_50 | eval_dataset_ch_55 | eval_dataset_ch_60 | eval_dataset_ch_65 | eval_dataset_ch_70 | eval_dataset_ch_75 | eval_dataset_ch_80 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Poisson5 | 0.9471 | 0.9619 | 0.9702 | 0.9742 | 0.9782 | 0.9801 | 0.9803 | 0.9825 | 0.9824 | 0.9837 | 0.9850 | 0.9850 | 0.9870 | 0.9863 | 0.9870 | 0.9872 |
| Poisson10 | 0.9457 | 0.9703 | 0.9745 | 0.9798 | 0.9827 | 0.9845 | 0.9859 | 0.9870 | 0.9882 | 0.9880 | 0.9893 | 0.9899 | 0.9882 | 0.9899 | 0.9905 | 0.9904 |
| Poisson30 | 0.9257 | 0.9595 | 0.9737 | 0.9832 | 0.9828 | 0.9883 | 0.9885 | 0.9892 | 0.9901 | 0.9916 | 0.9924 | 0.9925 | 0.9926 | 0.9929 | 0.9937 | 0.9936 |
| Uniform | 0.9410 | 0.9673 | 0.9718 | 0.9727 | 0.9818 | 0.9824 | 0.9826 | 0.9845 | 0.9856 | 0.9892 | 0.9892 | 0.9907 | 0.9890 | 0.9916 | 0.9914 | 0.9922 |

| iou | eval_dataset_ch_5 | eval_dataset_ch_10 | eval_dataset_ch_15 | eval_dataset_ch_20 | eval_dataset_ch_25 | eval_dataset_ch_30 | eval_dataset_ch_35 | eval_dataset_ch_40 | eval_dataset_ch_45 | eval_dataset_ch_50 | eval_dataset_ch_55 | eval_dataset_ch_60 | eval_dataset_ch_65 | eval_dataset_ch_70 | eval_dataset_ch_75 | eval_dataset_ch_80 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Poisson5 | 0.8995 | 0.9267 | 0.9421 | 0.9497 | 0.9574 | 0.9610 | 0.9614 | 0.9657 | 0.9655 | 0.9679 | 0.9704 | 0.9704 | 0.9743 | 0.9730 | 0.9744 | 0.9747 |
| Poisson10 | 0.8969 | 0.9424 | 0.9502 | 0.9604 | 0.9660 | 0.9696 | 0.9722 | 0.9743 | 0.9767 | 0.9762 | 0.9789 | 0.9800 | 0.9768 | 0.9801 | 0.9813 | 0.9810 |
| Poisson30 | 0.8617 | 0.9221 | 0.9488 | 0.9670 | 0.9662 | 0.9769 | 0.9773 | 0.9786 | 0.9803 | 0.9833 | 0.9850 | 0.9853 | 0.9855 | 0.9860 | 0.9875 | 0.9873 |
| Uniform | 0.8887 | 0.9367 | 0.9452 | 0.9468 | 0.9644 | 0.9655 | 0.9659 | 0.9695 | 0.9717 | 0.9787 | 0.9787 | 0.9816 | 0.9784 | 0.9835 | 0.9831 | 0.9845 |

## 6. References

- [Sosnovik I, & Oseledets I. Neural networks for topology optimization](https://arxiv.org/pdf/1709.09578)

- [Reference Code](https://github.com/ISosnovik/nn4topopt/blob/master/)
