# IFM-MLP

!!! note

    1. Before starting training and evaluation, please download the molecules dataset [dataset.zip](https://paddle-org.bj.bcebos.com/paddlescience/datasets/IFM/dataset.zip), or [Google Drive (Original link)](https://drive.google.com/drive/folders/1ZYdYQ0TtmShJC-z6dr4BU1aPfeQSE9gD?usp=sharing), and modify `data_dir` in the yaml configuration file to the path of the decompressed dataset.
    2. If you need to use a pre-trained model for evaluation, please download the pre-trained model [pretrained.zip](https://paddle-org.bj.bcebos.com/paddlescience/models/IFM/pretrained.zip) and unzip it, for example to the `pretrained` path.
    3. Before starting training and evaluation, please install `rdkit` and `scikit-learn`, etc. Execute `pip install requirements.txt` to install relevant dependencies.

=== "Model Training Command"

    ``` sh
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/IFM/dataset.zip
    unzip dataset.zip
    # Train MLP-IFM model on tox21/sider/hiv/bace/bbbp etc. data, embed_name optional IFM/None
    # Parameters such as mode/data_label/MODEL.embed_name can be configured in conf/ifm.yaml
    python ifm.py data_label=tox21 MODEL.embed_name='IFM'
    ```

=== "Model Evaluation Command"

    ``` sh
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/IFM/dataset.zip
    unzip dataset.zip
    # Evaluate MLP-IFM model on tox21/sider/hiv/bace/bbbp etc. data, embed_name optional IFM/None
    # Pre-trained model path example: pretrained/IFM/bace/model.pdparams or use self-trained model path
    python ifm.py mode=eval data_label=tox21 MODEL.embed_name='IFM' EVAL.pretrained_model_path=pretrained/IFM/bace/model.pdparams
    ```

## 1. Background Introduction

Molecular Property Prediction (MPP) is a key task in computational drug discovery aimed at identifying properties with desirable pharmacology and ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity). Machine learning models have been widely used in this rapidly developing field, and there are two commonly used models: traditional non-deep models and deep models. In non-deep models, molecules are fed into traditional machine learning models, such as calculated or manually designed molecular fingerprints into random forests and support vector machines. Another category utilizes deep models to extract representations of molecules in a data-driven manner. Specifically, for example, Multilayer Perceptrons (MLP) can be applied to calculated or manually designed molecular fingerprints; sequence-based neural network architectures including Recurrent Neural Networks (RNN), 1D Convolutional Neural Networks (1D CNN) and Transformers can be used to encode representations of molecular SMILES strings.

In addition, molecules can naturally be represented as graph structures with atoms as nodes and bonds as edges, inspiring a series of works dedicated to utilizing this structured inductive bias to obtain better molecular representations. A key outcome of these methods is Graph Neural Networks (GNN), which consider both graph structure and attribute features during learning. Recently, researchers have achieved better performance by incorporating 3D conformations of molecules into their representations. However, based on practical considerations such as computational cost, alignment invariance, uncertainty in conformation generation, and unavailable conformations for target molecules, the practical applicability of these models is limited. The authors summarized widely used molecular descriptors and their corresponding models for benchmarking. A large number of previous studies observed that deep models struggle to outperform non-deep models on molecular datasets. However, these studies did not consider emerging deep models (e.g., Transformer, SphereNet), nor did they study the impact of different molecular descriptors (e.g., 3D molecular graphs), nor did they investigate the deep reasons why models often perform poorly on molecules.

Therefore, the authors conducted a comprehensive benchmark study of molecular property prediction, as well as precise methods for dataset and hyperparameter tuning. The results confirmed observations from previous studies that deep models are generally difficult to outperform traditional non-deep models, even without considering the slower training speed of deep learning algorithms. Therefore, based on the above problems, the authors proposed a simple and effective feature mapping method IFM to help deep models learn non-smooth objective functions in theoretical situations, achieving better results.

## 2. IFM Model Principle

### 2.1 IFM Method

This chapter only briefly introduces the model principle of IFM. For detailed theoretical derivation, please read [Understanding the Limitations of Deep Models for Molecular property prediction: Insights and Solutions](https://openreview.net/pdf?id=NLFqlDeuzt).

As explained in the author's paper, deep models struggle to learn non-smooth objective function data for molecules, a phenomenon known in the literature as "spectral bias". To overcome this bias, some previous work experimentally found that heuristic sinusoidal mapping of input features allows MLPs to learn non-smooth objective functions. However, these mapping methods will inevitably mix in the original features. To address this situation, the authors introduced a new method called Independent Feature Mapping (IFM), which implements embedding separately before feeding each dimension of molecular features into the model. Denoting molecular features as $x ∈ \mathbb{R}^d$, we represent IFM as:

$$
\begin{equation}
f_x = [\sin(v)|| \cos(v)], v = [2πc_1x, . . . , 2πc_kx]
\end{equation}
$$

Where $||$ denotes concatenation of two vectors, $c = [c_1, c_2, ···, c_k]$ are learnable parameters initialized from $N(0, σ)$ and $f_x ∈ \mathbb{R}^{2k×d}$. The authors studied the impact of hyperparameters $k$ and $σ$. Since $\cos(a − b) = \cos a \cos b + \sin a \sin b$, we have:

$$
\begin{equation}
f_x · f_{x^′} =\sum_{i=1}^k cos(2πc_i(x − x^′)) := g_c(x − x^′)
\end{equation}
$$

Where · is the dot product, and $x^′$ is another molecular feature. Thus, IFM can map data points to a vector space such that their dot product achieves a certain distance metric, which is a characteristic of expected feature mapping methods. Based on previous research, the authors provide a theoretical basis for the effectiveness of IFM. As demonstrated by the effectiveness of some previous work, deep models can be approximated by Neural Tangent Kernels (NTK). Specifically, let $I$ represent a fully connected deep network whose weights $θ$ are from a Gaussian initialized distribution $N$. NTK theory shows that as the width of layers in $I$ becomes infinite and the learning rate of Stochastic Gradient Descent (SGD) approaches zero, the function $I(x; θ)$ converges during training to the kernel regression solution using the Neural Tangent Kernel (NTK), i.e.:

$$
\begin{equation}
h_{NTK}(x, x^′) = E_{θ∼N} \langle \frac{∂I(x; θ)}{∂θ} , \frac{∂I(x^′; θ)}{∂θ} \rangle
\end{equation}
$$

When inputs are restricted to a hypersphere, the NTK of an MLP can be expressed as a dot product kernel (of the form $h_{NTK}(x · x^′)$ for a scalar function $h_{NTK} : \mathbb{R} → \mathbb{R}$). In the author's scheme, the input to the deep model is $f_x$, and the combined kernel of IFM and NTK can be expressed as:

$$
\begin{equation}
h_{NTK} (f_x · f_{x^′} ) = h_{NTK} (g_c (x − x^′)) = (h_{NTK} \circ g_c)(x − x^′)
\end{equation}
$$

Therefore, training a deep model on these mapped molecular features corresponds to kernel regression with a fixed combined NTK function $h_{NTK} \circ g_c$. Considering that parameter $c$ is adjustable, IFM creates a combined NTK that is not fixed but adjustable. It allows us to efficiently control the learned frequency range by manipulating parameter $c$.

### 2.2 IFM Combined with MLP Model Training and Inference Experiments

In our experiments, we equipped various deep models with IFM. Specifically, for MLPs taking fingerprints as input, we directly applied the proposed feature mapping method to the fingerprints (after feature selection and normalization).

## 3. IFM Model Implementation

Next, we will explain how to implement IFM-MLP model training and inference based on PaddleScience code. For other details in this case, please refer to [API Documentation](../api/arch.md).

### 3.1 Dataset Introduction

The dataset uses the molecules dataset processed by the author [IFM](https://github.com/junxia97/IFM).

This dataset is processed and provided by the IFM author. In the article, the author compared 12 datasets, and the provided data download includes at least 5 molecular datasets, such as bace, bbbp, hiv, sider, tox21, etc. Datasets are saved in csv format. The dataset contains SMILES strings, labels and fingerprints of molecules.

**Fingerprints Data Settings**

Taking Fingerprints used by MLP as an example: Following common practice, the author concatenated various molecular fingerprints, including 881 PubChem fingerprints (PubchemFP), 307 substructure fingerprints (SubFP) and 206 MOE 1-D and 2-D descriptors, to provide SVM, XGB, RF and MLP models to comprehensively represent molecular structures, and removed some features through some preprocessing procedures, specifically: (1) missing values; (2) extremely low variance (variance < 0.05); (3) high correlation with another feature (Pearson correlation coefficient > 0.95). Retained features were normalized to mean 0 and variance 1. In addition, considering that traditional machine models (SVM, RF, XGB) cannot be directly applied to multi-task molecular datasets, the author divided multi-task datasets into multiple single-task datasets and used each dataset to train the model.

**Data Protocol and Test Setup**

First, the author randomly split the training set, validation set and test set in a ratio of 8:1:1. Subsequently, hyperparameters were adjusted based on the performance of the validation set, and using the previously determined best hyperparameters, 50 independent runs with different random seeds and different dataset splits were performed to obtain more reliable results. Following the MoleculeNet benchmark, the author used Area Under the Receiver Operating Characteristic Curve (AUC-ROC) to evaluate classification tasks, except for Area Under the Precision-Recall Curve (AUC-PRC) on the MUV dataset due to extreme bias in its data distribution. Root Mean Squared Error (RMSE) or Mean Absolute Error (MAE) was used to report performance for regression tasks. The author reported average performance for multi-tasks on some datasets as they contain multiple tasks. Furthermore, to avoid overfitting, if no improvement in validation performance was observed for 50 consecutive epochs, all deep models were trained using an early stopping scheme. The author set the maximum epoch to 300 and batch size to 128. For more details, including hyperparameter tuning space for each model, please refer to the author's original paper.

Specific hyperparameters used in this repository are preset in the yaml configuration file and can be adjusted according to the situation.

### 3.2 Model Pretraining

#### 3.2.1 Constraint Construction

This case solves the problem based on data-driven methods, so it is necessary to use `SupervisedConstraint` built in PaddleScience to construct supervised constraints. Before defining constraints, you need to first specify various parameters used for data loading in supervised constraints.

Data loading code is as follows:

``` py linenums="72" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py:72:92
--8<--
```

Among them, the "dataset" field defines the `Dataset` class name used as `IFMMoeDataset`, the "sampler" field defines the `Sampler` class name used as `BatchSampler`, `batch_size` is set to 128, and `num_works` is 1.

The code for defining supervised constraints is as follows:

``` py linenums="94" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py:94:100
--8<--
```

The first parameter of `SupervisedConstraint` is the data loading method, here `train_dataloader_cfg` defined above is used;

The second parameter is the definition of loss function, here a custom loss function is used; the author controls loss function selection via Regularization flag `reg` parameter: `MSELoss` or `BCEWithLogitsLoss`;

The third parameter is the name of the constraint condition, which is convenient for subsequent indexing. Here it is named `Sup`.

#### 3.2.2 Model Construction

In this case, the molecular property prediction model is implemented based on the MLP network model, expressed in PaddleScience code as follows:

``` py linenums="256" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py:256:271
--8<--
```

The parameters of the network model are set through the configuration file as follows:

``` yaml linenums="32" title="examples/ifm/conf/ifm.yaml"
--8<--
examples/ifm/conf/ifm.yaml:32:35
--8<--
```

Among them, `input_keys` and `output_keys` represent the names of input and output variables of the network model respectively. Specific hyperparameters hyper_paras refer to the `HYPER_OPT` field in ifm.yaml according to experimental configuration.

#### 3.2.3 Learning Rate and Optimizer Construction

The learning rate size used in this case is set to `0.001`. The optimizer uses `Adam`, and groups parameters to use different `weight_decay`, expressed in PaddleScience code as follows:

``` py linenums="141" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py:141:143
--8<--
```

#### 3.2.4 Validator Construction

During the training process of this case, the training status of the current model will be evaluated using the validation set at certain training round intervals, and `SupervisedValidator` is needed to construct the validator. The code is as follows:

``` py linenums="145" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py:145:182
--8<--
```

The `SupervisedValidator` validator is quite similar to `SupervisedConstraint`, the difference is that the validator needs to set evaluation metric `metric`, here custom evaluation metrics `AUC-ROC`, `PRC-AUC`, `RMSE`, `MAE` and `R2` are used, and the program will set it according to `data_label`, named `My_Metric`.

#### 3.2.5 Model Training and Evaluation

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver`, and then start training and evaluation.

``` py linenums="184" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py:184:202
--8<--
```

### 3.3 Model Evaluation

The code for building the model is:

``` py linenums="256" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py:256:271
--8<--
```

The code for building the validator is:

``` py linenums="273" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py:273:310
--8<--
```

## 4. Complete Code

``` py linenums="1" title="examples/ifm/ifm.py"
--8<--
examples/ifm/ifm.py
--8<--
```

## 5. Result Display

The table below shows the AUC_ROC performance comparison of MLP model without embedding and with IFM proposed by the author on different datasets. You can download the pre-trained model for evaluation [IFM-MLP](https://paddle-org.bj.bcebos.com/paddlescience/models/IFM/pretrained.zip)

|  | tox21 | sider | hiv | bace | bbbp |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **MLP-None** | 0.82682 | 0.50039 | 0.71932 | 0.88891 | 0.66834 |
| **MLP-IFM** | 0.84245 | 0.60289 | 0.74007 | 0.89553 | 0.84864 |
| **MLP-IFM Loss** | 0.25697 | 1.36643 | 0.15742 | 0.47294 | 1.39181 |

It can be seen that the model with IFM module can achieve better prediction results, which is consistent with the author's design purpose.

## 6. References

- [Understanding the Limitations of Deep Models for Molecular property prediction: Insights and Solutions](https://openreview.net/pdf?id=NLFqlDeuzt)
- [Author's original repository](https://github.com/junxia97/IFM)
