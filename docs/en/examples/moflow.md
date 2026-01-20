# MoFlow(Flow Model for Molecular)

<!--
<a href="https://aistudio.baidu.com/aistudio/projectdetail/6184070?contributionType=1&sUid=438690&shared=1&ts=1684239806160" class="md-button md-button--primary" style>AI Studio Quick Experience</a>
-->

!!! note "Precautions"

    1. Before starting training and evaluation, please download the [QM9 Dataset and ZINC Dataset](https://aistudio.baidu.com/datasetdetail/282687), and modify the `FILE_PATH` in the yaml configuration file to the path of the decompressed dataset. It is recommended to place it in example./datasets/moflow.
    2. Before starting training, testing, and optimization evaluation, please install additional chemical packages and data display conversion tools with the command `pip install -r requirements.txt`, install [rdkit](https://github.com/rdkit/rdkit) chemical tool and [cairosvg](https://github.com/Kozea/CairoSVG) data conversion and saving tool.
    3. The pre-trained model needs to be modified and placed in the specified folder, the corresponding yaml configuration file should be modified, and the prompts appearing when executing imperative molecular generation is unreasonable can be ignored.

=== "Model Training Command"

    ``` sh
    # qm9 dataset model training
    python moflow_train.py data_name=qm9

    # zinc250k dataset model training
    python moflow_train.py data_name=zinc250k
    ```

=== "Model Inference and Evaluation Command"

    ``` sh
    # qm9 dataset pre-trained model generation evaluation, where EVAL_mode=Reconstruct is reconstruction generation, EVAL_mode=Random is random generation, EVAL_mode=Inter2point is intermolecular interpolation generation, EVAL_mode=Intergrid is molecular grid interpolation generation, refer to 3.7 Model Generation Evaluation Construction for details
    python test_generate.py data_name=qm9 EVAL_mode=Reconstruct EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/qm9/qm9_pretrained.pdparams

    # zinc250k dataset pre-trained model generation evaluation, where EVAL_mode=Reconstruct is reconstruction generation, EVAL_mode=Random is random generation, EVAL_mode=Inter2point is intermolecular interpolation generation, EVAL_mode=Intergrid is molecular grid interpolation generation, refer to 3.7 Model Generation Evaluation Construction for details
    python test_generate.py data_name=zinc250k EVAL_mode=Reconstruct EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/zinc250k/zinc250k_pretrained.pdparams
    ```

=== "Model Optimization and Evaluation Command"

    ``` sh
    # Method 1: Do not use pre-trained model, the first run is model training, the second run is prediction generation result output
    # qm9 dataset pre-trained model optimization, where OPTIMIZE.property_name=qed is latent space to QED property, OPTIMIZE.property_name=plogp is from latent space to plogp property, refer to 3.8 Model Optimization Construction for details
    python optimize_moflow.py data_name=qm9  TRAIN.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/qm9/qm9_pretrained.pdparams  OPTIMIZE.property_name=qed

    # zinc250k dataset pre-trained model optimization, where OPTIMIZE.property_name=qed is latent space to QED property, OPTIMIZE.property_name=plogp is from latent space to plogp property, refer to 3.8 Model Optimization Construction for details
    python optimize_moflow.py data_name=zinc250k  TRAIN.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/zinc250k/zinc250k_pretrained.pdparams OPTIMIZE.property_name=qed

    # Method 2: Use provided pre-trained model, download optimized model for prediction result generation output
    # qm9 dataset pre-trained model optimization
    mkdir -p ./outputs_moflow_optimize/qm9/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/qm9/qed_opt_pretrained.pdparams -O ./outputs_moflow_optimize/qm9/qed_model.pdparams
    python optimize_moflow.py data_name=qm9  TRAIN.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/qm9/qm9_pretrained.pdparams  OPTIMIZE.property_name=qed

    wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/qm9/plogp_opt_pretrained.pdparams -O ./outputs_moflow_optimize/qm9/plogp_model.pdparams
    python optimize_moflow.py data_name=qm9  TRAIN.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/qm9/qm9_pretrained.pdparams  OPTIMIZE.property_name=plogp

    # zinc250k dataset pre-trained model optimization
    mkdir -p ./outputs_moflow_optimize/zinc250k/
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/zinc250k/qed_opt_pretrained.pdparams -O ./outputs_moflow_optimize/zinc250k/qed_model.pdparams
    python optimize_moflow.py data_name=zinc250k  TRAIN.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/zinc250k/zinc250k_pretrained.pdparams OPTIMIZE.property_name=qed

    wget -c https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/zinc250k/plogp_opt_pretrained.pdparams -O ./outputs_moflow_optimize/zinc250k/plogp_model.pdparams
    python optimize_moflow.py data_name=zinc250k  TRAIN.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/zinc250k/zinc250k_pretrained.pdparams OPTIMIZE.property_name=plogp
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [qm9](https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/qm9/qm9_pretrained.pdparams)|  loss(Residual): <br> 1.09976 |
| [zinc250k](https://paddle-org.bj.bcebos.com/paddlescience/models/MoFlow/zinc250k/zinc250k_pretrained.pdparams)|  loss(Residual):: <br> 1.12570 |

## 1. Background Introduction

MoFlow is a flow-based graph generation model designed to accelerate the drug discovery process by generating molecular graphs with desired chemical properties. Such graph generation models usually include two steps: learning latent representations and generating molecular graphs. Generating novel molecular graphs that conform to chemical rules from latent representations is very challenging because molecular graphs have chemical constraints and combinatorial complexity.

MoFlow is used to learn the invertible mapping between molecular graphs and their latent representations. First, bonds (edges) are generated through a Glow-based model, and then atoms (nodes) are generated given bonds through a novel graph conditional flow model. Finally, they are assembled into a molecular graph conforming to chemical rules through post-processing validity correction. It has advantages such as accurate and computable likelihood training, efficient one-time embedding and generation, chemical validity guarantee, 100% reconstruction of training data, and good generalization ability. A variant of the Glow model is used to generate bonds (multi-type edges, such as single bonds, double bonds, and triple bonds), and a novel graph conditional flow based on graph convolution is used to generate atoms (multi-type nodes, such as C, N, etc.) according to bonds, assembling atoms and bonds into effective molecular graphs conforming to bond valence constraints.

MoFlow is one of the first flow-based graph generation models capable of generating molecular graphs at once through invertible mapping with validity guarantee. In order to capture the combinatorial atom and bond structure of molecular graphs, the proposed Glow model is used to generate bonds (edges), and a novel method based on graph conditional flow is used to generate atoms (nodes) according to bonds, and then they are assembled into effective molecular graphs; many good results have been achieved in molecular graph generation, reconstruction, optimization, etc. One-time inference and generation are very efficient, which means it has the potential for efficiency and effectiveness in exploring chemical space for drug discovery.

## 2. Model Principle

This chapter only briefly introduces the model principle of MoFlow. For detailed model structure and derivation process, please read the paper [MoFlow: An Invertible Flow Model for Generating Molecular Graphs](https://arxiv.org/abs/2006.10137v1).

### 2.1 Basic Framework of the Model

The Flow model learns a series of invertible transformations $f_Θ = f_L ◦ ... ◦ f_1$ between complex high-dimensional data $X \sim P_\mathcal{X}(X)$ and $Z \sim P_\mathcal{Z}(Z)$ in a latent space with the same dimension, where the latent distribution $P_\mathcal{Z}(Z)$ is easy to model (for example, a strong independence assumption holds in such a latent space). Potential complex data in the original space can be modeled by the variable transformation formula, where $Z = f_Θ(X)$ and:

$$
\begin{aligned}
P_\mathcal{X}(X) & = P_\mathcal{Z}(Z)|\det(\frac{\partial Z}{\partial X})
\end{aligned}
$$

Sampling $\widetilde{X} \sim P_\mathcal{X}(X)$ transforms $f_Θ$ by sampling $\widetilde{Z} \sim P_\mathcal{Z}(Z)$ and then reverse mapping through $\widetilde{X} = f_Θ^{−1}\widetilde{Z}$. Let $Z = f_Θ(X) = f_L ◦ ... ◦ f_1(X), H_l = f_l(H_{l−1})$, where $f_l(l = 1, ...L ∈ \mathbb{N}^+)$ is an invertible map, $H_0 = X, H_L = Z$, and $P_\mathcal{Z}(Z)$ follows a standard isotropic Gaussian distribution with independent dimensions. Then, the log likelihood of $X$ can be obtained through the variable transformation formula:

$$
\begin{aligned}
\log P_\mathcal{X}(X) &=\log P_\mathcal{Z}(Z) + \log \left|\det\left(\frac{\partial Z}{\partial X}\right) \right| \\
&= \sum_{i} \log P_{\mathcal{Z}_i}(Z_i) + \sum_{l=1}^L \log \left|\det\left(\frac{\partial f_l}{\partial H_{l−1}}\right)\right|
\end{aligned}
$$

Where $P_{\mathcal{Z}_i}(Z_i)$ is the probability of the $i^{th}$ dimension of $Z$, and $fΘ = f_L ◦ ... ◦ f_1$ is the invertible deep neural network to be learned.

Coupling invertible affine coupling layer, designing an expressive structure with an invertible function f, capable of calculating the efficiency of the Jacobian determinant through an affine coupling transformation $Z = f_Θ(X): \mathbb{R}^n \mapsto \mathbb{R}^n$:

$$
\begin{aligned}
Z_{1:d} & = X_{1:d} \\
Z_{d+1:n} & = X_{d+1:n} ⊙ e^{S_Θ(X_{1:d})} + T_Θ(X_{1:d})
\end{aligned}
$$

By dividing $X$ into two partitions $X = (X_{1:d}, X_{d+1:n})$, invertibility is guaranteed by:

$$
\begin{aligned}
X_{1:d} & = Z_{1:d} \\
X_{d+1:n} & = (Z_{d+1:n} - T_Θ(X_{1:d}) )/ e^{S_Θ(Z_{1:d})}
\end{aligned}
$$

Expressiveness depends on the scale function $S_Θ：\mathbb{R}^d \mapsto \mathbb{R}^{n-d}$ and transformation function $T_Θ：\mathbb{R}^d \mapsto \mathbb{R}^{n-d}$ of any neural structure in the affine transformation of $X_{d+1:n}$. The Jacobian determinant can be efficiently calculated by: $\det(\frac{\partial Z}{\partial X}) = \exp(\sum_j S_Θ(X_{1:d}))$.

### 2.2 Principle of MoFlow Model

Consider the molecular graph $\text{M}$ as an undirected graph composed of atoms as nodes and bonds as edges. Its mathematical notation can be denoted as $\mathcal{M} =  \mathcal{A} \times  \mathcal{B} \subset \mathbb{R}^{n \times k} \times \mathbb{R}^{c \times n \times n}$, where the set has $n$ atoms and $k$ atom types, $A(i,k)=1$ represents node $i$ is a $k$-type atom, the set represents bonds (edges), bonds have $c$ types, $B(c,i,j)=1$ represents atom $i$ and $j$ are connected by a $c$-type bond. A molecule $\mathcal{M}$ can be regarded as an undirected graph with multi-type nodes and multi-type edges. The main goal is to learn a molecular generation model $P_{\mathcal{M}}(M)$, that is, the probability of sampling any molecule $\text{M}$ from $P_{\mathcal{M}}$. In order to capture the combinatorial atom and bond structure of the molecular graph, $P_{\mathcal{M}}(M)$ is decomposed into two parts:

$$
\begin{aligned}
P_\mathcal{M}(M) = P_{\mathcal{M}}((A, B)) ≈ P_{\mathcal{A|B}}(A|B; θ_{\mathcal{A|B}})P_\mathcal{B}(B; θ_\mathcal{B})
\end{aligned}
$$

Where $P_{\mathcal{M}}$ is the distribution of molecular graphs, $P_\mathcal{B}$ is the distribution of bonds (edges), similar to modeling multi-channel images, and $P_{\mathcal{A|B}}$ is the conditional distribution of atoms (nodes) given bonds, modeled by utilizing graph convolution operations. $θ_\mathcal{B}$ and $θ_{\mathcal{A|B}}$ are learnable modeling parameters. The objective function of the model is as follows:

$$
\begin{aligned}
\mathop{\arg\max}\limits_{\theta_\mathcal{B}, \theta_\mathcal{A|B}}
\mathbb{E}_{\mathcal{M}=(A,B) \sim \mathcal{PM}−data} [ \log P_\mathcal{A|B}(A | B; θ_\mathcal{A|B} + \log P_\mathcal{B}(B; θ_\mathcal{B})]
\end{aligned}
$$

Given bond tensor $B \in \mathcal{B} \subset \mathbb{R}^{c×n×n}$, generate correct atom type matrix $A \in \mathcal{A} \subset \mathbb{R}^{n×k}$ to form a valid molecule $M = (A, B) \in \mathcal{M} \subset \mathbb{R}^{n×k+c×n×n}$. First define $B$ conditional flow and graph conditional flow $f_\mathcal{A|B}$, transforming $A$ given $B$ into conditional latent variable $Z_{A|B} = f_\mathcal{A|B}(A|B)$, which follows isotropic Gaussian distribution $P_{\mathcal{Z}_\mathcal{A|B}}$. Through the conditional variable transformation formula, the conditional probability $P_\mathcal{A|B}$ of atom features given the bond graph can be obtained. $B$ conditional flow $Z_{A|B} = f_\mathcal{A|B}(A|B)$ is an invertible and dimension-preserving map, and there exists an inverse transformation $f^{−1}_\mathcal{A|B}(Z_{A|B} |B) = A|B$, where $f_\mathcal{A|B}$ and $f^{−1}_\mathcal{A|B}：\mathcal{A \times B} \mapsto \mathcal{A \times B}$. During the transformation, $B \in B$ remains unchanged. Under the condition of independence assumption of $A$ and $B$, the Jacobian matrix of $f_\mathcal{A|B}$ is:

$$
\begin{aligned}
\frac{\partial f_\mathcal{A|B}}{\partial (A, B)}=\bigg[\begin{matrix}
\frac{\partial f_\mathcal{A|B}}{\partial A} & \frac{\partial f_\mathcal{A|B}}{\partial B} \\ 0 & \mathbb{1}_B \end{matrix}\bigg]
\end{aligned}
$$

Having obtained the distribution, we can sample from it, use the inverse map to get $A|B$, and use the Jacobian matrix to give the probability distribution of $A|B$. The log likelihood of the conditional variable transformation formula is:

$$
\begin{aligned}
\log P_\mathcal{A|B}(A|B) = \log P_{\mathcal{Z}_\mathcal{A|B}}(Z_{A|B}) + \log |\det \frac{\partial f_\mathcal{A|B}}{\partial A}|
\end{aligned}
$$

Like flow-based RealNVP and Glow models, in order to obtain invertible mapping, Moflow introduces graph coupling layers. For each graph coupling layer, input $A \in \mathbb{R}^{n×k}$ is divided into two parts $A = (A_1, A_2)$ along the n row dimension, and then output $Z_{A|B} = (Z_{A_1|B}, Z_{A_2|B}) = f_\mathcal{A|B}(A|B)$ is obtained as follows, dividing the input into two parts $A_1$ and $A_2$:

$$
\begin{aligned}
Z_{A_1 |B} &= A_1 \\
Z_{A_2 |B} &= A_2 \odot \text{Sigmoid}(S_Θ(A_1 |B)) + T_Θ(A_1 |B)
\end{aligned}
$$

Inverting the above formula yields $A_1$ and $A_2$. The graph convolution layer is completed using Relational Graph Convolutional Network (R-GCN), specifically as follows:

$$
\begin{aligned}
\text{graphconv}(A_1) & = \sum_{i=1}^c \hat{B}_i (M \odot A)W_i + (M \odot A)W_0
\end{aligned}
$$

At the same time, multiple stacked graph convolution->BatchNorm1d->ReLU layers and a Multi-Layer Perceptron (MLP) output layer are used to construct graph scaling function $S_Θ$ and graph transformation function $T_Θ$. For numerical stability, the Sigmoid function is adopted in $S_Θ$ to achieve numerical stability when cascading multiple flow layers. The inverse map $f^{-1}_\mathcal{A|B}$ of the graph coupling layer is:

$$
\begin{aligned}
A_1 &= Z_{A_1}|B \\
A_2 &= (Z_{A_2}|B - T_Θ(Z_{A_1|B}|B)) / \text{Sigmoid}(S_Θ(Z_{A_1|B}|B))
\end{aligned}
$$

The logarithm of the Jacobian determinant of each graph coupling layer can be calculated as follows:

$$
\begin{aligned}
\log | \det （\frac{\partial f_\mathcal{A|B}}{\partial A}）|= \sum_j \log \text{Sigmoid}(S_Θ(A_1|B))_j
\end{aligned}
$$

Where $j$ iterates over each element. Arbitrarily complex graph convolution structures can be used to construct $S_Θ$ and $T_Θ$, because the above calculation of the Jacobian determinant of $f_\mathcal{A|B}$ does not involve the calculation of the Jacobian matrix of $S_Θ$ or $T_Θ$.

When learning atom representation, in order to ensure data stability, $\sigma^2 \in \mathbb{R}^{n \times 1}$ is used for normalization for each row dimension, so that the input result after normalization is $\hat A = \frac{A - \mu} {\sqrt{\sigma^2 + \epsilon}}$, where $\epsilon$ is a small constant. The inverse transformation is $A = \hat A \times \sqrt{\sigma^2 + \epsilon} + \mu$, and the logarithmic Jacobian determinant is:

$$
\begin{aligned}
\log | \det \frac{\partial actnorm2D}{\partial X}| = \frac{k}{2}\sum_i^n | \log(\sigma^2_i + \epsilon) |
\end{aligned}
$$

In learning bond data representation, the idea based on Glow is adopted, similar to the above steps of learning atom representation, and for data stability, the $1 \times 1$ convolution operation in the Glow model is also introduced.

Finally, chemical validity verification is performed, following the valence bond limit of each atom, adopting whether the combination of atoms and bonds conforms to the chemical bond valence constraint, and the bond valence constraint is defined:

$$
\begin{aligned}
\sum_{c,j}c \times B(c, i, j) \le \text{Valency}(\text{Atom}_i) + Ch
\end{aligned}
$$

Where $c$ is the type of bond (single bond, double bond, triple bond). Unlike other models, the constraint of formal charge $Ch$ is added. This effect may introduce additional bonds for charged atoms. For example, N of ammonium [NH4]+ may have 4 bonds instead of 3. Similarly, S+ and O+ may have 3 bonds instead of 2.

The model structure is shown in the figure:
<center class ='img'>
<img title="MoFlow Model Structure Diagram" src="https://paddle-org.bj.bcebos.com/paddlescience/docs/moflow.jpg" width="80%"></br>MoFlow Model Structure Diagram
</center>

### 2.3 Dataset Introduction

**QM9** dataset is derived from the enumerated subset of the GDB-17 database. GDB-17 is a chemical universe containing 166 billion small organic molecules, and QM9 filters out all stable molecules containing no more than 9 heavy atoms.
* Total number of molecules: about 134,000 (specifically 133,885 stable organic molecules).
* Atom composition: Only contains four heavy atoms: carbon (C), nitrogen (N), oxygen (O), fluorine (F), and hydrogen (H).
* Element vocabulary: In the QM9 implementation of MoFlow, the atom type list is strictly defined as ['C', 'N', 'O', 'F'].
* Maximum size: 9 heavy atoms. This means that in tensor representation, $N=9$.

Every molecule has undergone high-precision Density Functional Theory (DFT) calculations, specifically at the B3LYP/6-31G(2df,p) level. These calculations provide geometric, energetic, electronic, and thermodynamic properties. In MoFlow research, the following properties are focused on for conditional generation and property optimization tasks: Symbol Property Name Unit Physical Meaning Relevance to Generation Model HOMO Highest Occupied Molecular Orbital Energy eV Measures the electron donating ability of the molecule. The higher the HOMO energy, the easier it is for the molecule to lose electrons. LUMO Lowest Unoccupied Molecular Orbital Energy eV Measures the electron accepting ability of the molecule. The lower the LUMO energy, the easier it is for the molecule to accept electrons. Gap ($\Delta \epsilon$) HOMO-LUMO Gap eV Determines the chemical hardness and light absorption characteristics of the molecule. This is one of the core goals of MoFlow for property optimization. $\mu$ Dipole Moment Debye Describes the asymmetry of molecular charge distribution. Generating molecules with specific polarity is crucial in material design. $\alpha$ Isotropic Polarizability $Bohr^3$ The response ability of the molecule under an external electric field. $U_0$ Internal Energy at 0K Hartree Thermodynamic stability indicator of the molecule. $C_v$ Heat Capacity at Constant Volume cal/mol K Thermodynamic property, often used as the target for regression tasks.

**ZINC250k** contains about 249,455 molecular graphs. The construction of this subset is not random, but follows strict "Drug-likeness" and "Synthesizability" criteria, initially established by Gómez-Bombarelli et al. in their pioneering automatic chemical design paper, and subsequently widely adopted by subsequent studies such as MoFlow.
Screening Criteria include:
* Heavy atom count limit: Molecular size is limited to within 38 heavy atoms. This is much larger than QM9, allowing for more complex ring systems and long chain structures.
* Element diversity: Chemical space expanded from CHNOF to include elements such as halogen and phosphorus-sulfur. The atomic number list defined in MoFlow code is: ``, corresponding to C, N, O, F, P, S, Cl, Br, I respectively.
* LogP range: The lipid-water partition coefficient (LogP) of the molecule needs to be within a specific range to ensure oral bioavailability.
* Synthetic Accessibility (SA): Priority is given to molecules that are easy to synthesize.
* Structure filtering: Molecules containing rings larger than 8 members were eliminated, and complex salt forms with charges were also eliminated, simplifying the topological difficulty of graph generation.

## 3. Model Implementation

Next, we will explain how to implement the model reproduction of structural reconstruction in drug molecules based on PaddleScience code. To achieve MoFlow model construction, training, inference, and evaluation, only key steps such as model construction, training, testing, and evaluation are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Data Processing

In data processing, first by reading the chemical molecular structure and using chemical molecule library processing, extract chemical bonds and molecular nodes from the chemical structure part of the dataset, and process the atomic structure and bond values. Expressed in PaddleScience code as follows

``` py linenums="394" title="data/dataset/moflow_dataset.py"
--8<--
ppsci/data/dataset/moflow_dataset.py:394:427
--8<--
```

Training data is selected using set labels, dividing the dataset into training data and test data. The processing of qm9 and zinc250k datasets is consistent, with some differences in feature and atomic structure processing choices.

### 3.2 Constraint Construction

This case solves the problem based on the method of learning chemical bond constraints from data, so according to the PaddleScience API structure description, the built-in SupervisedConstraint is used to construct supervised constraints. Before defining constraints, you need to first specify various parameters used for data loading in supervised constraints.

``` py linenums="140" title="examples/moflow/moflow_train.py"
--8<--
examples/moflow/moflow_train.py:140:161
--8<--
```

Among them, the "dataset" field defines the used `Dataset` class name as `MOlFLOWDataset`, the "sampler" field defines the used `Sampler` class name as `BatchSampler`, the set `batch_size` is 256, and `num_works` is 8.

The code for defining supervised constraints is as follows:

``` py linenums="167" title="examples/moflow/moflow_train.py"
--8<--
examples/moflow/moflow_train.py:167:177
--8<--
```

### 3.3 Model Construction

In this case, the drug molecule prediction generation model is implemented based on the MoFlowNet network model. Combined with the PaddleScience code standard format, the model is encapsulated, and flow, grow and other models are called separately. The code for model composition is represented as follows:

``` py linenums="162" title="examples/moflow/moflow_train.py"
--8<--
examples/moflow/moflow_train.py:162:166
--8<--
```

Model network parameter configuration is as follows:

``` py linenums="97" title="examples/moflow/moflow_train.py"
--8<--
examples/moflow/moflow_train.py:97:128
--8<--
```

Parameters are set through the configuration file as follows:

``` py linenums="22" title="examples/moflow/conf/moflow_train.yaml"
--8<--
examples/moflow/conf/moflow_train.yaml:22:79
--8<--
```

Among them, data_name represents the selection of the dataset. After selection, the network parameter part corresponding to different datasets is selected accordingly. input_keys and output_keys represent the names of the input and output variables of the network model respectively. hyper_params represents the network parameters corresponding to different datasets, which will be updated in the model construction after dataset selection to facilitate unified construction of models under different datasets. Use the model's custom loss function for model training.

### 3.4 Learning Rate and Optimizer Construction

The learning rate size used in this case is set to `0.001`. The optimizer uses `Adam`, expressed in PaddleScience code as follows:

``` py linenums="181" title="examples/moflow/moflow_train.py"
--8<--
examples/moflow/moflow_train.py:181:183
--8<--
```

### 3.5 Validator Construction

During the training process of this case, the training status of the current model will be evaluated using the validation set at certain training round intervals, and `SupervisedValidator` is needed to construct the validator. The code is as follows:

``` py linenums="184" title="examples/moflow/moflow_train.py"
--8<--
examples/moflow/moflow_train.py:184:213
--8<--
```

Evaluation metric `metric` uses a custom function to generate molecules using molecular vector values, and independently evaluates the regenerated molecules. The custom evaluation metrics used here are `valid`, `unique`, and `abs_unique`.

### 3.6 Model Training and Evaluation

After completing the above settings, you only need to pass the instantiated objects to ppsci.solver.Solver in order, and then start training and evaluation.

``` py linenums="214" title="examples/moflow/moflow_train.py"
--8<--
examples/moflow/moflow_train.py:214:236
--8<--
```

### 3.7 Model Generation Evaluation Construction

For different dataset constructions, different evaluation methods for different models are provided. The model generation ability is comprehensively evaluated through reconstruction, random generation, and interpolation generation. Different methods have different parameter configurations. The parameter configuration file is as follows:

``` py linenums="81" title="examples/moflow/conf/moflow_test.yaml"
--8<--
examples/moflow/conf/moflow_test.yaml:81:126
--8<--
```

Among them, EVAL_mode is the selected evaluation mode, and different modes have different evaluation methods. Reconstruct (reconstruction generation) performs reconstruction generation of drug molecules for different datasets, reconstructing molecules in the selected dataset; Random (random generation) performs random generation from the latent space for different datasets, and the parameter setting is to randomly generate 5 times from 10000 samples; Inter2point (intermolecular interpolation generation) performs interpolation in the latent space, and visualizes generated molecular graphs by interpolating between two molecules; Intergrid (molecular grid interpolation generation) performs interpolation in the latent space, and visualizes generated molecular graphs using molecular grids (interpolation generation stores generated new molecules as visible pictures). Parameters in each mode are adjusted according to the actual situation, including result storage, number of generated molecules, etc. The rest of the configuration is the same as training. When selecting models trained on different datasets, pay attention to modifying the data name and checking the location of the pre-trained model.

The code for building the evaluator is:

``` py linenums="368" title="examples/moflow/test_generate.py"
--8<--
examples/moflow/test_generate.py:368:529
--8<--
```

### 3.8 Model Optimization Construction

After the model completes training, molecular optimization and constrained optimization are performed. An additional MLP model is trained from latent space to QED property or plogp property to obtain optimized molecular properties and evaluate them, subject to constraints on similarity with properties. If running for the first time, the selected pre-trained model will be optimized and trained. Different properties store different optimization models. QED property saves with prefix qed, and plogp property saves with prefix plogp. Relevant optimization models will be saved to the specified folder. When running for the second time, the optimized model will be evaluated. The code is as follows:

``` py linenums="470" title="examples/moflow/optimize_moflow.py"
--8<--
examples/moflow/optimize_moflow.py:470:586
--8<--
```

The main parameters are similar to training, and evaluation parameters need to be set separately. Parameters are set through the configuration file as follows:

``` py linenums="94" title="examples/moflow/conf/moflow_optimize.yaml"
--8<--
examples/moflow/conf/moflow_optimize.yaml:94:107
--8<--
```

## 4. Complete Training Code

``` py linenums="1" title="examples/moflow/moflow_train.py"
--8<--
examples/moflow/moflow_train.py
--8<--
```

## 5. References
[Article] [MoFlow: An Invertible Flow Model for Generating Molecular Graphs](https://arxiv.org/abs/2006.10137v1)

[Code] [Moflow](https://github.com/calvin-zcx/moflow)
