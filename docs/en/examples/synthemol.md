# Synthemol

!!! note

    1. Before starting training and evaluation, please download the dataset used in the experiment [Data.zip](https://paddle-org.bj.bcebos.com/paddlescience/datasets/synthemol/Data.zip), and modify `data_dir` in the yaml configuration file to the path of the decompressed dataset. For example: "./data/Data/..."; download [resources.zip](https://paddle-org.bj.bcebos.com/paddlescience/datasets/synthemol/resources.zip), and unzip it to examples/synthemol/synthemol/.
    2. If you need to use a pre-trained model for evaluation, please download the pre-trained model [pretrained.zip](https://paddle-org.bj.bcebos.com/paddlescience/models/synthemol/pretrained.zip) and unzip it, for example to the path ./pretrained/pretrained_chemprop.pdparams, and specify the path in PRE_COMPUTE.model_path of the yaml configuration file.
    3. Before starting training and generation, please install `rdkit` etc. For related dependencies, please execute `pip install requirements.txt` to install.

=== "Property Predictor Model Training Command"

    ``` sh
    # Use antibiotics and other data to train chemprop model to implement Property Predict
    # Configuration can be modified in conf/synthemol.yaml
    python main.py
    ```

=== "Property Predictor Model Evaluation Command"

    ``` sh
    # Download pre-trained model (optional, or specify your own trained model in configuration file)
    mkdir -p ./pretrained && wget -O ./pretrained/pretrained_chemprop.pdparams https://paddle-org.bj.bcebos.com/paddlescience/models/synthemol/pretrained_chemprop.pdparams
    # Use antibiotics and other data to evaluate chemprop model to implement Property Predict
    # Configuration can be modified in conf/synthemol.yaml
    python main.py mode=eval
    ```

=== "Pre-compute building blocks score command"

    ``` sh
    # Download pre-trained model (optional, or specify your own trained model in configuration file)
    mkdir -p ./pretrained && wget -O ./pretrained/pretrained_chemprop.pdparams https://paddle-org.bj.bcebos.com/paddlescience/models/synthemol/pretrained_chemprop.pdparams
    # Use trained model to score and compute building blocks to accelerate the next generation phase
    # Configuration can be modified in conf/synthemol.yaml
    python main.py mode=pre-compute
    ```

=== "Use synthemol to generate molecules command"

    ``` sh
    # Use pre-computed building blocks score guidance, combined with synthemol using Monte Carlo Tree Search to generate molecules
    # Configuration can be modified in conf/synthemol.yaml
    python main.py mode=generate
    ```

## 1. Background Introduction

The rapid emergence of pan-drug-resistant bacteria makes the development of structurally novel antibiotics urgent. Although artificial intelligence can discover new antibiotics, existing methods still have obvious flaws: property prediction models can only evaluate molecules one by one, which has extremely poor scalability when facing huge chemical spaces; while generative models can quickly explore huge chemical spaces, they often output molecules that are difficult to synthesize. To this end, the authors proposed SyntheMol, a generative model that can design new compounds that are easy to synthesize from a chemical space of nearly 30 billion molecules. The authors used SyntheMol to inhibit the growth of Acinetobacter baumannii (a tricky Gram-negative pathogen), synthesized 58 generated molecules and experimentally verified them, of which 6 structurally novel molecules showed antibacterial activity against Acinetobacter baumannii and other bacteria with significant phylogenetic differences. This study demonstrates the potential of generative AI to design structurally novel, synthesizable, and effective small-molecule antibiotic candidates in a huge chemical space, and provides experimental validation.

## 2. Synthemol Principle

This chapter only briefly introduces the model principle of Synthemol. For detailed theoretical derivation, please read [Generative AI for designing and validating easily synthesizable and structurally novel antibiotics](https://www.nature.com/articles/s42256-024-00809-7).

### 2.1 Property Predictor

Chemprop is a molecular property prediction model that uses directed message passing neural networks to process molecules and predict their properties. Chemprop first extracts simple atom and bond features (such as atom type and bond type) from the molecular graph to construct feature vectors for each atom and bond. Then, the model performs three rounds of message passing: in each round, the neural network layer iteratively fuses information from neighboring atoms and bonds. After message passing is completed, Chemprop sums all fused feature vectors to generate a single feature vector representing the entire molecule. This vector is then input into a two-layer feedforward neural network to predict molecular properties; in this study, it predicts the probability of inhibiting the growth of Acinetobacter baumannii. We use Chemprop v1.5.2, migrated from PyTorch v1.12.0.post2. For two other predictors, please refer to the original text.

### 2.2 Synthemol

SyntheMol is a generative model that explores a combinatorial chemical space composed of molecules generated by chemical reactions of molecular building blocks to find molecules with target properties. SyntheMol uses a Monte Carlo Tree Search (MCTS) algorithm similar to AlphaGo to efficiently search for ideal molecules in this chemical space. SyntheMol can not only quickly identify promising molecules, but also give their synthesis routes (that is, the complete steps of combining molecular building blocks through a series of one-step or multi-step chemical reactions). Below, we give the mathematical symbols required to describe the SyntheMol MCTS algorithm and provide the corresponding pseudocode.

### SyntheMol MCTS Algorithm

**Requires:**

- Synthesis tree `T`
- Property prediction model `M`
- Maximum number of rollouts `n_rollout`
- Maximum number of reactions `n_reaction`

---

**function `MCTS()`:**
&nbsp;&nbsp;&nbsp;&nbsp;**for** `i = 1` to `n_rollout` **do**:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`rollout(T.root)`
&nbsp;&nbsp;&nbsp;&nbsp;**end for**
&nbsp;&nbsp;&nbsp;&nbsp;**return** all visited nodes in `T` with:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1 molecule and ≥ 1 reaction

---

**function `rollout(N)`:**
&nbsp;&nbsp;&nbsp;&nbsp;**if** node `N` has undergone `≥ n_reaction` reactions **then**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**return** property prediction score of `M` applied to molecules in `N`
&nbsp;&nbsp;&nbsp;&nbsp;**end if**
&nbsp;&nbsp;&nbsp;&nbsp;`E ← expand_node(N)`
&nbsp;&nbsp;&nbsp;&nbsp;`S ← select` child node in `E` with largest MCTS score
&nbsp;&nbsp;&nbsp;&nbsp;**return** `rollout(S)`

---

**function `expand_node(N)`:**
&nbsp;&nbsp;&nbsp;&nbsp;`E ← empty set of nodes`
&nbsp;&nbsp;&nbsp;&nbsp;**foreach** reaction `R` **do**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** `R` is compatible with molecules in `N` **then**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Add new node to `E` with each product of `R` applied to molecules in `N`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end if**
&nbsp;&nbsp;&nbsp;&nbsp;**end for**
&nbsp;&nbsp;&nbsp;&nbsp;**foreach** building block `B` **do**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** any reaction is compatible with `B` and molecules in `N` **then**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Add new node to `E` with `B` and molecules in `N`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end if**
&nbsp;&nbsp;&nbsp;&nbsp;**end for**
&nbsp;&nbsp;&nbsp;&nbsp;**return** `E`

## 3. Synthemol Model Implementation

Next, we will explain how to implement the training, pre-calculation score and generation of the Synthemol model based on PaddleScience code. For other details in this case, please refer to [API Documentation](../api/arch.md).

### 3.1 Dataset Introduction

The dataset uses the Data.zip dataset from the author's repository [Synthemol](https://github.com/swansonk14/SyntheMol).

The training set consists of 3 compound libraries:

- Library 1 contains 2371 molecules from the Pharmakon-1760 library (containing 1360 FDA-approved drugs and 400 internationally approved drugs) and 800 natural products isolated from plant, animal and microbial sources.
- Library 2 is the Broad Drug Repurposing Hub, containing 6680 molecules, most of which are FDA-approved drugs or clinical candidate compounds.
- Library 3 is a small molecule synthesis screening library containing 5376 molecules, randomly sampled from a larger compound library of the Broad Institute.

All 3 libraries were screened for growth inhibition activity against Acinetobacter baumannii ATCC 17978 in duplicate biological replicates. The experimental process is as follows:

1. The strain was cultured overnight in 2 ml LB medium at 37 °C, and then diluted 1:10 000 in fresh LB.
2. Take 49.5 µl (384-well plate) or 99 µl (96-well plate) bacterial solution and add it to Corning flat-bottom microplate using manual or Agilent Bravo pipetting system.
3. Add the test compound to each well, final concentration 50 µM, final volume 50 µl (384-well plate) or 100 µl (96-well plate).
4. Incubate at 37 °C for 16 h.
5. Read absorbance at 600 nm using SpectraMax M3 microplate reader (Molecular Devices), normalize data by intra-plate quartile mean, and then aggregate and determine positive hits.

For more details, including hyperparameter adjustment space for each model, please refer to the author's original paper. Specific hyperparameters used in this repository are preset in the yaml configuration file and can be adjusted according to the situation.

### 3.2 Chemprop Model Training

#### 3.2.1 Constraint Construction

This case solves the problem based on data-driven methods, so it is necessary to use `SupervisedConstraint` built in PaddleScience to construct supervised constraints. Before defining constraints, you need to first specify various parameters used for data loading in supervised constraints.

The code for data loading is as follows:

``` py linenums="224" title="examples/synthemol/main.py"
--8<--
examples/synthemol/main.py:224:236
--8<--
```

Among them, the "dataset" field defines the used `Dataset` class name as `MoleculeDatasetIter`, and `num_works` is 1.

The code for defining supervised constraints is as follows:

``` py linenums="238" title="examples/synthemol/main.py"
--8<--
examples/synthemol/main.py:238:247
--8<--
```

The first parameter of `SupervisedConstraint` is the data loading method, here `train_dataloader_cfg` defined above is used;

The second parameter is the definition of loss function, here a custom loss function is used; the author controls loss function selection by passing parameters through `get_loss_func` function: the Chemprop model in the paper uses `CrossEntropyLoss`;

The third parameter is the name of the constraint condition, which is convenient for subsequent indexing. Here it is named `Sup`.

#### 3.2.2 Model Construction

In this case, the molecular property prediction model is implemented based on the Chemprop network model, expressed in PaddleScience code as follows:

``` py linenums="249" title="examples/synthemol/main.py"
--8<--
examples/synthemol/main.py:249:250
--8<--
```

The parameters of the network model are set through the configuration file as follows:

``` yaml linenums="32" title="examples/synthemol/conf/synthemol.yaml"
--8<--
examples/synthemol/conf/synthemol.yaml:32:36
--8<--
```

Among them, `input_keys` and `output_keys` represent the names of the input and output variables of the network model respectively.

#### 3.2.3 Learning Rate and Optimizer Construction

The learning rate size used in this case is set to `0.0001`. The optimizer uses `Adam`, and parameters are grouped, expressed in PaddleScience code as follows:

``` py linenums="252" title="examples/synthemol/main.py"
--8<--
examples/synthemol/main.py:252:256
--8<--
```

#### 3.2.4 Model Training

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training.

``` py linenums="258" title="examples/synthemol/main.py"
--8<--
examples/synthemol/main.py:258:275
--8<--
```

### 3.3 Pre-compute building blocks score

The code for constructing the model is:

``` py linenums="348" title="examples/synthemol/main.py"
--8<--
examples/synthemol/main.py:348:348
--8<--
```

### 3.4 Synthemol Generate Molecules

The code for constructing Generator is:

``` py linenums="514" title="examples/synthemol/main.py"
--8<--
examples/synthemol/main.py:514:528
--8<--
```

## 4. Complete Code

``` py linenums="1" title="examples/synthemol/main.py"
--8<--
examples/synthemol/main.py
--8<--
```

## 5. Result Display

Evaluate the training effect of the first step Chemprop model. By loading the pre-trained model and executing the evaluation command, the results can be obtained:

| | roc_auc | prc_auc |
|:-- | :-- | :-- |
| chemprop | 0.797 | 0.332 |

Checking the generated molecules.csv, you can see the generated molecular information similar to the table below:

| smiles | node_id | num_expansions | rollout_num | score | Q_value | num_reactions | reaction_1_id | building_block_1_1_id | building_block_1_1_smiles | building_block_1_2_id | building_block_1_2_smiles |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| C#CCN(C(=O)C(C)(C)C#C)C1CCN(C(=O)OC(C)(C)C)CC1 | 91431 | 20 | 1 |  |  | 1 | 22 | 4349560 | C#CCNC1CCN(C(=O)OC(C)(C)C)CC1 | 2998277 | C#CC(C)(C)C(=O)O |

It can be seen that molecular information meeting the requirements is generated, which is consistent with the author's design purpose.

## 6. References

- [Generative AI for designing and validating easily synthesizable and structurally novel antibiotics](https://www.nature.com/articles/s42256-024-00809-7)
- [Author's original repository](https://github.com/swansonk14/SyntheMol)
