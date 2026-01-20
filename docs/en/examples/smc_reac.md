# Suzuki-Miyaura Cross-Coupling Reaction Yield Prediction

!!! note

    1. Before starting training and evaluation, please download the data file [data_set.xlsx](https://paddle-org.bj.bcebos.com/paddlescience/datasets/SMCReac/data_set.xlsx), and modify `data_dir` in the yaml configuration file to the actual file path of `data_set.xlsx`.
    2. If you need to use a pre-trained model for evaluation, please download the pre-trained model [smc_reac_model.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/smc_reac/smc_reac_model.pdparams), and modify `load_model_path` in the yaml configuration file to the model parameter path.
    3. Before the first training and evaluation, please execute `pip install -r requirements.txt` to install `rdkit` and other related dependencies.

=== "Model Training Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/SMCReac/data_set.xlsx
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/SMCReac/data_set.xlsx -o data_set.xlsx
    python smc_reac.py
    ```

=== "Model Evaluation Command"

    ``` sh
    # linux
    wget -c https://paddle-org.bj.bcebos.com/paddlescience/datasets/SMCReac/data_set.xlsx
    # windows
    # curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/SMCReac/data_set.xlsx -o data_set.xlsx
    python smc_reac.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/smc_reac/smc_reac_model.pdparams
    ```

## 1. Background Introduction

The Suzuki-Miyaura cross-coupling reaction expression is shown below.

$$
\mathrm{Ar{-}X} + \mathrm{Ar'{-}B(OH)_2} \xrightarrow[\text{Base}]{\mathrm{Pd}^0} \mathrm{Ar{-}Ar'} + \mathrm{HX}
$$

Catalyzed by zero-valent palladium complexes, aryl or alkenyl boronic acids or boronic esters undergo cross-coupling with chloro-, bromo-, iodoarenes or alkenes. This reaction has the advantages of mild reaction conditions and high conversion rate, and plays an important role in fields such as material synthesis and drug development, but has problems such as long development cycle and high trial and error cost. This study establishes a prediction model by using high-throughput experimental data to analyze the effects of reaction substrates (including electrophiles and nucleophiles), catalytic ligands, bases, and solvents on the yield of coupling reactions.

## 2. Implementation of Suzuki-Miyaura Cross-Coupling Reaction Yield Prediction Model

This section will explain how to implement the construction, training, testing and evaluation of the Suzuki-Miyaura cross-coupling reaction yield prediction model based on PaddleScience code. The directory structure of the case is as follows.
``` log
smc_reac/
├──config/
│   └── smc_reac.yaml
├── data_set.xlsx
├── requirements.txt
└── smc_reac.py
```

### 2.1 Dataset Construction and Loading

The data used in this example comes from the open source data provided in reference [1], considering only the influence of reagents themselves on experimental results, and filtering out partial reaction data where reagents participated in all components, saved in the file `./data_set.xlsx`. This work developed an automated platform based on flow chemistry, which was assembled in an argon-protected glove box, using a modified high-performance liquid chromatography (HPLC) system combined with an automated sampling device to draw reaction components (electrophiles, nucleophiles, catalysts, ligands, and bases) from 192 reservoirs according to a set program and inject them into the flow carrier liquid. Each reaction segment reacts in a temperature-controlled reaction coil at a set flow rate, pressure, and time, and the reaction solution is detected in real time by UPLC-MS. By regulating the combination of electrophiles, nucleophiles, 11 ligands, 7 bases, and 4 solvents, a systematic screening of 5760 reaction conditions was finally achieved. Next, taking one piece of data as an example, combined with code to illustrate the construction and loading process of the dataset.

```
ClC=1C=C2C=CC=NC2=CC1 | CC=1C(=C2C=NN(C2=CC1)C1OCCCC1)B(O)O | C(C)(C)(C)P(C(C)(C)C)C(C)(C)C | [OH-].[Na+] | C(C)#N | 4.76
```
Where SMILES are used to represent electrophiles, nucleophiles, catalytic ligands, bases, solvents, and experimental yields in turn.

First, import experimental material information and reaction yield from the table file, and divide the training set and test set,

``` py linenums="26" title="examples/smc_reac/smc_reac.py"
--8<--
examples/smc_reac/smc_reac.py:26:34
--8<--
```

Apply `rdkit.Chem.rdFingerprintGenerator` to convert the SMILES descriptions of electrophiles, nucleophiles, catalytic ligands, bases, and solvents into Morgan fingerprints. Morgan fingerprint is a vectorized description of molecular structure, encoded as hash values through local topology, and mapped to 2048-bit fingerprint bits. Expressed in PaddleScience code as follows

``` py linenums="37" title="examples/smc_reac/smc_reac.py"
--8<--
examples/smc_reac/smc_reac.py:37:65
--8<--
```

### 2.2 Constraint Construction

This case uses supervised learning. According to the PaddleScience API structure description, the built-in `SupervisedConstraint` is used to construct supervised constraints. Expressed in PaddleScience code as follows

``` py linenums="73" title="examples/smc_reac/smc_reac.py"
--8<--
examples/smc_reac/smc_reac.py:73:88
--8<--
```

The second parameter of `SupervisedConstraint` indicates using mean squared error `MSELoss` as the loss function, and the third parameter indicates the name of the constraint condition, which is convenient for subsequent indexing.

### 2.3 Model Construction

This case designed five independent sub-networks (fully connected layer + ReLU activation), each sub-network extracts features of corresponding chemical substances respectively. Subsequently, these five feature vectors are weighted averaged through trainable weight parameters to achieve adaptive learning of the impact of different chemical components on reaction yield prediction. Finally, the fused features are input into a fully connected layer for further mapping to output the predicted value of reaction yield. The entire network structure reflects the independent extraction and weighted fusion of information of each component in the reaction, consistent with the characteristics of the reaction mechanism. Expressed in PaddleScience code as follows

``` py linenums="7" title="ppsci/arch/smc_reac.py"
--8<--
ppsci/arch/smc_reac.py:7:107
--8<--
```

The model is instantiated according to the configuration file information

``` py linenums="90" title="examples/smc_reac/smc_reac.py"
--8<--
examples/smc_reac/smc_reac.py:90:90
--8<--
```

Parameters are set through the configuration file as follows

``` py linenums="35" title="examples/smc_reac/config/smc_reac.yaml"
--8<--
examples/smc_reac/config/smc_reac.yaml:35:41
--8<--
```

### 2.4 Optimizer Construction

The trainer uses the Adam optimizer, and the learning rate setting is given by the configuration file. Expressed in PaddleScience code as follows

``` py linenums="92" title="examples/smc_reac/smc_reac.py"
--8<--
examples/smc_reac/smc_reac.py:92:92
--8<--
```

### 2.5 Model Training

After completing the above settings, you only need to pass the instantiated objects to `ppsci.solver.Solver` in order, and then start training. Expressed in PaddleScience code as follows

``` py linenums="95" title="examples/smc_reac/smc_reac.py"
--8<--
examples/smc_reac/smc_reac.py:95:104
--8<--
```

## 3. Complete Code

``` py linenums="1" title="examples/smc_reac/smc_reac.py"
--8<--
examples/smc_reac/smc_reac.py
--8<--
```

## 4. Result Display

The figure below shows the model prediction results for the yield of the Suzuki-Miyaura cross-coupling reaction.

<figure markdown>
  ![chem.png](https://paddle-org.bj.bcebos.com/paddlescience/docs/SMCReac/chem.png){ loading=lazy }
  <figcaption> Model prediction results for Suzuki-Miyaura cross-coupling reaction yield</figcaption>
</figure>

## 5. References

[1] Perera D, Tucker J W, Brahmbhatt S, et al. A platform for automated nanomole-scale reaction screening and micromole-scale synthesis in flow[J]. Science, 2018, 359(6374): 429-434.
