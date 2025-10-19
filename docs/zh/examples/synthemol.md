# Synthemol

!!! note

    1. 开始训练、评估前，请先下载实验所用数据集 [Data.zip](https://paddle-org.bj.bcebos.com/paddlescience/datasets/synthemol/Data.zip) ，并对应修改 yaml 配置文件中的 `data_dir` 为解压后的数据集路径。例如："./data/Data/..."；下载 [resources.zip](https://paddle-org.bj.bcebos.com/paddlescience/datasets/synthemol/resources.zip), 将其解压至examples/synthemol/synthemol/下。
    2. 如果需要使用预训练模型进行评估，请先下载预训练模型[pretrained.zip](https://paddle-org.bj.bcebos.com/paddlescience/models/synthemol/pretrained.zip)并解压，例如./pretrained/pretrained_chemprop.pdparams路径,并在yaml配置文件的PRE_COMPUTE.model_path指明路径。
    3. 开始训练、生成前，请安装 `rdkit` 等，相关依赖请执行`pip install requirements.txt`安装。

=== "Property Predictor模型训练命令"

    ``` sh
    # 使用antibiotics等数据训练模型chemprop模型,实现Property Predict
    # 配置可在conf/synthemol.yaml进行修改
    python main.py mode=train
    ```

=== "Property Predictor模型评估命令"

    ``` sh
    # 下载预训练模型（可选，或配置文件指定自己训练的模型）
    mkdir -p ./pretrained && wget -O ./pretrained/pretrained_chemprop.pdparams https://paddle-org.bj.bcebos.com/paddlescience/models/synthemol/pretrained_chemprop.pdparams
    # 使用antibiotics等数据评估模型chemprop模型,实现Property Predict
    # 配置可在conf/synthemol.yaml进行修改
    python main.py mode=eval
    ```

=== "预计算building blocks分数命令"

    ``` sh
    # 下载预训练模型（可选，或配置文件指定自己训练的模型）
    mkdir -p ./pretrained && wget -O ./pretrained/pretrained_chemprop.pdparams https://paddle-org.bj.bcebos.com/paddlescience/models/synthemol/pretrained_chemprop.pdparams
    # 使用训练好的模型进行building blocks的分数与计算，以加速下一个生成阶段
    # 配置可在conf/synthemol.yaml进行修改
    python main.py mode=pre-compute
    ```

=== "使用synthemol生成分子命令"

    ``` sh
    # 使用预计算的building blocks的分数指引，结合synthemol使用蒙特卡洛树搜索，进行分子生成
    # 配置可在conf/synthemol.yaml进行修改
    python main.py mode=generate
    ```

## 1. 背景简介

泛耐药菌的迅速出现，使得开发结构全新的抗生素变得刻不容缓。人工智能虽可发现新型抗生素，但现有方法仍有明显缺陷：性质预测模型只能逐一评估分子，面对庞大的化学空间时扩展性极差；而生成式模型虽能快速探索巨量化学空间，却常输出难以合成的分子。为此，作者提出了 SyntheMol，一种生成式模型，可从近 300 亿个分子的化学空间中设计出易于合成的新化合物。作者将 SyntheMol 用于抑制鲍曼不动杆菌（一种棘手的革兰阴性病原菌）的生长，共合成 58 个生成分子并进行实验验证，其中 6 个结构全新的分子对鲍曼不动杆菌及其他多种系统发育差异显著的细菌均表现出抗菌活性。该研究展示了生成式人工智能在庞大化学空间中设计结构新颖、可合成且有效的小分子抗生素候选物的潜力，并提供了实验验证。

## 2. Synthemol原理

本章节仅对 Synthemol 的模型原理进行简单地介绍，详细的理论推导请阅读 [Generative AI for designing and validating easily synthesizable and structurally novel antibiotics](https://www.nature.com/articles/s42256-024-00809-7)。

### 2.1 Property Predictor

Chemprop 是一种分子性质预测模型，它利用有向消息传递神经网络处理分子，并对其性质进行预测。Chemprop 首先从分子图中提取简单的原子与键特征（如原子类型和键类型），为每个原子和键构建特征向量。接着，模型执行三轮消息传递：在每一轮中，神经网络层将邻近原子和键的信息迭代融合。消息传递完成后，Chemprop 将所有融合后的特征向量求和，生成一个代表整个分子的单一特征向量。该向量再输入一个两层的前馈神经网络，以预测分子性质；在本研究中，即预测抑制鲍曼不动杆菌生长的概率。我们使用的版本为 Chemprop v1.5.2，迁移自 PyTorch v1.12.0.post2。另外两种predictor请参考原文。

### 2.2 Synthemol

SyntheMol 是一种生成式模型，它在组合化学空间中进行探索，该空间由分子砌块经化学反应所生成的分子构成，以寻找具有目标性质的分子。SyntheMol 采用与 AlphaGo 类似的蒙特卡洛树搜索（MCTS）算法，高效地在这一化学空间中搜寻理想分子。SyntheMol 不仅能迅速识别出有前景的分子，还能同时给出其合成路线（即通过一系列一步或多步化学反应，将分子砌块组合起来的完整步骤）。以下，我们给出描述 SyntheMol MCTS 算法所需的数学符号，并提供相应的伪代码。

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

## 3. Synthemol模型实现

接下来开始讲解如何基于 PaddleScience 代码，实现 Synthemol 模型的训练、预计算分数与生成。关于该案例中的其余细节请参考 [API文档](../api/arch.md)。

### 3.1 数据集介绍

数据集采用了作者仓库 [Synthemol](https://github.com/swansonk14/SyntheMol) 的 Data.zip 数据集。

训练集由 3 个化合物库组成：  

- 库 1 共 2371 个分子，来自 Pharmakon-1760 库（含 1 360 种 FDA 批准药物和 400 种国际批准药物）以及 800 种从植物、动物和微生物来源分离的天然产物。  
- 库 2 为 Broad Drug Repurposing Hub，共 6 680 个分子，其中多数为 FDA 批准药物或临床候选化合物。  
- 库 3 为一个小分子合成筛选库，含 5 376 个分子，系从 Broad Institute 更大的化合物库中随机抽样获得。

所有 3 个库均以两次生物学重复的形式，对鲍曼不动杆菌 ATCC 17978 进行生长抑制活性筛选。实验流程如下：

1. 将菌株于 37 °C 在 2 ml LB 培养基中过夜培养，随后以 1:10 000 稀释于新鲜 LB。  
2. 取 49.5 µl（384 孔板）或 99 µl（96 孔板）菌液，使用手工或 Agilent Bravo 移液系统加入 Corning 平底微孔板。  
3. 每孔加入待测化合物，终浓度 50 µM，终体积 50 µl（384 孔板）或 100 µl（96 孔板）。  
4. 37 °C 静置孵育 16 h。  
5. 使用 SpectraMax M3 酶标仪（Molecular Devices）于 600 nm 读取吸光度，数据按板内四分位均值归一化，随后进行汇总与阳性命中判定。

更多详细信息，包括每个模型的超参数调整空间等，请参考作者原始论文。本仓库使用的具体超参数已在yaml配置文件中预设，可根据情况自行调节。

### 3.2 Chemprop模型训练

#### 3.2.1 约束构建

本案例基于数据驱动的方法求解问题，因此需要使用 PaddleScience 内置的 `SupervisedConstraint` 构建监督约束。在定义约束之前，需要首先指定监督约束中用于数据加载的各个参数。

数据加载的代码如下:

``` py linenums="224" title="examples/synthemol/main.py"
--8<--
examples/synthemol/main.py:224:236
--8<--
```

其中，"dataset" 字段定义了使用的 `Dataset` 类名为 `MoleculeDatasetIter`，`num_works` 为 1。

定义监督约束的代码如下：

``` py linenums="238" title="examples/synthemol/main.py"
--8<--
examples/synthemol/main.py:238:247
--8<--
```

`SupervisedConstraint` 的第一个参数是数据的加载方式，这里使用上文中定义的 `train_dataloader_cfg`；

第二个参数是损失函数的定义，这里使用自定义的损失函数；作者通过 `get_loss_func` 函数通过传递参数控制损失函数选择： 文中Chemprop模型使用的为 `CrossEntropyLoss`；

第三个参数是约束条件的名字，方便后续对其索引。此处命名为 `Sup`。

#### 3.2.2 模型构建

在该案例中，分子属性预测模型基于 Chemprop 网络模型实现，用 PaddleScience 代码表示如下：

``` py linenums="249" title="examples/synthemol/main.py"
--8<--
examples/synthemol/main.py:249:250
--8<--
```

网络模型的参数通过配置文件进行设置如下：

``` yaml linenums="32" title="examples/synthemol/conf/synthemol.yaml"
--8<--
examples/synthemol/conf/synthemol.yaml:32:36
--8<--
```

其中，`input_keys` 和 `output_keys` 分别代表网络模型输入、输出变量的名称。

#### 3.2.3 学习率与优化器构建

本案例中使用的学习率大小设置为 `0.0001`。优化器使用 `Adam`，并将参数进行分组,用 PaddleScience 代码表示如下：

``` py linenums="252" title="examples/synthemol/main.py"
--8<--
examples/synthemol/main.py:252:256
--8<--
```

#### 3.2.4 模型训练

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练。

``` py linenums="258" title="examples/synthemol/main.py"
--8<--
examples/synthemol/main.py:258:275
--8<--
```

### 3.3 building blocks分数预计算

构建模型的代码为：

``` py linenums="348" title="examples/synthemol/main.py"
--8<--
examples/synthemol/main.py:348:348
--8<--
```

### 3.4 Synthemol生成分子

构建Generator的代码为：

``` py linenums="514" title="examples/synthemol/main.py"
--8<--
examples/synthemol/main.py:514:528
--8<--
```

## 4. 完整代码

``` py linenums="1" title="examples/synthemol/main.py"
--8<--
examples/synthemol/main.py
--8<--
```

## 5. 结果展示

评估第一步Chemprop模型的训练效果，通过加载预训练模型并执行评估命令，可以得到结果：

| | roc_auc | prc_auc |
|:-- | :-- | :-- |
| chemprop | 0.797 | 0.332 |

查看生成的molecules.csv，可以看到类似于下表的生成的分子信息：

| smiles | node_id | num_expansions | rollout_num | score | Q_value | num_reactions | reaction_1_id | building_block_1_1_id | building_block_1_1_smiles | building_block_1_2_id | building_block_1_2_smiles |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| C#CCN(C(=O)C(C)(C)C#C)C1CCN(C(=O)OC(C)(C)C)CC1 | 91431 | 20 | 1 |  |  | 1 | 22 | 4349560 | C#CCNC1CCN(C(=O)OC(C)(C)C)CC1 | 2998277 | C#CC(C)(C)C(=O)O |

可以看到生成了符合要求的分子信息，符合作者的设计目的。

## 6. 参考文献

- [Generative AI for designing and validating easily synthesizable and structurally novel antibiotics](https://www.nature.com/articles/s42256-024-00809-7)
- [作者原始仓库](https://github.com/swansonk14/SyntheMol)
