# DataEfficientNopt

## 论文信息

| 年份 | 会议                                                             | 作者                                                   | 引用数 | 论文 PDF                                                                                                                      |
| ---- | ---------------------------------------------------------------- | ------------------------------------------------------ | ------ | ----------------------------------------------------------------------------------------------------------------------------- |
| 2024 | 38th Conference on Neural Information Processing Systems (NeurIPS 2024) | 12      | Data-Efficient Operator Learning via Unsupervised Pretraining and In-Context Learning |

## 代码信息

=== "模型训练命令"

    ``` sh
    cd examples/data_efficient_nopt
    # Download possion_64 data from https://drive.google.com/drive/folders/1crIsTZGxZULWhrXkwGDiWF33W6RHxJkf
    # Download helmholtz_64 data from https://drive.google.com/drive/folders/1UjIaF6FsjmN_xlGGSUX-1K2V3EF2Zalw

    # Update the file paths in `cexamples/data_efficient_nopt/config/data_efficient_nopt.yaml`, specify to mode in `train`
    # UPdate the file paths in config/operators_poisson.yaml or config/operators_helmholtz.yaml, specify to `train_path`, `val_path`, `test_path`, `scales_path` and `train_rand_idx_path`

    # possion_64 or helmholtz_64 pretrain, for example, specify as following:
    #   run_name: r0
    #   config: helm-64-pretrain-o1_20_m1
    #   yaml_config: config/operators_helmholtz.yaml
    python data_efficient_nopt.py

    # possion_64 or helmholtz_64 finetune, for example, specify as following:
    #   run_name: r0
    #   config: helm-64-o5_15_ft5_r2
    #   yaml_config: config/operators_helmholtz.yaml
    python data_efficient_nopt.py
    ```

=== "模型评估命令"

    暂无

=== "模型导出命令"

    暂无

=== "模型推理命令"

    ``` sh
    cd examples/data_efficient_nopt
    # Update the file paths in `cexamples/data_efficient_nopt/config/data_efficient_nopt.yaml`, specify to mode in `infer`
    # Use a fine-tuned model as the checkpoint in 'exp' or utilize `model_convert.py` to convert the official checkpoint.
    # UPdate the file paths in config/inference_poisson.yaml or config/inference_poisson.yaml, specify to `train_path`, `test_path` and `scales_path`

    # use your onw finetune checkpoints, or download checkpoint from [FNO-Poisson](https://drive.google.com/drive/folders/1ekmXqqvpaY6pNStTciw1SCAzF0gjFP_V) or [FNO-Helmholtz](https://drive.google.com/drive/folders/1k7US8ZAgB14Wj9bfdgO_Cjw6hOrG6UaZ) and then convert to paddlepaddle weights.
    python model_convert.py --pt-model <pt_checkpiont> --pd-model <pd_checkpiont>

    # possion_64 inference, specify as following:
    #   evaluation: config/inference_poisson.yaml
    #   ckpt_path: <ckpt_path>
    python data_efficient_nopt.py
    ```

## 1. 背景简介

科学和工程领域的许多核心问题都涉及偏微分方程（PDEs）的求解，这通常需要耗费巨大的计算资源进行数值模拟。近年来，机器学习，特别是深度学习，与物理领域知识的结合（即科学机器学习，SciML）为解决这些PDE问题提供了新的范式。其中，算子学习 (Operator Learning) 是一个新兴且前景广阔的方向，其目标是学习从一个函数空间到另一个函数空间的映射（即“算子”），而非传统神经网络学习有限维度输入到有限维度输出的映射。这使得模型能够处理不同分辨率和离散化方案的数据，具有更强的泛化能力。

然而，如同大多数深度学习方法一样，神经算子（Neural Operators）也面临着数据饥渴 (data-hungry) 的挑战。训练高性能的神经算子需要大量的PDE解数据，而这些数据通常是通过高精度的数值模拟获得的，其计算成本极高且耗时。例如，论文中提到，模拟旧金山7.0级地震的整个过程需要巨大的计算能力和时间。这种高昂的数据获取成本，在一定程度上削弱了机器学习方法最初旨在避免昂贵数值模拟的初衷。

为了应对这一挑战，研究人员开始借鉴自然语言处理（NLP）和计算机视觉（CV）等领域在数据效率 (data-efficiency) 方面的成功经验。这些领域通过无监督预训练 (Unsupervised Pretraining) 或自监督学习 (Self-supervised Learning)，利用大规模无标签数据来学习有用的特征表示，从而显著减少了对带标签数据的需求。然而，在科学机器学习领域，无监督预训练的应用仍处于早期阶段，其潜力尚未被充分挖掘。

因此，本研究的背景旨在解决神经算子在数据效率方面的核心痛点，寻求一种能够有效利用无标签物理数据的方法，以减少对昂贵PDE解数据的依赖，从而使神经算子在实际应用中更具可行性和普适性。

## 2. 问题定义

### 2.1 数据集呈现

数据集分别从[Goole Drive for Helmholtz](https://drive.google.com/drive/folders/1UjIaF6FsjmN_xlGGSUX-1K2V3EF2Zalw)和[Google Drive for Poisson](https://drive.google.com/drive/folders/1crIsTZGxZULWhrXkwGDiWF33W6RHxJkf)，目前代码提供FNO_Poisson和FNO_Helmholtz网络的预训练和推理。

- 对Poisson方程，我们使用扩散特征值[1, 20]进行预训练，[5, 15]进行微调。
- 对Helmholtz方程，我们使用扩散特征值[1, 20]进行预训练，[5, 15]进行微调。

### 2.2 核心问题

本研究主要聚焦于解决以下两个核心问题：

- *高昂的标签数据成本*：现有的神经算子方法需要大量的PDE解（即标签数据）进行训练。这些解数据通常通过计算密集型的高精度数值模拟获得，导致数据收集或生成成本极高，限制了神经算子在实际科学和工程应用中的可扩展性。论文旨在寻找一种方法，在保持模型性能的同时，显著降低对昂贵标签数据的需求。

- *分布外（OOD）泛化能力不足*：尽管神经算子在训练数据分布内表现良好，但当面对与训练数据分布不同的新物理参数或初始条件（即OOD场景）时，其泛化能力往往会下降。这对于需要处理各种复杂、多变物理现象的科学计算而言是一个严重限制。论文希望通过引入新的策略来提高神经算子在未知或偏离训练数据分布情况下的预测准确性。

总的来说，论文旨在提出一种新的神经算子学习范式，能够实现 数据高效性（减少对昂贵标签数据的依赖）和 更强的OOD泛化能力，从而拓宽神经算子在复杂科学问题中的应用范围。

### 3. 问题求解

为了解决上述问题，论文提出了一种创新性的数据高效神经算子学习框架，该框架主要由两个阶段组成：*无监督预训练*和*情境学习推理*。

#### 3.1 无监督预训练 (Unsupervised Pretraining)

无监督预训练的目标是在不使用昂贵PDE解数据的情况下，预先训练神经算子以学习物理相关的特征表示，为后续的监督微调提供一个良好的初始化。

- 无标签PDE数据 (Unlabeled PDE Data)：
    论文引入了“无标签PDE数据”的概念，这种数据不包含完整的PDE解。它仅仅包含描述物理系统的输入信息，例如：

    - 物理参数 (Physical Parameters)：如扩散系数、粘度、源项等。
    - 空间坐标 (Spatial Coordinates)：网格点的位置信息。
    - 作用函数 (Forcing Functions)：如随空间变化的力场。
    - 初始条件 (Initial Conditions)：对于时间相关的PDEs，仅提供初始时刻的快照。 生成这类数据的成本远低于执行完整的PDE数值模拟。例如，模拟一个流体场在某个初始条件下的演化过程，远比只生成这个初始条件本身复杂和昂贵。

- 物理启发式代理任务 (Physics-Inspired Proxy Tasks)：
    为了在无标签数据上进行有效预训练，论文设计了两种基于重构（reconstruction-based）的代理任务，这些任务能够引导模型学习物理系统中的内在结构和规律：

    - 掩码自编码器 (Masked Autoencoder, MAE)：
        - 机制：类似于计算机视觉和自然语言处理中的MAE，该任务通过随机掩盖（masking）部分输入数据（例如，物理参数或初始条件网格中的一部分区域），然后要求神经算子预测被掩盖部分的内容。
        - 物理意义：这种任务迫使模型学习物理场在空间和时间上的局部相关性和稀疏感知不变性 (sparsity-aware invariances)。例如，在流体模拟中，一个区域的流体速度会与其相邻区域的速度高度相关，MAE通过预测缺失部分来捕捉这种关系。它有助于模型理解物理场中的冗余性和结构性。

    - 超分辨率 (Super-resolution, SR)：
        - 机制：该任务通过对原始高分辨率的无标签输入数据（如初始条件或作用函数）应用高斯模糊或其他降采样操作，生成低分辨率的输入。然后，神经算子被训练来从这些低分辨率的输入中重构出原始的高分辨率输入。
        - 物理意义：这使得模型能够学习从粗粒度信息推断细粒度细节的能力，这在科学计算中非常有用，因为高分辨率数据往往成本更高。它也促使模型学习对分辨率和模糊的不变性，即无论数据以何种分辨率呈现，模型都能识别并理解其内在的物理结构。

    - 模型架构 (Model Architectures)：
        论文评估了两种主流的编码器-解码器（encoder-decoder）神经算子架构进行预训练：
        - 傅里叶神经算子 (Fourier Neural Operator, FNO)：FNO通过傅里叶变换在频域进行操作，特别适合处理周期性和全局依赖关系。在预训练后，其解码器部分通常会被丢弃，只保留编码器作为特征提取器。
        - Transformer (基于注意力机制的架构)：Transformer通过自注意力机制捕捉长距离依赖关系，在序列数据处理中表现出色。预训练后，Transformer的解码器部分通常会保留，并与编码器一起进行后续的微调。 这两种架构在不同的代理任务和后续微调中展现出各自的优势。

### 3.2 情境学习推理 (In-Context Learning, ICL)

除了无监督预训练，论文还引入了情境学习（In-Context Learning, ICL）来进一步增强模型在分布外（OOD）场景下的泛化能力。ICL的关键在于在推理阶段利用少量与查询样本相似的“情境示例”来指导模型的预测，而无需对模型进行额外的训练或修改。

- 零训练开销 (Zero Training Overhead)：
ICL的显著优势在于，它不要求对预训练或微调后的神经算子模型进行任何额外的训练。它仅仅是在推理时，根据当前待求解的问题，动态地从一个可用的数据集中（例如，少量已有的带标签数据）选择相似的示例。

- 基于相似性的情境示例挖掘 (Similarity-Based In-Context Example Mining)：
    - 机制：对于给定的一个待求解的查询输入（例如，一组新的物理参数和初始条件），ICL方法会计算这个查询输入与数据集中所有可用情境示例之间的“相似性”。
    - 相似性度量：论文发现，在输出空间（即PDE解的特征空间）中衡量相似性比在输入空间中更有效。例如，可以通过计算不同PDE解在L2范数或特定物理量上的距离来衡量它们之间的相似性。
    - 示例选择与聚合：选择与查询输入最相似的K个情境示例及其对应的解。然后，这些情境示例的解会被聚合（例如，通过加权平均）起来，作为对查询输入的预测的补充或修正。这种聚合机制有助于模型根据“附近”的已知解来校准其预测。

- 优势：通过引入与当前查询输入相似的真实解作为“上下文”，模型能够更好地适应新的、未见的物理条件，尤其是在OOD设置下，因为这些情境示例提供了对未知物理规律的局部“提示”。

## 4. 完整代码

``` py linenums="1" title="examples/data_efficient_nopt/data_efficient_nopt.py"
--8<--
examples/data_efficient_nopt/data_efficient_nopt.py
--8<--
```

## 5. 结果展示

论文通过在多个PDE基准测试和模拟真实世界场景（如泊松方程、亥姆霍兹方程）中进行大量实证评估，全面展示了所提出框架的优越性。

- 显著提升数据效率 (Significant Improvement in Data Efficiency)：
    - 数据节省：实验结果表明，在无标签PDE数据上进行预训练能够显著减少对高成本标签数据的需求。与从随机初始化开始训练的神经算子相比，预训练后的模型在达到相同性能水平时，所需的标签数据量大大减少。
    - 量化效益：在不同的PDE系统中，该方法可以节省$5×10^2$到$8×10^5$个昂贵的模拟解。这意味着，原本需要数万甚至数十万次高精度模拟才能训练出的模型，现在可能只需要数百或数千次。

- 超越现有通用预训练模型 (Outperforming Existing General Pretrained Models)：
    - 论文还对比了本研究提出的物理领域特定预训练与来自计算机视觉领域的通用预训练模型（如在SSV2数据集上预训练的Video-MAE）。结果发现，即使是这些最先进的通用模型，在有限的PDE模拟数据下进行微调时，性能表现不佳。这强调了在SciML领域进行领域适应性无监督预训练的重要性，因为物理数据具有独特的结构和特性。

- 更好的泛化能力 (Improved Generalization Capability)：
    - 减小泛化差距：无监督预训练显著减小了模型的泛化差距（即测试误差与训练误差之间的差异）。这意味着预训练的模型在未见过的测试数据上表现更稳定、更准确。
    - OOD性能提升：结合情境学习，模型在分布外（OOD）的场景下展现出更强的泛化能力。例如，当物理参数或初始条件超出了训练范围时，模型仍能做出合理的预测。

- 更快的收敛速度 (Faster Convergence Speed)：
    - 无标签PDE数据上的预训练为神经算子提供了领域适应的良好初始化。这种预训练的权重使得模型在后续的监督微调阶段能够更快地收敛。这意味着训练时间缩短，计算资源消耗降低。

- 情境学习的额外泛化优势 (Additional Generalization Advantage from In-Context Learning)：
    - 无需训练开销：情境学习在推理阶段灵活地引入相似示例，且不增加任何训练成本。
    - 持续性能提升：实验证明，通过增加情境示例的数量，神经算子在各种PDE上的OOD泛化能力能够持续得到提升。这为在模型部署后动态提升其在复杂未知情况下的性能提供了可能。尤其是在解决OOD问题时，情境示例能够帮助模型校准输出的量级和模式，使其更接近真实解。

总而言之，这篇论文提出了一种创新且高效的神经算子学习框架，通过无监督预训练在大量廉价的无标签物理数据上学习通用表示，并通过情境学习在推理阶段利用少量相似案例来提升OOD泛化能力。这一框架显著降低了对昂贵模拟数据的需求，并提高了模型在复杂物理问题中的适应性和泛化性，为科学机器学习的数据高效发展开辟了新途径。

## 6. 参考资料

- [Data-Efficient Operator Learning via Unsupervised Pretraining and In-Context Learning](https://arxiv.org/abs/2402.15734)
