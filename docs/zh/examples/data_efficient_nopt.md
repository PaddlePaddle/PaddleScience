# DataEfficientNopt

=== "模型训练命令"

    ``` sh
    cd examples/data_efficient_nopt
    # Download possion_64 data from https://drive.google.com/drive/folders/1crIsTZGxZULWhrXkwGDiWF33W6RHxJkf
    # Download helmholtz_64 data from https://drive.google.com/drive/folders/1UjIaF6FsjmN_xlGGSUX-1K2V3EF2Zalw

    # Update the file paths in `cexamples/data_efficient_nopt/config/data_efficient_nopt.yaml`, specify to mode in `train`
    # UPdate the file paths in config/operators_poisson.yaml or config/operators_helmholtz.yaml, specify to `train_path`, `val_path`, `test_path`, `scales_path` and `train_rand_idx_path`

    # possion_64 pretrain, specify as following:
    #   run_name: r0
    #   config: pois-64-pretrain-e1_20_m3
    #   yaml_config: config/operators_poisson.yaml
    python data_efficient_nopt.py

    # helmholtz_64 pretrain, specify as following:
    #   run_name: r0
    #   config: helm-64-pretrain-o1_20_m1
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

    # possion_64 inference, specify as following:
    #   evaluation: config/inference_poisson.yaml
    #   ckpt_path: <ckpt_path>
    python data_efficient_nopt.py
    ```

## 1. 背景简介

data_efficient_nopt旨在提高偏微分方程（PDE）算子学习的数据效率，通过设计无监督预训练方法减少对高成本模拟数据的依赖。利用未标记的PDE数据（无需模拟解），并通过基于物理启发的重建代理任务对神经算子进行预训练。为了提升分布外（OOD）泛化性能，我们进一步引入了一种基于相似性的上下文学习方法，使神经算子能够灵活利用上下文示例，而无需额外的训练成本或设计。在多种PDE上的实验表明，该方法具有高度的数据效率、更强的泛化能力，甚至优于传统的视觉预训练模型。

## 2. 模型原理

这篇论文主要解决使用深度学习方法解决基于偏微分方程（PDEs）的科学问题时的数据效率问题。具体来说，作者指出当前的神经算子（Neural Operators）方法需要大量的高保真PDE数据，这导致了高昂的数值模拟成本。为了减少对这些昂贵数据的依赖，作者提出了一种无监督预训练方法，旨在通过使用无标签的PDE数据来提高模型的数据效率和泛化能力。

论文通过以下方案解决上述提到的问题：

1. 无监督预训练
    - 无标签PDE数据定义：作者定义了无标签的PDE数据，这些数据不包含PDE的解，从而避免了昂贵的数值模拟。
    - 物理启发的代理任务：作者提出了两个基于重构的代理任务，分别是Masked Autoencoder（MAE）和Super-resolution（SR）。MAE通过随机遮蔽部分输入并要求模型重建完整的输入来学习稀疏感知的不变性；SR通过应用高斯滤波器使输入模糊，然后要求模型重建高分辨率的输入来学习分辨率和模糊的不变性。
    - 预训练过程：使用无标签的PDE数据和上述代理任务进行无监督预训练，从而获得更好的初始模型，减少后续监督训练所需的模拟数据量。

2. 上下文学习
    - 相似性挖掘：在推理时，通过计算查询输入与支持示例（demos）的输出距离来找到相似的示例。
    - 聚合预测：对于每个查询的时空位置，找到相似的示例后，通过聚合这些示例的解来生成最终的预测。
    - 方法优势：这种方法在推理时引入了零额外训练成本，而且可以无缝集成到现有的训练管道中，提高了模型在OOD数据上的泛化能力。

## 5. 完整代码

``` py linenums="1" title="examples/data_efficient_nopt/pretrain_basic.py"
--8<--
examples/data_efficient_nopt/pretrain_basic.py
--8<--
```

``` py linenums="1" title="examples/data_efficient_nopt/inference_fno_helmholtz_poisson.py"
--8<--
examples/data_efficient_nopt/inference_fno_helmholtz_poisson.py
--8<--
```

## 6. 结果展示

## 7. 参考资料

- [Data-Efficient Operator Learning via Unsupervised Pretraining and In-Context Learning](https://arxiv.org/abs/2402.15734)
