# DoMINO

=== "模型训练命令"

    ``` sh
    cd examples/domino

    # 1. Download the DrivAer ML dataset using the provided download_aws_dataset.sh script or using the Hugging Face repo(https://huggingface.co/datasets/neashton/drivaerml).
    sh download_aws_dataset.sh

    # 2. Specify the configuration settings in `examples/domino/conf/config.yaml`.

    # 3. Run process_data.py. This will process VTP/VTU files and save them as npy for faster processing in DoMINO datapipe. Modify data_processor key in config file. Additionally, run cache_data.py to save outputs of DoMINO datapipe in the .npy files. The DoMINO datapipe is set up to calculate Signed Distance Field and Nearest Neighbor interpolations on-the-fly during training. Caching will save these as a preprocessing step and should be used in cases where the STL surface meshes are upwards of 30 million cells. The final processed dataset should be divided and saved into 2 directories, for training and validation. Specify these directories in conf/config.yaml.
    # specify mode using `process`, set path to data_processor.output_dir and data_processor.input_dir
    python3 domino.py

    # 4. run train, specify mode using `train`, set path to data.input_dir and data.input_dir_val
    python3 domino.py
    ```

=== "模型评估命令"

    暂无

=== "模型导出命令"

    暂无

=== "模型推理命令"

    ``` sh
    cd examples/domino
    # specify mode using `eval`, and set path to eval.test_path, eval.save_path and eval.checkpoint_name
    python3 domino.py
    ```

## 1. 背景简介

外部空气动力学涉及高雷诺数Navier-Stokes方程求解，传统CFD方法计算成本高昂。神经算子通过端到端映射提升了效率，但面临多尺度耦合建模与长期预测稳定性不足的挑战。Decomposable Multi-scale Iterative Neural Operator（Domino）提出可分解多尺度架构，通过分层特征解耦、迭代残差校正及参数独立编码，显著提升跨尺度流动建模精度与泛化能力。实验显示，其计算速度较CFD快2-3个量级，分离流预测精度较FNO等模型提升约40%，为飞行器设计等工程问题提供高效解决方案。

## 2. 模型原理

DOMINO (Decomposable Multi-scale Iterative Neural Operator)是一种新颖的机器学习模型架构，旨在解决大规模工程仿真代理建模中的挑战。它是一个基于点云的机器学习模型，利用局部几何信息来预测离散点上的流场 。

以下是DOMINO模型的主要原理：

- 全局几何表示学习（Global Geometry Representation）：
    - 模型首先以几何体的三维表面网格作为输入。
    - 在几何体周围构建一个紧密贴合的表面包围盒和一个表示计算域的包围盒。
    - 几何点云的特征（如空间坐标）通过可学习的点卷积核投影到表面包围盒上的N维结构化网格上（分辨率为$m×m×m×f$）。
    - 点卷积核的实现使用了NVIDIA Warp加速的自定义球查询层 。
    - 通过两种方法将几何特征传播到计算域包围盒中：1）学习一组单独的多尺度点卷积核，将几何信息投影到计算域网格上；2）使用包含卷积、池化和反池化层的CNN块，将表面包围盒网格上的特征$G_s$​传播到计算域包围盒网格$G_c$。CNN块会迭代评估。
    - 计算域网格上计算出的$m×m×m×f$特征代表了几何点云的全局编码。此外，还会计算符号距离场（SDF）及其梯度分量，并附加到学习到的特征中，以提供关于几何拓扑的额外信息。

- 局部几何表示（Local Geometry Representation）：
    - 局部几何表示取决于计算域中评估解场的物理位置。
    - 在计算局部几何表示之前，会在计算域中采样一批离散点。
    - 对于批次中每个采样点，在其周围定义一个大小为$l×l×l$的子区域，并计算局部几何编码。
    - 局部编码本质上是全局编码的一个子集，取决于其在计算域中的位置，并通过点卷积计算。
    - 提取的局部特征通过全连接神经网络进一步转换。
    - 这种局部几何表示用于使用聚合网络评估采样点上的解场。

- 聚合网络（Aggregation Network）：
    - 局部几何表示代表了采样点及其邻居的计算模板附近几何和解的学习特征。
    - 计算模板中的每个点都由其在计算域中的物理坐标、这些坐标处的SDF、来自域质心的法向量以及表面法向量（如果点在表面上）表示。
    - 这些输入特征通过一个全连接神经网络（称为基函数神经网络），计算出一个潜在向量，代表计算模板中每个点的这些特征。
    - 每个潜在向量与局部几何编码连接，并通过另一组全连接层，以预测计算模板中每个点上的解向量。
    - 解向量通过逆距离加权方案进行平均，以预测采样点处的最终解向量。
    - 对于每个解变量，都使用聚合网络的一个独立实例，但全局几何编码网络在它们之间是共享的。

DOMINO模型通过这种分解式、多尺度和迭代的方法，能够有效地处理大规模仿真数据，捕捉长距离和短距离的相互作用，并在不牺牲准确性的情况下提供可扩展、准确和可推广的代理模型 。

## 3. 完整代码

``` py linenums="1" title="examples/domino/domino.py"
--8<--
examples/domino/domino.py
--8<--
```

## 4. 结果展示

## 5. 参考资料

- [DoMINO: A Decomposable Multi-scale Iterative Neural Operator for Modeling Large Scale Engineering Simulations](https://arxiv.org/abs/2501.13350)
