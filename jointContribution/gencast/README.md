# GenCast

该目录包含用于推理天气模型[GenCast](https://arxiv.org/abs/2312.15796)的示例代码.

它还提供了预训练模型的权重、归一化统计数据和示例输入数据，这些内容可以在 [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/dm_graphcast) 上获取。

完整的模型训练需要下载 [ERA5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5) 数据集，该数据集可以从 [ECMWF](https://www.ecmwf.int/) 获取。通过 [Weatherbench2 的 ERA5 数据](https://weatherbench2.readthedocs.io/en/latest/data-guide.html#era5) 以 Zarr 格式访问这些数据是最佳选择。

用于操作性微调的数据可以通过 [Weatherbench2 的 HRES 第0帧数据](https://weatherbench2.readthedocs.io/en/latest/data-guide.html#ifs-hres-t-0-analysis) 访问。

请注意，这些数据集可能受制于不同的条款和条件或许可条款。您使用这些第三方材料需遵循相关条款，使用前请确认您能够遵守任何适用的限制或条款和条件。

## 1. 背景简介

天气预报本质上存在不确定性，因此预测可能天气情景的范围对于许多重要决策至关重要，从警告公众危险天气到规划可再生能源的使用。在此，我们介绍了 GenCast，这是一种概率性天气模型，其技能和速度优于世界顶级的中期天气预报——欧洲中期天气预报中心（ECMWF）的集合预报 ENS。与基于数值天气预报（NWP）的传统方法不同，GenCast 是一种机器学习天气预报（MLWP）方法，基于数十年的再分析数据进行训练。GenCast 能够在 8 分钟内生成一个随机的 15 天全球预报集合，以 12 小时为步长，0.25 度的纬度-经度分辨率，覆盖 80 多个地表和大气变量。在我们评估的 1320 个目标中，GenCast 在 97.4% 上表现优于 ENS，并能更好地预测极端天气、热带气旋和风力发电。该工作帮助开启了操作性天气预报的下一个篇章，使依赖天气的重要决策能够以更高的准确性和效率做出。

## 2. 模型原理

在这里，我们介绍了一种概率性天气模型——GenCast，它以0.25°的分辨率生成全球15天的集合预报，首次实现了比顶级操作性集合系统ECMWF的ENS更高的准确性。在云TPUv5设备上生成一个单一的15天GenCast预报大约需要8分钟，可以并行生成多个预报集合。

GenCast 模型化了未来天气状态 \(X^{t+1}\) 的条件概率分布 \(p(X^{t+1} | X^t, X^{t-1})\)，这个分布是基于当前和之前的天气状态的条件来进行的。长度为 \(T\) 的预报轨迹 \(X^{1:T}\) 是通过对初始和之前状态 \((X^0, X^{-1})\) 进行条件化来建模的，并对连续状态的联合分布进行分解：

\[
p(X^{1:T} | X^0, X^{-1}) = \prod_{t=0}^{T-1} p(X^{t+1} | X^t, X^{t-1})
\]

每个状态都是通过自回归采样得出的。

全球天气状态 \(X\) 的表示包括6个地表变量和13个垂直压力层上的6个大气变量，分布在0.25°的纬度-经度网格上（详见表B1）。预报时长为15天，连续步骤 \(t\) 和 \(t+1\) 之间的间隔为12小时，因此 \(T = 30\)。

GenCast 实现为一个条件扩散模型，这是一种生成式机器学习模型，用于从给定数据分布生成新样本，这为自然图像、声音和视频建模的许多最新进展提供了支持，被称为“生成式 AI”。扩散模型通过迭代细化的过程运行。未来的大气状态 \(X^{t+1}\) 是通过迭代细化候选状态 \(Z_0^{t+1}\) 产生的，该状态纯粹从噪声初始化，并以之前的两个大气状态 \((X^t, X^{t-1})\) 为条件。图中的蓝色框显示了第一个预报步骤如何从初始条件生成，以及整个轨迹 \(X^{1:T}\) 如何通过自回归生成。由于预报中的每个时间步都是用噪声（即 \(Z_0^{t+1}\)）初始化的，因此可以用不同的噪声样本重复该过程，以生成轨迹集合。

<figure markdown>
  ![gencast.png](https://paddle-org.bj.bcebos.com/paddlescience/docs/gencast/gencast.png){ loading=lazy }
  <figcaption> GenCast模型示意图。</figcaption>
</figure>

在迭代细化过程的每个阶段，GenCast 应用一个由编码器、处理器和解码器组成的神经网络架构。编码器组件将输入 \(Z_n^{t+1}\) 和条件 \((X^t, X^{t-1})\) 从原始的纬度-经度网格映射到六次细化的二十面体网格上的内部学习表示。处理器组件是一个Graph Transformer，其中每个节点关注其在内部网格上的k跳邻居。解码器组件将内部网格表示映射回 \(Z_{n+1}^{t+1}\)，其定义在纬度-经度网格上。

GenCast 在40年的ERA5再分析数据上进行训练，时间范围从1979年到2018年，使用标准的扩散模型去噪目标。重要的是，尽管只在单步预测任务上直接训练GenCast，但它可以通过自回归展开来生成15天的集合预报。

## 3. 快速运行

### 3.1 环境依赖

* paddlepaddle
* matpoltlib （用于图像绘制）
* pickle （用于存储和加载图模板）
* xarray （用于加载.nc数据）
* trimesh （用于制作mesh数据）
* scipy （用于球谐变换过程中的稀疏矩阵操作）
* math （用于球谐变换过程中的数学计算）

### 3.2 数据准备

在 [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/dm_graphcast) 上获取相关数据，并将之放到`gencast.yaml`文件中数据配置的路径下。

### 3.3 模型评估命令

    ``` sh
    # linux
    wget -nc https://paddle-org.bj.bcebos.com/paddlescience/models/gencast/gencast_params_GenCast-1p0deg-Mini-_2019.pdparams -P ./data/params/

    python run_gencast.py
    ```

## 4. 参考资料

* [GenCast: Diffusion-based ensemble forecasting for medium-range weather](https://arxiv.org/abs/2312.15796)
* [GenCast Github地址](https://github.com/deepmind/graphcast)
