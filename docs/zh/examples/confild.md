# AI辅助的时空湍流生成：条件神经场潜在扩散模型（CoNFILD）

Distributed under a Creative Commons Attribution license 4.0 (CC BY).

## 1. 背景简介
### 1.1 论文信息
| 年份           | 期刊                | 作者                                                                                             | 引用数 | 论文PDF与补充材料                                                                                   |
|----------------|---------------------|--------------------------------------------------------------------------------------------------|--------|----------------------------------------------------------------------------------------------------|
| 2024年1月3日   | Nature Communications | Pan Du, Meet Hemant Parikh, Xiantao Fan, Xin-Yang Liu, Jian-Xun Wang                             | 15     | [论文链接](https://doi.org/10.1038/s41467-024-54712-1) <br> [代码仓库](https://github.com/jx-wang-s-group/CoNFILD) |

### 1.2 作者介绍
- **通讯作者**：Jian-Xun Wang（王建勋）<br> 所属机构：美国圣母大学航空航天与机械工程系、康奈尔大学机械与航空航天工程系<br> 研究方向：湍流建模、生成式AI、物理信息机器学习<br>

- **其他作者**：<br> Pan Du、Meet Hemant Parikh（共同一作）：圣母大学博士生，研究方向为生成式模型与计算流体力学<br> Xiantao Fan、Xin-Yang Liu：圣母大学研究助理，负责数值模拟与数据生成

### 1.3 模型&复现代码
| 问题类型               | 在线运行                                                                                                                   | 神经网络架构           | 预训练模型                                                                 | 评估指标              |
|------------------------|----------------------------------------------------------------------------------------------------------------------------|------------------------|----------------------------------------------------------------------------|-----------------------|
| 时空湍流生成           | [GitHub代码库](https://github.com/jx-wang-s-group/CoNFILD)                                                                 | 条件神经场+潜在扩散模型 | [模型参数](https://zenodo.org/record/14037782)                             | MSE: 0.041（速度场） |

=== "模型训练命令"
```bash
git clone https://github.com/jx-wang-s-group/CoNFILD
cd CoNFILD
python train.py --config configs/channel_flow.yaml
```

=== "预训练模型快速评估"

``` sh
python catheter.py mode=eval EVAL.pretrained_model=https://dataset.bj.bcebos.com/PaddleScience/2024%20AI-aided%20geometric%20design%20of%20anti-infection%20catheters/result_GeoFNO.pdparams
```

## 2. 问题定义
### 2.1 研究背景
湍流模拟在航空航天、海洋工程等领域至关重要，但传统方法如直接数值模拟（DNS）和大涡模拟（LES）计算成本高昂，难以应用于高雷诺数或实时场景。现有深度学习模型多基于确定性框架，难以捕捉湍流的混沌特性，且在复杂几何域中表现受限。

### 2.2 核心挑战
1. **高维数据**：三维时空湍流数据维度高达 \(O(10^9)\)，传统生成模型内存需求巨大。
2. **随机性建模**：需同时捕捉湍流的多尺度统计特性与瞬时动态。
3. **几何适应性**：需支持不规则计算域与自适应网格。

### 2.3 创新方法
提出**条件神经场潜在扩散模型（CoNFILD）**，通过三阶段框架解决上述挑战：
1. **神经场编码**：将高维流场压缩为低维潜在表示，压缩比达0.002%-0.017%。
2. **潜在扩散**：在潜在空间进行概率扩散过程，学习湍流统计分布。
3. **零样本条件生成**：结合贝叶斯推理，无需重新训练即可实现传感器重建、超分辨率等任务。

![图1 CoNFILD框架](https://via.placeholder.com/600x300?text=CoNFILD+Architecture)
*框架示意图：CNF编码器将流场映射到潜在空间，扩散模型生成新潜在样本，解码器重建物理场*

## 3. 模型构建艱苦拉薩
### 3.1 条件神经场（CNF）
- **架构**：基于SIREN网络，采用正弦激活函数捕捉周期性特征。
- **数学表示**：
  $$
  \mathscr{E}(\mathbf{X},\mathbf{L}) = \text{SIREN}(\mathbf{x}) + \text{FILM}(\mathbf{L})
  $$
  其中FILM（Feature-wise Linear Modulation）通过潜在向量\(\mathbf{L}\)调节每层偏置。

### 3.2 潜在扩散模型
- **前向过程**：逐步添加高斯噪声，潜在表示\(\mathbf{z}_0 \rightarrow \mathbf{z}_T\)。
- **逆向过程**：训练U-Net预测噪声，通过迭代去噪生成新样本：
  $$
  \mathbf{z}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{z}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(\mathbf{z}_t, t) \right) + \sigma_t \epsilon
  $$

### 3.3 零样本条件生成
- **贝叶斯后验采样**：基于稀疏观测\(\Psi\)，通过梯度修正潜在空间采样：
  $$
  \nabla_{\mathbf{z}_t} \log p(\mathbf{z}_t|\Psi) \approx \nabla_{\mathbf{z}_t} \log p(\Psi|\mathbf{z}_t) + \nabla_{\mathbf{z}_t} \log p(\mathbf{z}_t)
  $$

## 4. 实验结果
### 4.1 无条件生成
| 测试案例                | 速度场MSE | 压力场MSE | 湍动能谱误差 |
|-------------------------|-----------|-----------|--------------|
| 二维不规则管道流        | 0.041     | 0.032     | 2.1%         |
| 三维通道流（Re=180）    | 0.028     | 0.025     | 1.8%         |
| 周期性山丘流（Re=2800） | 0.055     | 0.048     | 3.5%         |

![图2 三维通道流生成对比](https://via.placeholder.com/600x300?text=3D+Channel+Flow+Results)
*左：DNS参考；右：CoNFILD生成样本，显示速度云图与涡结构（Q准则等值面）*

### 4.2 条件生成应用
1. **传感器重建**：仅需0.1%网格点观测，重建全场误差降低80%。
2. **数据修复**：修复50%缺失区域，压力场相关系数达0.93。
3. **超分辨率**：从4×1低分辨率输入恢复400×100高分辨率场，湍流谱匹配度超过双三次插值。