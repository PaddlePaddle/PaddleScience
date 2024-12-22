# DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks(DrivAerNet++：一个包含计算流体力学模拟和深度学习基准的大规模多模态汽车数据集)

## 论文信息:
|年份 | 会议 | 作者|引用数 | 论文PDF |
|-----|-----|-----|---|-----|
|2024| Conference and Workshop on Neural Information Processing Systems |Mohamed Elrefaie, Florin Morar, Angela Dai, Faez Ahmed|4|DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks|

## 代码信息
=== "模型训练命令"

~~~bash
``` sh
python DrivAerNetPlusPlus.py
```
~~~

=== "模型评估命令"

~~~bash
``` sh
python DrivAerNetPlusPlus.py mode=eval EVAL.pretrained_model_path=“训练的时候保存的模型权重”
```
~~~

## 1. 背景简介
DrivAerNet++ 是一个大规模多模态的汽车数据集，主要用于数据驱动的空气动力学设计。该数据集包含 8,000 种多样化的汽车设计，通过高保真计算流体动力学 (CFD) 模拟生成。这些汽车设计涵盖了不同的车身类型（如快背型、掀背型和三厢型）、底盘配置和轮胎设计，既代表了传统内燃机汽车（ICE），也包括电动汽车（EV）的设计特征。

每个数据条目包含详细的 3D 网格、参数化模型、空气动力学系数、广泛的流场和表面场数据，以及用于汽车分类的分段部件和点云数据。这些数据支持广泛的机器学习应用，如基于数据的设计优化、生成建模、替代模型训练、CFD 模拟加速和几何分类。

DrivAerNet++ 数据集的亮点包括：

1. **数据规模和多样性**：提供了超过 8,000 个汽车设计，包含丰富的设计参数（26-50 个参数）和多样的形状变化。
2. **多模态数据**：集成了 3D 网格、点云、CFD 数据、参数化数据和部件标注等多种数据模态。
3. **高保真度**：每个汽车设计的 CFD 模拟网格包含约 2,400 万个单元，确保捕捉复杂的气流和湍流特性。
4. **公开资源**：总计 39TB 的工程数据以开放访问的形式提供，为研究社区提供高质量和多样化的训练数据。

该数据集还包括多种任务的机器学习基准测试（如空气阻力系数预测），验证了其在支持汽车设计、CFD 加速和工程设计优化中的潜力。DrivAerNet++ 被认为能够显著推动工程设计和空气动力学领域的研究与创新。

## 2. 问题定义

DrivAerNet++ 提供了一个用于汽车空气动力学性能预测的多模态数据集，任务目标是基于输入数据预测汽车的空气阻力系数（$C_d$）。

##### 输入 ：

输入数据包括以下多模态信息：

1. **点云数据**：汽车外形的三维几何表面表示，记为：   $$   x \in \mathbb{R}^{N \times 3},   $$   其中 $N$ 是点云顶点数量，每个点 $x_i = (x_i, y_i, z_i)$ 表示一个顶点的三维坐标。
2. **设计参数**：参数化汽车设计的向量表示，记为：   $$   \mathbf{p} \in \mathbb{R}^d,   $$   其中：   - $d$ 是设计参数的数量（例如 26-50 个参数），用于描述车身类型、底盘配置、轮胎设计等特征。
3. **附加流场数据（可选）**：   - 表面压力场：$\mathbf{p_s}(x)$。   - 表面速度场：$\mathbf{u}(x) = (u, v, w)$。 不属于本案例，本案例仅涉及空气阻力系数（$C_d$）预测。

##### 任务目标 ：

目标是构建一个深度学习模型 $f(\cdot)$，输入点云数据 $x$ ，预测汽车的空气阻力系数 $C_d$，即：
$$
\hat{C}_d = f(x),
$$
其中： - $\hat{C}_d$ 是模型预测的空气阻力系数。 - $C_d$ 是真实的空气阻力系数。 模型需要学习点云的几何信息的相关性，精准预测阻力系数。

##### 评估指标 ：

模型的性能通过以下指标进行评估：

1. **均方误差（MSE）**：  

$$
\text{MSE} = \frac{1}{m} \sum_{i=1}^m \left( C_d^{(i)} - \hat{C}_d^{(i)} \right)^2,
$$

其中：   - $m$ 是测试样本数量。   - $C_d^{(i)}$ 是真实的阻力系数。   - $\hat{C}_d^{(i)}$ 是模型预测的阻力系数。

2. **平均绝对误差（MAE）**：  

$$
\text{MAE} = \frac{1}{m} \sum_{i=1}^m \left| C_d^{(i)} - \hat{C}_d^{(i)} \right|,
$$



2.  **最大绝对误差（Max AE）**：  

$$
\text{Max AE} = \max \left| C_d^{(i)} - \hat{C}_d^{(i)} \right|,
$$

该指标衡量模型在所有测试样本中的最差预测表现。

4. **决定系数（$R^2$ Score）**：  

$$
R^2 = 1 - \frac{\sum_{i=1}^m \left( C_d^{(i)} - \hat{C}_d^{(i)} \right)^2}{\sum_{i=1}^m \left( C_d^{(i)} - \bar{C}_d \right)^2},  
$$

其中 $\bar{C}_d$ 是真实阻力系数的均值。

任务总结 DrivAerNet++ 的任务目标是通过多模态输入（点云数据、或设计参数）预测汽车空气动力学性能，核心评估指标为 $C_d$ 的预测误差。模型可用于快速评估汽车设计性能并优化空气动力学表现。

## 3. 问题求解

为了预测汽车的空气阻力系数 ($C_d$)，我们采用基于深度学习的回归方法，使用两种点云处理模型 **RegDGCNN** 和 **RegPointNet**，分别从输入数据中提取几何特征并完成回归任务。这些模型能够高效处理 3D 点云数据，并结合设计参数，构建端到端的预测框架。

##### **1. RegDGCNN**

RegDGCNN（Dynamic Graph Convolutional Neural Network for Regression）是一种动态图卷积网络，能够捕获点云数据的局部和全局几何特征。具体包括以下核心步骤：

**动态图构建**：通过 K 近邻算法 (KNN) 动态构建点云的局部图结构。

**图卷积操作**：使用 EdgeConv 对局部邻域的特征进行卷积，提取局部关系。  

**全局特征整合**：通过池化操作将局部特征聚合为全局特征，描述整个汽车的几何属性。  

**输出回归**：将全局特征输入到回归头，预测空气阻力系数 $C_d$。 模型的优点是能够高效捕获点云的局部几何关系，并结合全局上下文特征，适用于处理复杂的 3D 形状。

##### **2. RegPointNet**

RegPointNet 是一种经典的点云处理网络，直接对 3D 点的坐标进行学习，无需显式构建邻域。具体包括以下步骤：  

**局部特征提取**：通过共享多层感知机 (MLP) 对每个点的特征进行提取。

**全局特征聚合**：使用对称函数（如最大池化）将所有点的特征整合为全局特征。

**输出回归**：全局特征通过全连接层映射到空气阻力系数 $C_d$。 模型的优点是结构简单且参数量较少，能够高效处理较小规模的点云数据。

### 3.1 数据集介绍

数据集下载教程参考，从[数据集下载地址](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OYU2FG)下载DrivAerNet++: 3D Meshes，即.stl网格数据。
**Linux：**

1. [Globus Connect Personal](https://www.globus.org/globus-connect-personal)是 Globus 提供的免费客户端。提供 Linux、Mac 和 Windows 版本。

```
下载地址：https://www.globus.org/globus-connect-personal
```

2. 使用 wget 或 curl 直接下载 Globus Connect Personal：

```bash
$ wget https://downloads.globus.org/globus-connect-personal/linux/stable/globusconnectpersonal-latest.tgz
```

3. 从下载的 tarball 中提取文件。

```bash
$ tar xzf globusconnectpersonal-latest.tgz
# 替代 `x.y.z` 为下载的具体版本号
$ cd globusconnectpersonal-x.y.z
```

4. 启动 Globus Connect 个人版。由于第一次运行，因此必须先完成设置，然后才能运行完整的应用程序。

```bash
$ ./globusconnectpersonal
```

5. 设置过程，运行`./globusconnectpersonal`后弹出如下内容，通过登录网址获取认证代码。`== starting endpoint setup`后设置endpoint名字并获取endpoint的ID序号。

```bash
Detected that setup has not run yet, and '-setup' was not used
Will now attempt to run
  globusconnectpersonal -setup

Globus Connect Personal needs you to log in to continue the setup process.

We will display a login URL. Copy it into any browser and log in to get a
single-use code. Return to this command with the code to continue setup.

Login here:
-----
https://auth.globus.org/你的网址内容
-----
Enter the auth code: 你的认证代码
== starting endpoint setup

Input a value for the Endpoint Name: 你设置的endpoint名字
registered new endpoint, id: 你的endpoint的ID
setup completed successfully
```

6. 无GUI运行，后台启动 Globus Connect Personal。

```bash
$ ./globusconnectpersonal -start &
```

7. 查看 Globus Connect Personal 的状态，使用`-status`可以查看后台运行的 Globus Connect Personal 的状态。

```bash
$ ./globusconnectpersonal -status
Globus Online: connected
Transfer Status: idle
```

8. 添加路径Globus下载路径。

```bash
vim ~/.globusonline/lta/config-paths
```

9. 添加存储路径，更多信息可参考Globus官方教程。

```txt
 ~/,0,0
你的路径地址,0,1
```

10. 使用Globus需要安装globus-cli。

```bash
$ pip install globus-cli
```

11. 登录，通过登录网址获取认证代码。

```bash
$ globus login --no-local-server

Please authenticate with Globus here:
------------------------------------
https://auth.globus.org/你的网址信息
------------------------------------

Enter the resulting Authorization Code here: 从网址获取的认证代码

You have successfully logged in to the Globus CLI!

You can check your primary identity with
  globus whoami

For information on which of your identities are in session use
  globus session show

Logout of the Globus CLI with
  globus logout
```

12. 找出要下载数据的名称和账户，以PubDAS为例。

```bash
$ globus endpoint search "PubDAS" --filter-owner-id 4c984b40-a0b2-4d9e-b132-b32                                                                               735905e23@clients.auth.globus.org
ID                                   | Owner                                                        | Display Name
------------------------------------ | ------------------------------------------------------------ | -------------
706e304c-5def-11ec-9b5c-f9dfb1abb183 | 4c984b40-a0b2-4d9e-b132-b32735905e23@clients.auth.globus.org | PubDAS
1013e4a6-5df1-11ec-bded-55fe55c2cfea | 4c984b40-a0b2-4d9e-b132-b32735905e23@clients.auth.globus.org | PubDAS-upload
```

13. 简化下载数据源的ID名称（可选）

```bash
export ep1=706e304c-5def-11ec-9b5c-f9dfb1abb183
```

14. 查看该路径下的数据。

```bash
$ globus ls $ep1:
DAS-Month-02.2023/
FORESEE/
FOSSA/
Fairbanks/
LaFargeConcoMine/
Stanford-1-Campus/
Stanford-2-Sandhill-Road/
Stanford-3-ODH4/
Valencia/
License.txt
```

15. 获取自己Globus的ID。

```bash
$ globus endpoint search "YourName(STEP2)" --filter-owner-id yourname(step1)@globusid.org
ID                                   | Owner                        | Display Name  
------------------------------------ | ---------------------------- | --------------------------
-----------------ID----------------- | yourname(step1)@globusid.org | YourName(STEP2)
```

16. 同理可简化自己的ID（可选）

```bash
export ep2=-----------------ID-----------------
```

17. 下载数据，将PubDAS中的License.txt从ep1数据源传送到ep2（自己路径下，第8，9步设置）。

```bash
# here is defaut path (your home path)
$ globus transfer $ep1:License.txt $ep2:/~/License.txt
Message: The transfer has been accepted and a task has been created and queued for execution
Task ID: -----------------传送任务ID-----------------
```

18. 利用上面的Task ID查看文件传输状态！

```bash
$ globus task show -----------------传送任务ID-----------------
Label:                        None
Task ID:                      -----------------传送任务ID-----------------
Is Paused:                    False
Type:                         TRANSFER
Directories:                  0
Files:                        1
Status:                       SUCCEEDED
Request Time:                 2022-01-24T17:20:07+00:00
Faults:                       0
Total Subtasks:               2
Subtasks Succeeded:           2
Subtasks Pending:             0
Subtasks Retrying:            0
Subtasks Failed:              0
Subtasks Canceled:            0
Subtasks Expired:             0
Subtasks with Skipped Errors: 0
Completion Time:              2022-01-24T17:20:08+00:00
Source Endpoint:              ESnet Read-Only Test DTN at Starlight
Source Endpoint ID:           57218f41-3200-11e8-b907-0ac6873fc732
Destination Endpoint:         Globus Tutorial Endpoint 1
Destination Endpoint ID:      -----------------ID-----------------
Bytes Transferred:            1000000
Bytes Per Second:             587058
```

数据文件说明如下：

|          Dataset          | Size  | $C_d$ | $C_l$ | $u$  | $p$  | $τ_w$ | Wheels/Underbody Modeling | Parametric | Design Parameters | Shape Variation | Experimental Validation |   Modalities   | Open-source |
| :-----------------------: | :---: | :---: | :---: | :--: | :--: | :---: | :-----------------------: | :--------: | :---------------: | :-------------: | :---------------------: | :------------: | :---------: |
|  Usama et. al 2021 [63]   |  500  |   ✔   |   ✘   |  ✘   |  ✘   |   ✘   |             ✘             |     ✔      |      40 (2D)      |        ✘        |            ✘            |       P        |      ✘      |
|   Li et. al 2023 [43]*    |  551  |   ✔   |   ✔   |  ✘   |  ✘   |   ✘   |             ✘             |     ✔      |      6 (3D)       |        ✘        |            ✘            |    M, P, C     |      ✘      |
|  Rios et. al 2021 [58]†   |  600  |   ✔   |   ✔   |  ✘   |  ✘   |   ✘   |             ✘             |     ✘      |         -         |        ✘        |            ✘            |     M, PC      |      ✘      |
|   Li et. al 2023 [43]†    |  611  |   ✔   |   ✔   |  ✘   |  ✘   |   ✘   |             ✘             |     ✘      |         -         |        ✘        |            ✘            |      M, C      |      ✘      |
| Umetani et. al 2018 [67]† |  889  |   ✔   |   ✘   |  ✔   |  ✘   |   ✘   |             ✘             |     ✘      |         -         |        ✘        |            ✘            |      M, C      |      ✔      |
| Gunpinar et. al 2019 [29] | 1,000 |   ✔   |   ✘   |  ✘   |  ✘   |   ✘   |             ✘             |     ✔      |      21 (2D)      |        ✘        |            ✘            |       P        |      ✘      |
|  Jacob et. al 2021 [37]*  | 1,000 |   ✔   |   ✔   |  ✔   |  ✘   |   ✘   |             ✘             |     ✔      |      15 (3D)      |        ✘        |            ✔            |    M, C, P     |      ✘      |
|  Trinh et. al 2024 [66]   | 1,121 |   ✔   |   ✔   |  ✔   |  ✘   |   ✘   |             ✘             |     ✔      |         -         |        ✘        |            ✘            |      M, C      |      ✘      |
| Remelli et. al 2020 [53]  | 1,400 |   ✔   |   ✘   |  ✘   |  ✘   |   ✘   |             ✘             |     ✘      |         -         |        ✘        |            ✘            |      M, C      |      ✘      |
|   Baque et. al 2018 [6]   | 2,000 |   ✔   |   ✘   |  ✘   |  ✘   |   ✘   |             ✘             |     ✔      |      21 (3D)      |        ✘        |            ✘            |       M        |      ✘      |
|  Song et. al 2023 [63]†   | 2,474 |   ✔   |   ✔   |  ✔   |  ✘   |   ✘   |             ✘             |     ✔      |      50 (3D)      |        ✘        |            ✔            |   M, P, C, P   |      ✔      |
|     DrivAerNet [22]*      | 4,000 |   ✔   |   ✔   |  ✔   |  ✔   |   ✘   |             ✔             |     ✔      |      50 (3D)      |        ✔        |            ✔            |  M, PC, C, P   |      ✔      |
|   DrivAerNet++ (Ours)*    | 8,000 |   ✔   |   ✔   |  ✔   |  ✔   |   ✔   |             ✔             |     ✔      |    26-50 (3D)     |        ✔        |            ✔            | M, PC, C, P, A |      ✔      |

**表格说明：**

1. **Dataset**：数据集的名称和出处（引用文献）。

2. **Size**：数据集的规模，指包含的样本数量。

3. **$C_d$ / $C_l$ / $u$ / $p$ / $τ_w$**：

- **$C_d$**：是否包含空气阻力系数。
- **$C_l$**：是否包含升力系数。
- **$u$**：是否包含速度场数据。
- **$p$**：是否包含压力场数据。
- **$τ_w$**：是否包含壁面剪切应力数据。

4. **Wheels/Underbody Modeling**：是否建模了汽车的车轮和底盘。

5. **Parametric**：是否支持参数化设计。

6. **Design Parameters**：设计参数的数量及其维度（2D 或 3D）。

9. **Shape Variation**：是否支持形状变化。

10. **Experimental Validation**：是否提供了实验验证数据。

11. **Modalities**：数据的模态，包括：

- **M**：点云或 3D 几何模型。
- **P**：参数化数据。
- **C**：CFD（计算流体动力学）数据。
- **PC**：流场和压力场数据。
- **A**：实验数据。

12. **Open-source**：数据集是否开源。

**1.数据增强类：`DataAugmentation：**

用于对点云进行随机变换，包括平移、加噪声和随机丢点，以提升模型的泛化能力。

```py
class DataAugmentation:
    """
    Class encapsulating various data augmentation techniques for point clouds.
    """

    @staticmethod
    def translate_pointcloud(
        pointcloud: paddle.Tensor,
        translation_range: Tuple[float, float] = (2.0 / 3.0, 3.0 / 2.0),
    ) -> paddle.Tensor:
        """
        Translates the pointcloud by a random factor within a given range.

        Args:
            pointcloud: The input point cloud as a paddle.Tensor.
            translation_range: A tuple specifying the range for translation factors.

        Returns:
            Translated point cloud as a paddle.Tensor.
        """
        xyz1 = np.random.uniform(
            low=translation_range[0], high=translation_range[1], size=[3]
        )  # 随机生成轴缩放因子
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])  # 随机生成平移偏移
        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype(
            "float32"
        )
        return paddle.to_tensor(data=translated_pointcloud, dtype="float32")

    @staticmethod
    def jitter_pointcloud(
        pointcloud: paddle.Tensor, sigma: float = 0.01, clip: float = 0.02
    ) -> paddle.Tensor:
        """
        Adds Gaussian noise to the pointcloud.

        Args:
            pointcloud: The input point cloud as a paddle.Tensor.
            sigma: Standard deviation of the Gaussian noise.
            clip: Maximum absolute value for noise.

        Returns:
            Jittered point cloud as a paddle.Tensor.
        """
        N, C = tuple(pointcloud.shape)
        jittered_pointcloud = pointcloud + paddle.clip(
            x=sigma * paddle.randn(shape=[N, C]), min=-clip, max=clip
        )  # 加入高斯噪声并限制范围
        return jittered_pointcloud

    @staticmethod
    def drop_points(pointcloud: paddle.Tensor, drop_rate: float = 0.1) -> paddle.Tensor:
        """
        Randomly removes points from the point cloud based on the drop rate.

        Args:
            pointcloud: The input point cloud as a paddle.Tensor.
            drop_rate: The percentage of points to be randomly dropped.

        Returns:
            The point cloud with points dropped as a paddle.Tensor.
        """
        num_drop = int(drop_rate * pointcloud.shape[0])  # 计算需要丢弃的点数
        drop_indices = np.random.choice(pointcloud.shape[0], num_drop, replace=False)
        keep_indices = np.setdiff1d(np.arange(pointcloud.shape[0]), drop_indices)
        dropped_pointcloud = pointcloud[keep_indices, :]  # 保留剩余点
        return dropped_pointcloud
```

**2.数据集类：`DrivAerNetPlusPlusDataset`：**

用于加载 DrivAerNetPlusPlus 数据集，并处理点云数据（如采样、增强和归一化）。

```py
class DrivAerNetPlusPlusDataset(paddle.io.Dataset):
    """
    Paddle Dataset class for the DrivAerNet dataset, handling loading, transforming, and augmenting 3D car models.

    This dataset is designed for tasks involving aerodynamic simulations and deep learning models,
    specifically for predicting aerodynamic coefficients (e.g., Cd values) from 3D car models.

    Examples:
    >>> import ppsci
    >>> dataset = ppsci.data.dataset.DrivAerNetPlusPlusDataset(
    ...     input_keys=("vertices",),
    ...     label_keys=("cd_value",),
    ...     weight_keys=("weight_keys",),
    ...     subset_dir="/path/to/subset_dir",
    ...     ids_file="train_ids.txt",
    ...     root_dir="/path/to/DrivAerNetPlusPlusDataset",
    ...     csv_file="/path/to/aero_metadata.csv",
    ...     num_points=1024,
    ...     transform=None,
    ... )  # doctest: +SKIP
    """

    def __init__(self,
                 input_keys: Tuple[str, ...],
                 label_keys: Tuple[str, ...],
                 weight_keys: Tuple[str, ...],
                 subset_dir: str,
                 ids_file: str,
                 root_dir: str,
                 csv_file: str,
                 num_points: int,
                 transform: Optional[Callable] = None,
                 pointcloud_exist: bool = False):
        """
        Initializes the DrivAerNetDataset instance.

        Args:
            input_keys (Tuple[str, ...]): Tuple of strings specifying the input keys.
                These keys correspond to the features extracted from the dataset,
                typically the 3D vertices of car models.
                Example: ("vertices",)

            label_keys (Tuple[str, ...]): Tuple of strings specifying the label keys.
                These keys correspond to the ground-truth labels, such as aerodynamic
                coefficients (e.g., Cd values).
                Example: ("cd_value",)

            weight_keys (Tuple[str, ...]): Tuple of strings specifying the weight keys.
                These keys represent optional weighting factors used during model training
                to handle class imbalance or sample importance.
                Example: ("weight_keys",)

            subset_dir (str): Path to the directory containing subsets of the dataset.
                This directory is used to divide the dataset into different subsets
                (e.g., train, validation, test) based on provided IDs.

            ids_file (str): Path to the file containing the list of IDs for the subset.
                The file specifies which models belong to the current subset (e.g., training IDs).

            root_dir (str): Root directory containing the 3D STL files of car models.
                Each 3D model is expected to be stored in a file named according to its ID.

            csv_file (str): Path to the CSV file containing metadata for the car models.
                The CSV file includes information such as aerodynamic coefficients,
                and may also map model IDs to specific attributes.

            num_points (int): Number of points to sample or pad each 3D point cloud to.
                If the model has more points than `num_points`, it will be subsampled.
                If it has fewer points, zero-padding will be applied.

            transform (Optional[Callable]): Optional transformation function applied to each sample.
                This can include augmentations like scaling, rotation, or jittering.

            pointcloud_exist (bool): Whether the point clouds are pre-processed and saved as `.pt` files.
                If `True`, the dataset will directly load the pre-saved point clouds
                instead of generating them from STL files.
        """
        super().__init__()
        self.root_dir = root_dir
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.weight_keys = weight_keys
        self.subset_dir = subset_dir
        self.ids_file = ids_file
        try:
            self.data_frame = pd.read_csv(csv_file)
        except Exception as e:
            logging.error(f'Failed to load CSV file: {csv_file}. Error: {e}')
            raise
        self.transform = transform
        self.num_points = num_points
        self.augmentation = DataAugmentation()
        self.pointcloud_exist = pointcloud_exist
        self.cache = {}

        try:
            with open(os.path.join(self.subset_dir, self.ids_file), 'r') as file:
                subset_ids = file.read().split()
            self.data_frame[self.data_frame['Design'].isin(subset_ids)].index.tolist()
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f'Error loading subset file {self.ids_file}: {e}')

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data_frame)
```

**3.数据加载：`DrivAerNetPlusPlusDataset[__getitem__]`：**

返回样本，包括点云、标签和权重。

```py
    def __getitem__(self, idx: int, apply_augmentations: bool = True) -> tuple[
        dict[str, paddle.Tensor], dict[str, paddle.Tensor], dict[str, paddle.Tensor]]:
        """
        Retrieves a sample and its corresponding label from the dataset, with an option to apply augmentations.

        Args:
            idx (int): Index of the sample to retrieve.
            apply_augmentations (bool, optional): Whether to apply data augmentations. Defaults to True.

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]: The sample (point cloud) and its label (Cd value).
        """
        if paddle.is_tensor(idx):
            idx = idx.tolist()

        if idx in self.cache:
            return self.cache[idx]
        while True:
            row = self.data_frame.iloc[idx]
            design_id = row['Design']
            cd_value = row['Average Cd']

            if self.pointcloud_exist:
                vertices = self._load_point_cloud(design_id)

                if vertices is None:
                    # logging.warning(f"Skipping design {design_id} because point cloud is not found or corrupted.")
                    idx = (idx + 1) % len(self.data_frame)
                    continue
            else:
                geometry_path = os.path.join(self.root_dir, f"{design_id}.stl")
                try:
                    mesh = trimesh.load(geometry_path, force='mesh')
                    vertices = paddle.to_tensor(mesh.vertices, dtype=paddle.float32)
                    vertices = self._sample_or_pad_vertices(vertices, self.num_points)
                except Exception as e:
                    logging.error(f"Failed to load STL file: {geometry_path}. Error: {e}")
                    raise

            if apply_augmentations:
                vertices = self.augmentation.translate_pointcloud(vertices.numpy())
                vertices = self.augmentation.jitter_pointcloud(vertices)

            if self.transform:
                vertices = self.transform(vertices)

            point_cloud_normalized = self.min_max_normalize(vertices)
            cd_value = paddle.to_tensor(float(cd_value), dtype=paddle.float32).reshape([-1])

            self.cache[idx] = (point_cloud_normalized, cd_value)
            # return point_cloud_normalized, cd_value
            return (
                {self.input_keys[0]: point_cloud_normalized},
                {self.label_keys[0]: cd_value},
                {self.weight_keys[0]: paddle.to_tensor(1)},
            )
```

### 3.2 模型选择

在本问题中，使用两种模型（RegDGCNN 和 PointNet）对 DrivAerNet 数据集进行学习，以预测输入点云的空气阻力系数（$C_d$）。这两种模型分别在特征提取方法和网络架构设计上有所不同，具体如下：

**输出**：预测的空气阻力系数（$C_d$），作为模型的回归输出。

| 特性         | RegDGCNN                             | PointNet                         |
| ------------ | ------------------------------------ | -------------------------------- |
| 特征学习方式 | 动态构建图结构，捕获局部几何关系     | 无需构建图结构，直接学习顶点特征 |
| 局部特征提取 | 使用 EdgeConv 聚合邻域特征           | 使用共享 MLP 对单点进行编码      |
| 全局特征聚合 | 动态图特征池化                       | 最大池化                         |
| 适用场景     | 复杂几何形状、点云局部关系显著的任务 | 点云分布均匀或较少点数的任务     |

```python
    model = ppsci.arch.RegPointNet(
        input_keys=cfg.MODEL.input_keys,
        label_keys=cfg.MODEL.output_keys,
        weight_keys=cfg.MODEL.weight_keys,
        args=cfg.MODEL)  # 根据自己的需求选择模型，RegDGCNN可参考DrivAerNet的设置。
```

模型参数具体如下：

```yaml
MODEL:
  input_keys: ["vertices"]         # 输入的关键字段，表示3D点云的顶点数据
  output_keys: ["cd_value"]        # 输出的关键字段，表示模型预测的空气阻力系数（C_d）
  weight_keys: ["weight_keys"]     # 权重字段，用于加权数据的损失计算
  dropout: 0.0                     # Dropout率，防止过拟合；此处设置为0.0表示不使用Dropout
  emb_dims: 1024                   # 特征嵌入的维度，控制全局特征的表示能力
  channels: [6, 64, 128, 256, 512, 1024] # 特征通道数，每一层提取的特征维度；通常从低到高逐步增加
  linear_sizes: [128, 64, 32, 16]  # 全连接层的尺寸，表示回归头的逐层神经元数量
  k: 40                            # K近邻数，表示动态图构建时的邻域点数量
  output_channels: 1               # 模型最终输出通道数，此处为1，表示单个输出值（空气阻力系数C_d）
```

### 3.3 约束构建

#### 3.3.1 监督约束

由于我们以监督学习方式进行训练，此处采用监督约束 `SupervisedConstraint`：

```py
train_dataloader_cfg = {
    "dataset": {
        "name": "DrivAerNetPlusPlusDataset",  # 数据集名称，指定为 DrivAerNet++ 数据集
        "root_dir": cfg.ARGS.dataset_path,   # 数据集的根目录路径，从配置文件中获取
        "input_keys": ("vertices",),        # 输入的关键字段，这里是点云顶点数据
        "label_keys": ("cd_value",),        # 输出的关键字段，这里是空气阻力系数 (C_d)
        "weight_keys": ("weight_keys",),    # 权重字段，用于加权样本的损失计算
        "subset_dir": cfg.ARGS.subset_dir,  # 子集目录路径，用于区分训练、验证、测试数据
        "ids_file": cfg.TRAIN.train_ids_file, # 指定包含训练样本 ID 的文件
        "csv_file": cfg.ARGS.aero_coeff,    # 包含空气动力学系数的 CSV 文件路径
        "num_points": cfg.MODEL.num_points  # 每个点云的采样点数，从模型配置中获取
    },
    "batch_size": cfg.TRAIN.batch_size,     # 每个批次的数据样本数，从训练配置中获取
    "sampler": {
        "name": "BatchSampler",            # 批采样器的名称，这里使用批采样器
        "drop_last": False,                # 如果最后一个批次不足 batch_size，不丢弃
        "shuffle": True,                   # 打乱数据样本顺序
    },
    "num_workers": cfg.TRAIN.num_workers,  # 数据加载的并行线程数，从训练配置中获取
}

drivaernetplusplus_constraint = ppsci.constraint.SupervisedConstraint(
    train_dataloader_cfg,                   # 训练数据加载器的配置
    ppsci.loss.MSELoss("mean"),             # 损失函数，使用均方误差 (MSE)，计算方式为均值
    name="DrivAerNetplusplus_constraint",  # 约束的名称，用于标识
)

```

### 3.4 优化器构建

优化器是模型训练中的关键部分，用于通过梯度下降法（或其他算法）调整模型参数。在本场景中，使用了`Adam`和`SGD`优化器，并通过学习率调度器来动态调整学习率。

```py
# 优化器的选择与配置
optimizer = paddle.optimizer.Adam(
    parameters=model.parameters(),          # 模型的待优化参数
    learning_rate=cfg.ARGS.lr,              # 初始学习率，从配置文件中获取
    weight_decay=cfg.ARGS.weight_decay      # 权重衰减系数（L2 正则化），防止过拟合
) if cfg.ARGS.optimizer == 'adam' else paddle.optimizer.SGD(
    parameters=model.parameters(),          # 如果不是 Adam，则使用 SGD 优化器
    learning_rate=cfg.ARGS.lr,              # 初始学习率
    weight_decay=cfg.ARGS.weight_decay      # 权重衰减系数
)

# 学习率调度器配置
tmp_lr = paddle.optimizer.lr.ReduceOnPlateau(
    mode=cfg.TRAIN.scheduler.mode,          # 调度器模式，"min" 或 "max"（监控目标最小值或最大值）
    patience=cfg.TRAIN.scheduler.patience,  # 容忍不改进的训练轮数，超出后减少学习率
    factor=cfg.TRAIN.scheduler.factor,      # 学习率衰减因子，减少为原来的 factor 倍
    verbose=cfg.TRAIN.scheduler.verbose,    # 是否输出学习率调整的日志
    learning_rate=optimizer.get_lr()        # 当前学习率，初始化为优化器的学习率
)

# 将学习率调度器绑定到优化器
optimizer.set_lr_scheduler(tmp_lr)
scheduler = tmp_lr  # 保存调度器对象

```

### 3.5 评估器构建

在训练过程中通常会按一定轮数间隔，用验证集（测试集）评估当前模型的训练情况，因此使用 `ppsci.validate.SupervisedValidator` 构建评估器。

```py
# 验证数据加载器的配置
valid_dataloader_cfg = {
    "dataset": {
        "name": "DrivAerNetPlusPlusDataset",  # 数据集名称，指定为 DrivAerNet++ 数据集
        "root_dir": cfg.ARGS.dataset_path,    # 数据集的根目录路径，从配置文件中获取
        "input_keys": ("vertices",),         # 输入的关键字段，表示点云顶点数据
        "label_keys": ("cd_value",),         # 输出的关键字段，表示空气阻力系数 (C_d)
        "weight_keys": ("weight_keys",),     # 权重字段，用于损失函数的加权计算
        "subset_dir": cfg.ARGS.subset_dir,   # 子集目录路径，用于区分训练、验证、测试数据
        "ids_file": cfg.TRAIN.eval_ids_file, # 验证集样本 ID 的文件路径
        "csv_file": cfg.ARGS.aero_coeff,     # 包含空气动力学系数的 CSV 文件路径
        "num_points": cfg.MODEL.num_points   # 每个点云的采样点数，从模型配置中获取
    },
    "batch_size": cfg.TRAIN.batch_size,      # 验证批次大小，从训练配置中获取
    "sampler": {
        "name": "BatchSampler",             # 批采样器的名称，这里使用批采样器
        "drop_last": False,                 # 如果最后一个批次不足 batch_size，不丢弃
        "shuffle": True,                    # 是否打乱数据顺序，这里启用
    },
    "num_workers": cfg.TRAIN.num_workers,   # 数据加载的并行线程数，从训练配置中获取
}

# 定义验证器
drivaernetplusplus_valid = ppsci.validate.SupervisedValidator(
    valid_dataloader_cfg,                   # 验证数据加载器的配置
    loss=ppsci.loss.MSELoss("mean"),        # 验证损失函数，使用均方误差 (MSE)
    metric={"MSE": ppsci.metric.MSE()},     # 验证指标，设置为均方误差 (MSE)
    name="DrivAerNetplusplus_valid",        # 验证器的名称，用于标识
)

# 将验证器放入字典以便后续使用
validator = {drivaernetplusplus_valid.name: drivaernetplusplus_valid}
```

评价指标 `metric` 选择 `ppsci.metric.MSE` 即可,也可根据需求自己选择其他评估指标。

### 3.6 模型训练、评估

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估。

```py
# 初始化求解器
solver = ppsci.solver.Solver(
    model=model,                              # 模型对象（如 RegDGCNN 或 RegPointNet）
    constraint=constraint,                    # 约束条件，定义训练目标和损失函数
    output_dir=cfg.output_dir,                # 输出目录，用于保存训练结果（如模型权重、日志）
    optimizer=optimizer,                      # 优化器对象，用于训练参数的更新（如 Adam 或 SGD）
    lr_scheduler=scheduler,                   # 学习率调度器，动态调整学习率以提高训练效果
    epochs=cfg.TRAIN.epochs,                  # 总训练轮数，从配置文件中获取
    validator=validator,                      # 验证器，用于在训练过程中评估模型性能
    eval_during_train=cfg.TRAIN.eval_during_train,  # 是否在训练期间进行验证
    eval_with_no_grad=cfg.EVAL.eval_with_no_grad    # 验证时是否使用 `no_grad`，避免梯度计算以节省内存
)

# 开始模型训练
solver.train()  # 运行训练过程，根据约束条件优化模型参数

# 运行模型评估
solver.eval()   # 评估模型在验证集或测试集上的性能

```

## 4. 完整代码

=== "DrivAerNetPlusPlus.py"

```py
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hydra
from omegaconf import DictConfig
from os import path as osp
import paddle
import ppsci
from ppsci.utils import logger


def train(cfg: DictConfig):
    # set seed
    ppsci.utils.misc.set_random_seed(cfg.TRAIN.seed)

    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "train.log"), "info")

    # set model
    model = ppsci.arch.RegPointNet(
        input_keys=cfg.MODEL.input_keys,
        label_keys=cfg.MODEL.output_keys,
        weight_keys=cfg.MODEL.weight_keys,
        args=cfg.MODEL)

    train_dataloader_cfg = {
        "dataset": {
            "name": "DrivAerNetPlusPlusDataset",
            "root_dir": cfg.ARGS.dataset_path,
            "input_keys": ("vertices",),
            "label_keys": ("cd_value",),
            "weight_keys": ("weight_keys",),
            "subset_dir": cfg.ARGS.subset_dir,
            "ids_file": cfg.TRAIN.train_ids_file,
            "csv_file": cfg.ARGS.aero_coeff,
            "num_points": cfg.MODEL.num_points
        },
        "batch_size": cfg.TRAIN.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "num_workers": cfg.TRAIN.num_workers,
    }

    drivaernetplusplus_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        ppsci.loss.MSELoss("mean"),
        name="DrivAerNetplusplus_constraint",
    )

    constraint = {drivaernetplusplus_constraint.name: drivaernetplusplus_constraint}

    valid_dataloader_cfg = {
        "dataset": {
            "name": "DrivAerNetPlusPlusDataset",
            "root_dir": cfg.ARGS.dataset_path,
            "input_keys": ("vertices",),
            "label_keys": ("cd_value",),
            "weight_keys": ("weight_keys",),
            "subset_dir": cfg.ARGS.subset_dir,
            "ids_file": cfg.TRAIN.eval_ids_file,
            "csv_file": cfg.ARGS.aero_coeff,
            "num_points": cfg.MODEL.num_points
        },
        "batch_size": cfg.TRAIN.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "num_workers": cfg.TRAIN.num_workers,
    }

    drivaernetplusplus_valid = ppsci.validate.SupervisedValidator(
        valid_dataloader_cfg,
        loss=ppsci.loss.MSELoss("mean"),
        metric={"MSE": ppsci.metric.MSE()},
        name="DrivAerNetplusplus_valid",
    )

    validator = {drivaernetplusplus_valid.name: drivaernetplusplus_valid}

    # set optimizer
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
                                      learning_rate=cfg.ARGS.lr,
                                      weight_decay=cfg.ARGS.weight_decay) if cfg.ARGS.optimizer == 'adam' else paddle.optimizer.SGD(
        parameters=model.parameters(), learning_rate=cfg.ARGS.lr, weight_decay=cfg.ARGS.weight_decay)

    tmp_lr = paddle.optimizer.lr.ReduceOnPlateau(mode=cfg.TRAIN.scheduler.mode, patience=cfg.TRAIN.scheduler.patience,
                                                 factor=cfg.TRAIN.scheduler.factor, verbose=cfg.TRAIN.scheduler.verbose,
                                                 learning_rate=optimizer.get_lr())
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr

    # initialize solver
    solver = ppsci.solver.Solver(
        model=model,
        constraint=constraint,
        output_dir=cfg.output_dir,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        epochs=cfg.TRAIN.epochs,
        validator=validator,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad
    )

    # train model
    solver.train()

    solver.eval()


def evaluate(cfg: DictConfig):
    # set seed
    ppsci.utils.misc.set_random_seed(cfg.TRAIN.seed)

    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "eval.log"), "info")

    # set model
    model = ppsci.arch.RegPointNet(
        input_keys=cfg.MODEL.input_keys,
        label_keys=cfg.MODEL.output_keys,
        weight_keys=cfg.MODEL.weight_keys,
        args=cfg.MODEL)

    valid_dataloader_cfg = {
        "dataset": {
            "name": "DrivAerNetPlusPlusDataset",
            "root_dir": cfg.ARGS.dataset_path,
            "input_keys": ("vertices",),
            "label_keys": ("cd_value",),
            "weight_keys": ("weight_keys",),
            "subset_dir": cfg.ARGS.subset_dir,
            "ids_file": cfg.EVAL.ids_file,
            "csv_file": cfg.ARGS.aero_coeff,
            "num_points": cfg.MODEL.num_points
        },
        "batch_size": cfg.TRAIN.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "num_workers": cfg.TRAIN.num_workers,
    }

    drivaernetplusplus_valid = ppsci.validate.SupervisedValidator(
        valid_dataloader_cfg,
        loss=ppsci.loss.MSELoss("mean"),
        metric={"MSE": ppsci.metric.MSE()},
        name="DrivAerNetPlusPlus_valid",
    )

    validator = {drivaernetplusplus_valid.name: drivaernetplusplus_valid}

    solver = ppsci.solver.Solver(
        model=model,
        output_dir=cfg.output_dir,
        validator=validator,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad
    )

    # evaluate model
    solver.eval()


@hydra.main(version_base=None, config_path="./conf", config_name="DriveAerNetPlusPlus.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(
            f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()
```

## 5. 结果展示

下方展示实验结果：

|    Model    | $MSE (10^{-5})$ | $MAE (10^{-3})$ | $Max$  $AE (10^{-3})$ | $R^2$  |
| :---------: | :-------------: | :-------------: | :-------------------: | :----: |
|  RegDGCNN   |      14.8       |      9.54       |         13.81         | 0.6310 |
| RegPointNet |      11.2       |      8.28       |         14.9          | 0.7019 |

## 6. 参考

参考代码：https://github.com/Mohamedelrefaie/DrivAerNet/tree/main/DeepSurrogates

参考文献列表

1. [1] Asad Abbas, Ashkan Rafiee, Max Haase, and Andrew Malcolm. Geometrical deep learning for performance prediction of high-speed craft. Ocean Engineering, 258:111716, 2022.
2. [2] S. R. Ahmed, G. Ramm, and G. Faltin. Some salient features of the time -averaged ground vehicle wake. SAE Transactions, 93:473–503, 1984. ISSN 0096736X. URL http://www.jstor.org/stable/ 44434262.
3. [3] Mubashara Akhtar, Omar Benjelloun, Costanza Conforti, Pieter Gijsbers, Joan Giner-Miguelez, Nitisha Jain, Michael Kuchnik, Quentin Lhoest, Pierre Marcenac, Manil Maskey, Peter Mattson, Luis Oala, Pierre Ruyssen, Rajat Shinde, Elena Simperl, Goeffry Thomas, Slava Tykhonov, Joaquin Vanschoren, Jos van der Velde, Steffen Vogler, and Carole-Jean Wu. Croissant: A metadata format for ml-ready datasets. DEEM ’24, page 1–6, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400706110. doi: 10.1145/3650203.3663326. URL https://doi.org/10.1145/3650203.3663326.
4. [4] Neil Ashton, A West, S Lardeau, and Alistair Revell. Assessment of rans and des methods for realistic automotive models. Computers & fluids, 128:1–15, 2016.
5. [5] Neil Ashton, Paul Batten, Andrew Cary, and Kevin Holst. Summary of the 4th high-lift prediction workshop hybrid rans/les technology focus group. Journal of Aircraft, pages 1–30, 2023.
6. [6] Pierre Baque, Edoardo Remelli, Francois Fleuret, and Pascal Fua. Geodesic convolutional shape optimization. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 472–481. PMLR, 10–15 Jul 2018. URL https://proceedings.mlr.press/v80/baque18a.html.
7. [7] Florent Bonnet, Jocelyn Mazari, Paola Cinnella, and Patrick Gallinari. Airfrans: High fidelity computational fluid dynamics dataset for approximating reynolds-averaged navier–stokes solutions. Advances in Neural Information Processing Systems, 35:23463–23478, 2022.
8. [8] Christian Brand, Jillian Anable, Ioanna Ketsopoulou, and Jim Watson. Road to zero or road to nowhere? disrupting transport and energy in a zero carbon world. Energy Policy, 139:111334, 2020.
9. [9] Adam Brandt, Henrik Berg, Michael Bolzon, and Linda Josefsson. The effects of wheel design on the aerodynamic drag of passenger vehicles. SAE International Journal of Advances and Current Practices in Mobility, 1(2019-01-0662):1279–1299, 2019.
10. [10] Leo Breiman. Random forests. Machine learning, 45:5–32, 2001.
11. [11] Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012, 2015.
12. [12] Tianqi Chen and Carlos Guestrin. Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining, pages 785–794, 2016.
13. [13] Wai Tong Chung, Bassem Akoush, Pushan Sharma, Alex Tamkin, Ki Sung Jung, Jacqueline Chen, Jack Guo, Davy Brouzet, Mohsen Talei, Bruno Savard, et al. Turbulence in focus: Benchmarking scaling behavior of 3d volumetric super-resolution with blastnet 2.0 data. Advances in Neural Information Processing Systems, 36, 2024.
14. [14] Adam Cobb, Anirban Roy, Daniel Elenius, Frederick Heim, Brian Swenson, Sydney Whittington, James Walker, Theodore Bapty, Joseph Hite, Karthik Ramani, et al. Aircraftverse: A large-scale multimodal dataset of aerial vehicle designs. Advances in Neural Information Processing Systems, 36:44524–44543, 2023.
15. [15] Blender Online Community. Blender - a 3D modelling and rendering package. Blender Foundation, Stichting Blender Foundation, Amsterdam, 2018. URL http://www.blender.org.
16. [16] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017.
17. [17] Guillaume Damblin, Mathieu Couplet, and Bertrand Iooss. Numerical studies of space-filling designs: optimization of latin hypercube samples and subprojection properties. Journal of Simulation, 7(4):276–289, 2013.
18. [18] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.
19. [19] Open MPI Documentation. mpirun / mpiexec, 2024. URL https://docs.open-mpi.org/en/v5.0.x/ man-openmpi/man1/mpirun.1.html. Accessed: 2024-05-26.
20. [20] Benet Eiximeno, Arnau Miró, Ivette Rodríguez, and Oriol Lehmkuhl. Toward the usage of deep learning surrogate models in ground vehicle aerodynamics. Mathematics, 12(7):998, 2024.
21. [21] Mohamed Elrefaie, Tarek Ayman, Mayar A Elrefaie, Eman Sayed, Mahmoud Ayyad, and Mohamed M AbdelRahman. Surrogate modeling of the aerodynamic performance for airfoils in transonic regime. In AIAA SCITECH 2024 Forum, page 2220, 2024.
22. [22] Mohamed Elrefaie, Angela Dai, and Faez Ahmed. Drivaernet: A parametric car dataset for data-driven aerodynamic design and graph-based drag prediction. arXiv preprint arXiv:2403.08055, 2024.
23. [23] Mohamed Elrefaie, Steffen Hüttig, Mariia Gladkova, Timo Gericke, Daniel Cremers, and Christian Breitsamter. Real-time and on-site aerodynamics using stereoscopic piv and deep optical flow learning. arXiv preprint arXiv:2401.09932, 2024.
24. [24] Nick Erickson, Jonas Mueller, Alexander Shirkov, Hang Zhang, Pedro Larroy, Mu Li, and Alexander Smola. Autogluon-tabular: Robust and accurate automl for structured data. arXiv preprint arXiv:2003.06505, 2020.
25. [25] Matthias Fey and Jan Eric Lenssen. Fast graph representation learning with pytorch geometric. arXiv preprint arXiv:1903.02428, 2019.
26. [26] Jerome H Friedman. Greedy function approximation: a gradient boosting machine. Annals of statistics, pages 1189–1232, 2001.
27. [27] Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé Iii, and Kate Crawford. Datasheets for datasets. Communications of the ACM, 64(12): 86–92, 2021.
28. [28] Christopher Greenshields. OpenFOAM v11 User Guide. The OpenFOAM Foundation, London, UK, 2023. URL https://doc.cfd.direct/openfoam/user-guide-v11.
29. [29] Erkan Gunpinar, Umut Can Coskun, Mustafa Ozsipahi, and Serkan Gunpinar. A generative design and drag coefficient prediction system for sedan car side silhouettes based on computational fluid dynamics. CAD Computer Aided Design, 111:65–79, 6 2019. ISSN 00104485. doi: 10.1016/j.cad.2019.02.003.
30. [30] Sheikh Md Shakeel Hassan, Arthur Feeney, Akash Dhruv, Jihoon Kim, Youngjoon Suh, Jaiyoung Ryu, Yoonjin Won, and Aparna Chandramowlishwaran. Bubbleml: A multiphase multiphysics dataset and benchmarks for machine learning. Advances in Neural Information Processing Systems, 36, 2024.
31. [31] Angelina Heft, Thomas Indinger, and Nikolaus Adams. Investigation of unsteady flow structures in the wake of a realistic generic car model. In 29th AIAA applied aerodynamics conference, page 3669, 2011.
32. [32] Angelina I Heft, Thomas Indinger, and Nikolaus A Adams. Experimental and numerical investigation of the drivaer model. In Fluids Engineering Division Summer Meeting, volume 44755, pages 41–51. American Society of Mechanical Engineers, 2012.
33. [33] Angelina I Heft, Thomas Indinger, and Nikolaus A Adams. Introduction of a new realistic generic car model for aerodynamic investigations. Technical report, SAE Technical Paper, 2012.
34. [34] Wolf-Heinrich Hucho. Aerodynamik des Automobils: eine Brücke von der Strömungsmechanik zur Fahrzeugtechnik. Springer-Verlag, 2013.
35. [35] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning, pages 448–456. pmlr, 2015.
36. [36] M Islam, F Decker, E De Villiers, Aea Jackson, J Gines, T Grahs, A Gitt-Gehrke, and J Comas i Font. Application of detached-eddy simulation for automotive aerodynamics development. Technical report, SAE Technical Paper, 2009.
37. [37] Sam Jacob Jacob, Markus Mrosek, Carsten Othmer, and Harald Köstler. Deep learning for real-time aerodynamic evaluations of arbitrary vehicle shapes. SAE International Journal of Passenger Vehicle Systems, 15(2):77–90, mar 2022. ISSN 2770-3460. doi: https://doi.org/10.4271/15-15-02-0006. URL https://doi.org/10.4271/15-15-02-0006.
38. [38] Max Jaderberg, Karen Simonyan, Andrew Zisserman, et al. Spatial transformer networks. Advances in neural information processing systems, 28, 2015.
39. [39] Ali Kashefi and Tapan Mukerji. Physics-informed pointnet: A deep learning solver for steady-state incompressible flows and thermal fields on multiple sets of irregular geometries. Journal of Computational Physics, 468:111510, 2022.
40. [40] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. Lightgbm: A highly efficient gradient boosting decision tree. Advances in neural information processing systems, 30, 2017.
41. [41] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
42. [42] Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016.
43. [43] Zongyi Li, Nikola Borislavov Kovachki, Chris Choy, Boyi Li, Jean Kossaifi, Shourya Prakash Otta, Mohammad Amin Nabian, Maximilian Stadler, Christian Hundt, Kamyar Azizzadenesheli, and Anima Anandkumar. Geometry-informed neural operator for large-scale 3d pdes, 2023.
44. [44] H Martins, CO Henriques, JR Figueira, CS Silva, and AS Costa. Assessing policy interventions to stimulate the transition of electric vehicle technology in the european union. Socio-Economic Planning Sciences, 87: 101505, 2023.
45. [45] Florian R Menter, Martin Kuntz, Robin Langtry, et al. Ten years of industrial experience with the sst turbulence model. Turbulence, heat and mass transfer, 4(1):625–632, 2003.
46. [46] Peter Mock and Sonsoles Díaz. Pathways to decarbonization: the european passenger car market in the years 2021–2035. communications, 49:847129–848102, 2021.
47. [47] Chair of Aerodynamics and Technical University of Munich Fluid Mechanics. Drivaer model geometry. https://www.epc.ed.tum.de/en/aer/research-groups/automotive/drivaer/ geometry/, 2024. Accessed: 2024-05-21.
48. [48] OpenFOAM Foundation. Meshing with snappyHexMesh, 2023. URL https://www.openfoam.com/ documentation/guides/latest/doc/guide-meshing-snappyhexmesh.html. Accessed: 2024-0605.
49. [49] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.
50. [50] Giancarlo Pavia and Martin Passmore. Characterisation of wake bi-stability for a square-back geometry with rotating wheels. In Progress in Vehicle Aerodynamics and Thermal Management: 11th FKFS Conference, Stuttgart, September 26-27, 2017 11, pages 93–109. Springer, 2018.
51. [51] Songyou Peng, Michael Niemeyer, Lars Mescheder, Marc Pollefeys, and Andreas Geiger. Convolutional occupancy networks. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part III 16, pages 523–540. Springer, 2020.
52. [52] Leif E Peterson. K-nearest neighbor. Scholarpedia, 4(2):1883, 2009.
53. [53] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 652–660, 2017.
54. [54] Peng Qin, Alessio Ricci, and Bert Blocken. Cfd simulation of aerodynamic forces on the drivaer car model: Impact of computational parameters. Journal of Wind Engineering and Industrial Aerodynamics, 248:105711, 2024. ISSN 0167-6105. doi: https://doi.org/10.1016/j.jweia.2024.105711. URL https: //www.sciencedirect.com/science/article/pii/S0167610524000746.
55. [55] Edoardo Remelli, Artem Lukoianov, Stephan Richter, Benoit Guillard, Timur Bagautdinov, Pierre Baque, and Pascal Fua. Meshsdf: Differentiable iso-surface extraction. Advances in Neural Information Processing Systems, 33:22468–22478, 2020. URL https://proceedings.neurips.cc/paper_files/paper/ 2020/file/fe40fb944ee700392ed51bfe84dd4e3d-Paper.pdf.
56. [56] Thiago Rios, Patricia Wollstadt, Bas Van Stein, Thomas Back, Zhao Xu, Bernhard Sendhoff, and Stefan Menzel. Scalability of learning tasks on 3d cae models using point cloud autoencoders. pages 13671374. Institute of Electrical and Electronics Engineers Inc., 12 2019. ISBN 9781728124858. doi: 10.1109/SSCI44817.2019.9002982.
57. [57] Thiago Rios, Bas Van Stein, Thomas Back, Bernhard Sendhoff, and Stefan Menzel. Point2ffd: Learning shape representations of simulation-ready 3d models for engineering design optimization. pages 10241033. Institute of Electrical and Electronics Engineers Inc., 2021. ISBN 9781665426886. doi: 10.1109/ 3DV53792.2021.00110.
58. [58] Thiago Rios, Bas van Stein, Patricia Wollstadt, Thomas Bäck, Bernhard Sendhoff, and Stefan Menzel. Exploiting local geometric features in vehicle design optimization with 3d point cloud autoencoders. In 2021 IEEE Congress on Evolutionary Computation (CEC), pages 514–521, 2021. doi: 10.1109/CEC45853. 2021.9504746.
59. [59] Thomas Schütz. Hucho-Aerodynamik des Automobils: Strömungsmechanik, Wärmetechnik, Fahrdynamik, Komfort. Springer-Verlag, 2013.
60. [60] Shengrong Shen, Tian Han, and Jiachen Pang. Car drag coefficient prediction using long–short term memory neural network and lasso. Measurement, 225:113982, 2024.
61. [61] Binyang Song, Chenyang Yuan, Frank Permenter, Nikos Arechiga, and Faez Ahmed. Surrogate modeling of car drag coefficient with depth and normal renderings. arXiv preprint arXiv:2306.06110, 2023.
62. [62] D Brian Spalding. The numerical computation of turbulent flow. Comp. Methods Appl. Mech. Eng., 3:269, 1974.
63. [63] Guocheng Tao, Chengwei Fan, Wen Wang, Wenjun Guo, and Jiahuan Cui. Multi-fidelity deep learning for aerodynamic shape optimization using convolutional neural network. Physics of Fluids, 36(5), 2024.
64. [64] Nils Thuerey, Konstantin Weißenow, Lukas Prantl, and Xiangyu Hu. Deep learning methods for reynoldsaveraged navier–stokes simulations of airfoil flows. AIAA Journal, 58(1):25–36, 2020.
65. [65] Artur Toshev, Gianluca Galletti, Fabian Fritz, Stefan Adami, and Nikolaus Adams. Lagrangebench: A lagrangian fluid mechanics benchmarking suite. Advances in Neural Information Processing Systems, 36, 2024.
66. [66] Thanh Luan Trinh, Fangge Chen, Takuya Nanri, and Kei Akasaka. 3d super-resolution model for vehicle flow field enrichment. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 5826–5835, 2024.
67. [67] Nobuyuki Umetani and Bernd Bickel. Learning three-dimensional flow for interactive aerodynamic design. ACM Transactions on Graphics, 37, 2018. ISSN 15577368. doi: 10.1145/3197517.3201325.
68. [68] Muhammad Usama, Aqib Arif, Farhan Haris, Shahroz Khan, S. Kamran Afaq, and Shahrukh Rashid. A datadriven interactive system for aerodynamic and user-centred generative vehicle design. In 2021 International Conference on Artificial Intelligence (ICAI), pages 119–127, 2021. doi: 10.1109/ICAI52203.2021.9445243.
69. [69] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9(11), 2008.
70. [70] Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E Sarma, Michael M Bronstein, and Justin M Solomon. Dynamic graph cnn for learning on point clouds. ACM Transactions on Graphics (tog), 38(5):1–12, 2019.
71. [71] Mark D Wilkinson, Michel Dumontier, IJsbrand Jan Aalbersberg, Gabrielle Appleton, Myles Axton, Arie Baak, Niklas Blomberg, Jan-Willem Boiten, Luiz Bonino da Silva Santos, Philip E Bourne, et al. The fair guiding principles for scientific data management and stewardship. Scientific data, 3(1):1–9, 2016.
72. [72] Zhirong Wu, Shuran Song, Aditya Khosla, Fisher Yu, Linguang Zhang, Xiaoou Tang, and Jianxiong Xiao. 3d shapenets: A deep representation for volumetric shapes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1912–1920, 2015.
73. [73] Yu Xiang, Wonhui Kim, Wei Chen, Jingwei Ji, Christopher Choy, Hao Su, Roozbeh Mottaghi, Leonidas Guibas, and Silvio Savarese. Objectnet3d: A large scale database for 3d object recognition. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part VIII 14, pages 160–176. Springer, 2016.
