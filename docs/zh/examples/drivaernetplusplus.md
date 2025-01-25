# DrivAerNet++

DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks(DrivAerNet++：一个包含计算流体力学模拟和深度学习基准的大规模多模态汽车数据集)

## 论文信息
| 年份 | 会议                                                         | 作者                                                   | 引用数 | 论文PDF                                                      |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| 2024 | Conference and Workshop on Neural Information Processing Systems | Mohamed Elrefaie, Florin Morar, Angela Dai, Faez Ahmed | 4      | DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks |

## 代码信息
|                          预训练模型                          |  神经网络   |   指标    |
| :----------------------------------------------------------: | :---------: | :-------: |
| [DragPrediction_DrivAerNet_PointNet_r2_batchsize16_200epochs_100kpoints_tsne_NeurIPS_best_model.pdparams](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/DragPrediction_DrivAerNet_PointNet_r2_batchsize16_200epochs_100kpoints_tsne_NeurIPS_best_model.pdparams) | RegPointNet | $R^2:92%$ |

=== "模型训练命令"

    ``` sh
    python drivaernetplusplus.py
    ```

=== "模型评估命令"

    ``` sh
    python drivaernetplusplus.py mode=eval EVAL.pretrained_model_path=https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/DragPrediction_DrivAerNet_PointNet_r2_batchsize16_200epochs_100kpoints_tsne_NeurIPS_best_model.pdparams
    ```

## 修正
### 关于 R² 计算的正确性讨论

在机器学习和深度学习中，R²（决定系数）是一种常用的回归模型评估指标。它衡量的是模型预测值与真实值之间的拟合程度。R² 的计算公式为：

$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
$$

其中：
- $SS_{res}$ 是残差平方和，表示预测值与真实值之间的差异。
- $SS_{tot}$  是总平方和，表示真实值与其均值之间的差异。


#### 问题：基于批次（batch）计算 R² 的不合理性

在某些实现中，R² 可能是通过每个批次（batch）单独计算的，然后将这些批次的结果汇总。这种做法在数学上是不合理的，原因如下：

1. **批次内方差 vs. 全局方差**：
   - 当你在每个批次内计算 R² 时，$SS{tot}$是基于该批次内的真实值均值计算的。这会导致$SS{tot}$ 只反映了批次内的方差，而忽略了整个数据集的全局方差。
   - 由于不同批次的数据分布可能不同，批次内的均值和方差可能会有很大差异，导致 R² 的估计不准确。

2. **批次大小的影响**：
   - 如果使用不同的批次大小（batch size），R² 的结果可能会显著不同。例如，较小的批次可能会导致更大的方差，从而影响 R² 的计算。因此，基于批次的 R² 计算结果依赖于批次大小的选择，缺乏稳定性。

```python  
    # 源代码DeepSurrogates/train_RegPointNet.py中的R²计算
    def r2_score(output, target):
    """Compute R-squared score."""
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

    with torch.no_grad():
        for data, targets in test_dataloader:
            start_time = time.time()  # Start time for inference

            data, targets = data.to(device), targets.to(device).squeeze()

            data = data.permute(0, 2, 1)
            outputs = model(data)

            end_time = time.time()  # End time for inference
            inference_time = end_time - start_time
            total_inference_time += inference_time  # Accumulate total inference time
            # print(outputs)
            # print(targets)
            mse = F.mse_loss(outputs, targets)  # Mean Squared Error (MSE)
            mae = F.l1_loss(outputs, targets)  # Mean Absolute Error (MAE),
            r2 = r2_score(outputs, targets)  # R-squared

            # Accumulate metrics to compute averages later
            total_mse += mse.item()
            total_mae += mae.item()
            total_r2 += r2.item()
            max_mae = max(max_mae, mae.item())
            total_samples += targets.size(0)  # Increment total sample count

    print(total_mse)
    print(total_mae)
    print(len(test_dataloader))
    # Compute average metrics over the entire test set
    avg_mse = total_mse / len(test_dataloader)
    avg_mae = total_mae / len(test_dataloader)
    avg_r2 = total_r2 / len(test_dataloader)
```

#### 正确的做法：基于所有数据计算 R²

为了确保 R² 的计算结果是准确且稳定的，应该基于整个数据集（而不是单个批次）来计算 R²。具体步骤如下：

1. **收集所有预测值和真实值**：
   - 在训练或验证过程中，将所有批次的预测值和真实值收集起来，形成一个完整的预测值列表 $y_{pred}$ 和真实值列表 $y_{true}$。

2. **计算全局均值**：
   - 使用整个数据集的真实值 $y_{true}$ 来计算全局均值 $y_{mean}$，而不是每个批次内的局部均值。

3. **计算全局的 $SS{tot}$ 和 $SS{res}$**：
   - 使用全局均值 $y_{mean}$ 来计算总平方和 $SS{tot}$ 和残差平方和 $SS{res}$。

4. **计算最终的 R²**：
   - 使用全局的 $SS{tot}$ 和 $SS{res}$ 来计算最终的 R²。


## 1. 背景简介
本研究展示了DrivAerNet + +，这是目前最大、最全面的用于气动汽车设计的多模态数据集。DrivAerNet + +包括8000种不同的汽车设计，采用高保真的计算流体动力学( CFD )模拟进行建模。该数据集包括不同的汽车配置，如快背式、切角背式和地产背式，具有不同的底盘和车轮设计，以代表内燃机和电动汽车。数据集中的每个入口都具有详细的三维网格、参数化模型、气动力系数和广泛的流场和表面场数据，以及用于汽车分类的分割零件和点云数据。该数据集支持广泛的机器学习应用，包括数据驱动的设计优化、生成式建模、代理模型训练、CFD模拟加速和几何分类。DrivAerNet + +拥有超过39TB的公开可用工程数据，填补了可用资源的重大缺口，提供了高质量、多样化的数据，以增强模型训练，促进泛化，加速汽车设计过程。除了严格的数据集验证，本研究还在气动阻力预测任务上提供了ML基准测试结果，展示了本研究数据集支持的应用广度。该数据集将通过促进创新和提高空气动力学评估的保真度，对汽车设计和更广泛的工程学科产生重大影响。

汽车设计是一个复杂和迭代的过程，需要设计师和工程师之间的密切合作，设计师专注于美学，工程师确保设计满足性能约束。其中一个关键的挑战是在美学吸引力和空气动力学效率之间取得平衡，这直接影响了燃料消耗。随着内燃机汽车( ICE )油耗法规的日益严格和纯电动汽车( BEV ) [ 46、8、44]续航里程要求的提高，保证高效的汽车空气动力学性能变得至关重要。因此，人们对开发用于汽车空气动力学建模的机器学习方法产生了极大的兴趣。

数据驱动方法可以大大缩短获得性能估计之前所需的过程，通常包括生成3D网格，确保水密性和模拟就绪，执行CFD网格划分，定义求解器和边界条件，运行CFD和后处理结果。通过简化这些步骤，数据驱动的方法提高了效率，加快了设计过程。这使得设计人员能够通过实时、准确的性能估计来探索各种想法，最终以更大的设计自由度提高成果。几何深度学习方法[ 57、56、55、1、39、66、61、6]的最新进展表明，它们能够快速地从CFD中估计性能值，从而促进交互式设计修改。然而，由于缺乏公开数据集，这些方法往往局限于简单问题，限制了其更广泛的适用性。

现有的数据集通常集中于较简单的2D案例[ 7、64、21、68、39、29]或简化的3D模型[ 6、55、43、67、58、61、66]，通常不包括关键的部件，如车轮、镜子和底体。正如[ 32 ]所强调的那样，包括这些元素显著影响气动性能，导致阻力显著增加。在CFD模拟中，阻力值增加了约142 %，在风洞实验中，阻力值增加了120 %。这突出了全面的三维建模在实现准确的气动评估方面的关键作用。此外，大约25 %的乘用车气动阻力直接或间接地归因于车轮[ 9 ]。此外，许多大尺度数据集[ 61、67 ]缺乏物理风洞试验对基线模拟的试验验证，以及对每次模拟的个体收敛性的验证。

公开的、大规模的、多模态的汽车数据集显著缺乏，阻碍了数据驱动设计的进展。这与其他领域不同，标准化的数据集如ImageNet [ 18 ]，ObjectNet3D [ 73 ]，ModelNet [ 72 ]和ScanNet [ 16 ]推动了显著的进步。

本研究使用DrivAerNet + + 2数据集(见图1)来解决这些挑战。DrivAerNet + +数据集代表了对其前身DrivAerNet数据集[ 22 ]的重大进步，该数据集集成了4000种不同的汽车形状。这种改进使数据集的体积增加一倍，总共有8000个工业标准的汽车设计，并显著地提高了仿真的逼真度，具有更复杂的单元结构( 24M细胞,与原始数据集的8 - 16M相反)。此外，DrivAerNet + +通过纳入详细的三维流场数据、参数化数据、气动性能系数和部分注释来扩展其实用性。该数据集包含了广泛的几何形状和配置，涵盖了大多数传统汽车的设计类别，包括传统ICE汽车的详细底盘和电动汽车的光滑底盘。

大规模、多样化和高保真度的数据集对于推进CFD和工程设计的深度学习方法至关重要，为新方法的开发、验证和比较提供了标准化的数据。新兴的数据集，如AirfRANS [ 7 ]，BubbleML [ 30 ]，Lagrangebench [ 65 ]和BLASTNet [ 13 ]，通过为训练和基准测试提供全面的数据，对流体力学中的机器学习社区做出了重大贡献。在工程设计中，航空器设计数据集(如Aircraft Verse [ 14 ] )提供了详细多样的航空器设计配置，帮助工程师验证新的设计策略并确保高性能。然而，目前还没有大规模的三维外形数据集，可以将高保真的CFD模拟与专门为汽车气动设计量身定制的工程设计相结合。

在表1中提出的比较通过强调缺乏包含数据驱动气动设计的全面特征的开源数据集来支持本研究的动机。这一差距强调了数据集的必要性，不仅要提供高保真度的模拟，还要确保实验验证，以确认计算模型的准确性和可靠性。DrivAerNet + +通过包含多种数据模态( 3D网格、点云、CFD数据、参数化数据和部分注释)来解决这些需求，并考虑了旋转车轮和下车体的建模。虽然DrivAerNet [ 22 ]基于单一的汽车类别，但DrivAerNet + +融合了多种汽车设计和类别。

|          Dataset          | Size | Aerodynamics Data | Wheels/Underbody Modeling | Parametric Design Parameters | Shape Variation | Experimental Validation | Modalities  | Open-source |
| :-----------------------: | :--: | :---------------: | :-----------------------: | :--------------------------: | :-------------: | :---------------------: | :---------: | :---------: |
|  Usama et. al 2021 [68]   | 500  |       ✅❌❌❌❌       |             ❌             |              ✅               |     40 (2D)     |            ❌            |      P      |      ❌      |
|   Li et. al 2023 [43]*    | 551  |       ✅❌❌✅✅       |             ❌             |              ✅               |     6 (3D)      |            ❌            |    M,P,C    |      ❌      |
|  Rios et. al 2021 [58]†   | 600  |       ✅✅❌❌❌       |             ❌             |              ❌               |        -        |            ❌            |    M,P,C    |      ❌      |
|   Li et. al 2023 [43]†    | 611  |       ✅❌❌✅✅       |             ❌             |              ❌               |        -        |            ✅            |     M,C     |      ❌      |
| Umetani et. al 2018 [67]† | 889  |       ✅❌✅✅❌       |             ❌             |              ❌               |        -        |            ✅            |     M,C     |      ✅      |
| Gunpinar et. al 2019 [29] | 1000 |       ✅❌❌❌❌       |             ❌             |              ✅               |     21 (2D)     |            ❌            |      P      |      ❌      |
|  Jacob et. al 2021 [37]*  | 1000 |       ✅✅✅❌✅       |             ✅             |              ✅               |     15 (3D)     |            ✅            |    M,C,P    |      ❌      |
|  Trinh et. al 2024 [66]   | 1121 |       ❌❌✅✅❌       |             ❌             |              ❌               |        -        |            ❌            |     M,C     |      ❌      |
| Remelli et. al 2020 [53]† | 1400 |       ❌❌❌✅❌       |             ❌             |              ❌               |        -        |            ✅            |     M,C     |      ❌      |
|   Baque el al. 2018 [6]   | 2000 |       ✅❌❌❌❌       |             ❌             |              ✅               |     21 (3D)     |            ❌            |     M,P     |      ❌      |
|  Song et. al 2023 [63]†   | 2474 |       ✅❌❌❌❌       |             ❌             |              ❌               |        -        |            ✅            |      M      |      ✅      |
|     DrivAerNet [22]*      | 4000 |       ✅✅✅✅✅       |             ✅             |              ✅               |     50 (3D)     |            ✅            |   M,P,C,P   |      ✅      |
|   DrivAerNet++ (Ours)*    | 8000 |       ✅✅✅✅✅       |             ✅             |              ✅               |   26-50 (3D)    |            ✅            | M,P,C,C,P,A |      ✅      |

表1：对各种空气动力学数据集进行了全面的比较，突出了数据集大小，包括空气动力学系数( $C_d$、$C_l$)、速度( $u$ )、压力($p$ )、流体剪切力( $τw$ )场、车轮/车体建模、参数研究能力、设计参数数量、形状变化、实验验证数据、多模态性和开源可用性等关键方面。M指3D网格，PC指点云，P指参数化数据，A指部分注释，C指CFD数据。数据集基于ShapeNet [ 11 ]。

除此之外，本研究还描述了先前数据集的局限性：

- 缺乏多样性：来自[ 37、43、22、6、20、60]的数据集基于相同的参数化模型，导致生成的汽车来自相同的汽车设计。这种多样性的缺乏限制了设计探索中的泛化能力和创造力。

- 数据集规模较小：在工程设计过程中，变化并不局限于简单的几何参数调整，往往涉及增加或删除整个组件。一个显著的限制是CFD模拟所需的高质量、水密网格的可用性。现有的大多数数据集[ 43、56、67、20]要么是基于ShapeNet [ 11 ]，其中包含非常少的适用于CFD的汽车设计，并且与学术界或工业界通常用于汽车气动设计的网格相比，具有低分辨率的特征；要么是基于单个设计的变形几何，如Ahmed车身[ 2 ]或DrivAer车身。因此，大多数现有的数据集相对较小，通常在数百个数量级，最大的是[ 61 ]，有2474辆汽车。

- 较低的模拟保真度：由于运行高保真度CFD模拟的昂贵计算成本，在数据集大小和模拟保真度之间存在权衡。因此，现有的数据集，如[ 61、6、55、56、68、29]，使用的模拟保真度较低，降低了实际效用。

本研究的数据集DrivAerNet + +试图同时提供设计变化和多样性，以及模拟逼真度，使其高度适用于概念设计阶段。这种平衡保证了设计人员可以在不牺牲模拟质量的前提下探索广泛的空气动力学概念。

## 2. 问题定义

#### 2.1 数据集呈现

stl源数据集下载教程参考，从[数据集下载地址](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OYU2FG)下载DrivAerNet++: 3D Meshes，即.stl网格数据。
**Linux：**

1. [Globus Connect Personal](https://www.globus.org/globus-connect-personal)是 Globus 提供的免费客户端。提供 Linux、Mac 和 Windows 版本。

    ``` sh
    下载地址：https://www.globus.org/globus-connect-personal
    ```

2. 使用 wget 或 curl 直接下载 Globus Connect Personal：

    ``` sh
    $ wget https://downloads.globus.org/globus-connect-personal/linux/stable/globusconnectpersonal-latest.tgz
    ```

3. 从下载的 tarball 中提取文件。

    ``` sh
    $ tar xzf globusconnectpersonal-latest.tgz
    # 替代 `x.y.z` 为下载的具体版本号
    $ cd globusconnectpersonal-x.y.z
    ```

4. 启动 Globus Connect 个人版。由于第一次运行，因此必须先完成设置，然后才能运行完整的应用程序。

    ``` sh
    $ ./globusconnectpersonal
    ```

5. 设置过程，运行`./globusconnectpersonal`后弹出如下内容，通过登录网址获取认证代码。`== starting endpoint setup`后设置endpoint名字并获取endpoint的ID序号。

    ``` sh
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

    ``` sh
    $ ./globusconnectpersonal -start &
    ```

7. 查看 Globus Connect Personal 的状态，使用`-status`可以查看后台运行的 Globus Connect Personal 的状态。

    ``` sh
    $ ./globusconnectpersonal -status
    Globus Online: connected
    Transfer Status: idle
    ```

8. 添加路径Globus下载路径。

    ``` sh
    vim ~/.globusonline/lta/config-paths
    ```

9. 添加存储路径，更多信息可参考Globus官方教程。

    ``` sh
     ~/,0,0
    你的路径地址,0,1
    ```

10. 使用Globus需要安装globus-cli。

    ``` sh
    $ pip install globus-cli
    ```

11. 登录，通过登录网址获取认证代码。

    ``` sh
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

    ``` sh
    $ globus endpoint search "PubDAS" --filter-owner-id 4c984b40-a0b2-4d9e-b132-b32                                                                               735905e23@clients.auth.globus.org
    ID                                   | Owner                                                        | Display Name
    ------------------------------------ | ------------------------------------------------------------ | -------------
    706e304c-5def-11ec-9b5c-f9dfb1abb183 | 4c984b40-a0b2-4d9e-b132-b32735905e23@clients.auth.globus.org | PubDAS
    1013e4a6-5df1-11ec-bded-55fe55c2cfea | 4c984b40-a0b2-4d9e-b132-b32735905e23@clients.auth.globus.org | PubDAS-upload
    ```

13. 简化下载数据源的ID名称（可选）

    ``` sh
    export ep1=706e304c-5def-11ec-9b5c-f9dfb1abb183
    ```

14. 查看该路径下的数据。

    ``` sh
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

    ``` sh
    $ globus endpoint search "YourName(STEP2)" --filter-owner-id yourname(step1)@globusid.org
    ID                                   | Owner                        | Display Name  
    ------------------------------------ | ---------------------------- | --------------------------
    -----------------ID----------------- | yourname(step1)@globusid.org | YourName(STEP2)
    ```

16. 同理可简化自己的ID（可选）

    ``` sh
    export ep2=-----------------ID-----------------
    ```

17. 下载数据，将PubDAS中的License.txt从ep1数据源传送到ep2（自己路径下，第8，9步设置）。

    ``` sh
    # here is defaut path (your home path)
    $ globus transfer $ep1:License.txt $ep2:/~/License.txt
    Message: The transfer has been accepted and a task has been created and queued for execution
    Task ID: -----------------传送任务ID-----------------
    ```

18. 利用上面的Task ID查看文件传输状态！

    ``` sh
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

飞桨版数据下载：

    ``` sh
    wget https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/DrivAer%2B%2B_Points.tar
    wget https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/DrivAerNetPlusPlus_Drag_8k.csv
    wget https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/test_design_ids.txt
    wget https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/train_design_ids.txt
    wget https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/val_design_ids.txt
    mkdir data/subset_dir
    mv train_design_ids.txt data/subset_dir
    mv val_design_ids.txt data/subset_dir
    mv test_design_ids.txt data/subset_dir
    tar -xvf DrivAer++_Points.tar -C data/
    mv data/workspace/gino_data/14_DrivAer++/paddle_tensor data/DrivAerNetPlusPlus_Processed_Point_Clouds_100k_paddle
    rm -rf data/workspace
    mv DrivAerNetPlusPlus_Drag_8k.csv data/
    ```

**基准几何体生成：**在汽车空气动力学中，根据汽车后端[ 33、34 ]处的气流形态，通常将生产汽车分为三大类：后备箱型、快背型和凹槽型汽车。为了确保本研究的数据集涵盖了大多数传统汽车设计的整个设计空间，本研究基于DrivAer模型[ 47 ]创建了具有不同设计的多个参数化模型。这包括不同的后部构型- -快背、后掠和凹口- -导致不同的尾流结构和流场形态。此外，本研究还改变了车轮，包括开放和封闭的设计，以及平滑和详细的选项。对于汽车底座，本研究既包括典型的ICE汽车的详细底座，也包括适用于电动汽车(见图2)的平滑底座。通过探索各种后部、车轮和下车体构型，本研究旨在提供对其气动影响的全面理解，从而支持开发更鲁棒和可泛化的深度学习模型。对于参数化模型的创建，本研究利用商业软件ANSA ®定义了26个几何参数，允许本研究对这些参数化模型进行变形，从而得到一个大规模的3D汽车数据集。本研究的目标是开发一个过程生成器，以创建拓扑有效的汽车设计，确保每个设计都满足CFD求解器评估的必要要求和汽车设计师的可用性。

![fig2](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/fig/fig2.jpg)

图2：导出DrivAerNet + +的参数化模型的基线模型，展示了一系列的形状设计和配置。变化包括后备箱、快背和凹口车车身类型以及不同的底盘配置，如平滑和详细。轮式选项以封闭、开放、详细、流畅的风格呈现。

**高分辨率3D行业标准设计：**本研究的设计策划包括选择有效的汽车配置，然后进行详细的CFD模拟，以评估空气动力学性能。本研究的目标是创建一个平衡的数据集，包含各种各样的汽车设计，确保覆盖不同的空气动力学性能指标和美学考虑。图3展示了用于生成DrivAerNet + +数据集的设计参数子集。通过为每个参数定义一个下界和上界，并对基准参数模型进行变形，本研究确保了适合工程应用和模拟的全面和定义良好的表示。本研究的设计方法是多样性保持的，确保优化不会导致过于相似的设计。为了实现这一点，本研究采用了最优拉丁超立方采样进行实验设计( DoE )。具体来说，本研究采用增强型随机进化算法( ESE ) [ 17 ]来保证设计空间的高效采样。使用这些步骤，DrivAerNet + +的多样性明显高于DrivAerNet [ 22 ]。

![fig3](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/fig/fig3.jpg)

图3：DrivAerNet + +数据集生成的设计参数。选取了几个对气动力有显著影响的几何参数，并在特定范围内变化。这些参数范围的选择是为了避免难以制造或不美观的值。汽车草图改编自[ 32 ]。

**CFD网格生成：**对于网格生成，本研究使用了开源的SnappyHexMesh工具[ 48 ]。遵循[ 31、54、32 ]的最佳网格划分方法，本研究确保了本研究的网格能够准确地模拟边界层相互作用。每个网格共有2400万个单元格，其中500 ~ 750k个单元格专门用于汽车表面，确保对车身和车轮进行详细的网格划分，以准确捕捉必要的空气动力学现象。作为比较，在DrivAerNet [ 22 ]和DrivAerNet + +之后，由[ 61 ]引入的最大的数据集，它有2474个汽车设计，每个CFD模拟使用了大约200万个单元。关于啮合过程和验证的所有技术细节都在附录中提供。

**自动高保真CFD模拟：**本研究使用开源软件OpenFOAM v11 [ 28 ]，基于Menter的公式[ 45 ]，使用k - ω SST湍流模型进行稳态不可压缩模拟。本研究对生成的几何体进行了质量检查，以确保它们在CFD域内模拟准备和正确对齐，然后对CFD网格划分进行质量检查，最后检查以确保每个CFD模拟的收敛性。总的来说，本研究在最近发布的DrivAerNet数据集[ 22 ]中生成了4，000个额外的模拟，包括各种设计、流动行为、湍流和分离现象。

**计算成本运行：**DrivAerNet + +的高保真CFD仿真需要大量的计算资源。模拟在MIT Supercloud上进行，通过60个节点的并行化，总共2880个CPU核心，每个CFD案例使用256个核心和1000 GB的内存。整个数据集需要39TB的存储空间，CFD模拟的文件总数为834332。作业并行化采用MPI进行管理[ 19 ]，保证了计算任务的高效分配。仿真耗时约3 × 106个CPU小时。

**数据集结构：**本研究的数据集表示使用多种模态的汽车设计，以确保在各种应用中的全面覆盖和易用性。

- 3D汽车设计：本研究为工程设计、CFD分析、设计优化和生成式AI应用提供了理想的3D STL网格。

- ANSA / 3D参数化模型：提供使研究人员能够生成自己的数据集，并根据特定研究目标的需要加入自定义参数。

- 表格参数化数据：每个3D汽车几何参数化有26个参数，这些参数完整地描述了设计。该数据可用于参数回归、分类和设计特征重要性分析。

- 气动性能数据：包括阻力$C_d$、升力(总$C_l$ ,后$C_{l,r}$ ,前$C_{l,f}$)和力矩$C_m$值等力系数，包括均值和标准差。

- CFD数据：包括原始和后处理数据，包含速度和压力的三维全流场信息，包含压力和壁面剪切应力的表面场，车身周围的流线和2D切片。

- 点云：在5k，10k，100k，250k和500k节点上对汽车表面均匀采样的表示。

- 注释标签：对于每辆汽车的不同汽车部件的汽车设计和分割，都适用于目标检测、语义分割、参数研究和自动CFD网格划分等任务。

图4显示了不同设计构型和类别之间气动性能的比较分析，突出了本研究数据集的多样性和规模。

![fig4](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/fig/fig4.jpg)

图4：最上面一行的散点图显示了不同构型的$C_d$和$C_l$之间的关系：第一幅图显示了底盘构型的影响，比较了电动汽车中常用的详细和光滑的底盘。第二个情节突出了设计美学和风格跨越汽车类别(客货两用汽车、有长坡度车顶的汽车和Estateback)的影响。第三个方案考察了不同车轮构型的影响，强调了小的几何修改对空气动力学的重要性。下方的密度图显示了相同构型下$C_d$的分布，提供了这些设计元素和类别如何影响气动效率的详细视图。

本研究使用克鲁瓦桑格式[ 3 ]为DrivAerNet + +数据集提供详细的元数据，以确保文档的全面性和研究社区的易用性。本研究还包括数据集的数据表[ 27 ]，该数据集是在Creative Commons AttributionNonCommercial ( CC BY-NC)许可证下提供的。DrivAerNet + +将托管在哈佛数据逆向仓库上，以确保最佳的可访问性和系统的数据管理。由于39TB的数据可能会对数据共享和访问带来挑战，本研究还提供了针对不同任务定制的数据集子集，其中包含了详细的元数据，以方便可用性。

#### 2.2 基准设置

在本文中，本研究探讨了各种机器学习任务，特别关注气动阻力( $C_d$ )的代理建模(回归)。这项研究与众不同，因为它是第一个使用大规模、多样化和高保真度数据集对多样化模型进行测试的。虽然先前的研究[ 22、37、43、6、67、68]通常将比较局限于单一的汽车设计或类别，但本研究的方法通过使用全面和公开的数据集，实现了模型之间的公平和广义比较，显示了在汽车空气动力学中的实际应用和性能。

**度量和可视化：**本研究使用几个度量来评估不同模型的性能：均方误差( MSE )量化了预测值和实际值之间的平均平方差异，使其对较大的误差敏感。平均绝对误差( MAE )衡量误差的平均大小，受异常值的影响较小。最大绝对误差( Max AE )识别最大的预测误差，表示最差情况下的准确性。决定系数( R2 Score )表示模型解释的方差比例，其值为1表示完美拟合。较低的MSE和MAE值，以及较高的$R^2$分数和较低的Max AE，表明预测更准确。对于$C_d$的估计，与风洞测量值相比，MAE小于0.005被认为是可接受的[ 59、37 ]。

DrivAerNet++ 提供了一个用于汽车空气动力学性能预测的多模态数据集，任务目标是基于输入数据预测汽车的空气阻力系数（$C_d$）。

##### 输入

输入数据包括以下多模态信息：

1. **点云数据**：汽车外形的三维几何表面表示，记为：   $x \in \mathbb{R}^{N \times 3},$  其中 $N$ 是点云顶点数量，每个点 $x_i = (x_i, y_i, z_i)$ 表示一个顶点的三维坐标。
2. **设计参数**：参数化汽车设计的向量表示，记为：   $\mathbf{p} \in \mathbb{R}^d,$  其中：   - $d$ 是设计参数的数量（例如 26-50 个参数），用于描述车身类型、底盘配置、轮胎设计等特征。
3. **附加流场数据（可选）**：   - 表面压力场：$\mathbf{p_s}(x)$。   - 表面速度场：$\mathbf{u}(x) = (u, v, w)$。 不属于本案例，本案例仅涉及空气阻力系数（$C_d$）预测。

##### 任务目标

目标是构建一个深度学习模型 $f(\cdot)$，输入点云数据 $x$ ，预测汽车的空气阻力系数 $C_d$，即：

$$
\hat{C}_d = f(x),
$$

其中： - $\hat{C}_d$ 是模型预测的空气阻力系数。 - $C_d$ 是真实的空气阻力系数。 模型需要学习点云的几何信息的相关性，精准预测阻力系数。

##### 评估指标

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

#### 3.1 气动阻力的代理模型

在汽车设计的概念阶段和初始阶段，气动阻力系数是一个关键的指标，因为它表明了设计效率并影响了续驶里程。因此，准确、快速的阻力估算在设计过程中至关重要。在这一部分，本研究介绍了两种气动阻力预测的方法：第一，使用三维网格的三维几何深度学习，第二，使用基于参数化数据集的自动机器学习。

##### 3.1.1 基于几何深度学习的三维网格气动阻力预测

为了预测汽车的空气阻力系数 ($C_d$)，本研究采用基于深度学习的回归方法，使用两种点云处理模型 **RegDGCNN** 和 **RegPointNet**，分别从输入数据中提取几何特征并完成回归任务。这些模型能够高效处理 3D 点云数据，并结合设计参数，构建端到端的预测框架。

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

在这里，本研究测试了在PyTorch [ 49 ]和PyTorch Geometric [ 25 ]中实现的不同几何深度学习模型(点网络、GCNN 、RegDGCNN )，用于气动阻力的代理模型建模任务，突出了数据集多样性和缩放的重要性。具体来说，本研究使用不同的表示来训练模型，包括基于图的模型和基于点云的模型。与以往研究[ 37、20、43、6]不同的是，在一个汽车设计(快车)上训练模型，在另一个实验中，在所有设计(有长坡度车顶的汽车、客货两用汽车和Estateback)上训练模型。首先，本研究在DrivAerNet数据集[ 22 ]上训练深度学习模型，该数据集包括带有详细底座、开轮和镜子的快背板的变体。该数据集包含4，000个汽车设计( 2800个用于训练,大约600个用于验证, 600个用于测试)，结果如表2所示。然后，在DrivAerNet + +数据集上训练和测试相同的模型，该数据集包含8，000个(有长坡度车顶的汽车、带后背、客货两用汽车、流畅细致的底座、不同的车轮配置)广泛变化的汽车设计，分为5，600个用于训练，1，200个用于验证，1，200个用于测试，结果如表3所示。

|     Model     | MSE ($×10^{-5}$) | MAE ($×10^{-3}$) | Max AE ($×10^{-3}$) | $R^2$ | Training Time | Inference Time | Number of Parameters |
| :-----------: | :--------------: | :--------------: | :-----------------: | :---: | :-----------: | :------------: | :------------------: |
| PointNet [53] |       12.0       |       8.85       |        10.18        | 0.826 |    0.5hrs     |     0.51s      |      2,348,097       |
|   GCNN [44]   |       10.7       |       7.17       |        10.97        | 0.874 |    10.4hrs    |     20.71s     |       100,481        |
| RegDGCNN [22] |       8.01       |       6.91       |        8.80         | 0.901 |    3.2hrs     |     0.52s      |      3,164,257       |

表2：在包含600个汽车设计的DrivAerNet [ 22 ]数据集(采用开轮、带反射镜和详细的底架结构的快背式设计)的测试集上对用于气动阻力预测的深度学习模型进行对比分析。

|     Model     | MSE ($×10^{-5}$) | MAE ($×10^{-3}$) | Max AE ($×10^{-3}$) | $R^2$ | Training Time | Inference Time | Number of Parameters |
| :-----------: | :--------------: | :--------------: | :-----------------: | :---: | :-----------: | :------------: | :------------------: |
| PointNet [53] |       14.9       |       9.60       |        12.45        | 0.643 |    2.06hrs    |     0.84s      |      2,348,097       |
|   GCNN [42]   |       17.1       |      10.43       |        15.03        | 0.596 |     49hrs     |     50.8s      |       100,481        |
| RegDGCNN [22] |       14.2       |       9.31       |        12.79        | 0.641 |    12.6hrs    |     0.85s      |      3,164,257       |

表3：深度学习模型在DrivAerNet + + ( All car )包含1200款汽车设计的测试集上进行气动阻力预测的对比分析。

DrivAerNet + + (见图4)中的形状变化带来了额外的挑战。例如，将详细的下车体替换为光滑的下车体可以改变相同汽车设计的阻力分布。将车轮由开式改为闭式可以略微影响拖曳力。此外，不同的尾部构型会导致不同的流场分离行为，从而引起阻力值的显著变化。这些因素使得DrivAerNet + +对于泛化来说是一个非常具有挑战性的任务，因为看似微小的变化会显著地影响拖拽值，并为深度学习模型准确地学习这些变化的特征带来困难。

##### 3.1.2 基于表格参数化数据的气动阻力预测

本研究还探讨了利用参数化数据进行气动阻力预测的任务。为此，本研究采用贝叶斯超参数调优的Auto ML (自动化机器学习)框架[ 24 ]，以及Gradient Boosting [ 26 ]、XGBoost [ 12 ]、LightGBM [ 40 ]、Random Forests [ 10 ]等模型。这些方法利用设计参数估算气动阻力，不需要详细的三维几何结构。这种方法对于有效地评估几何修改对阻力和汽车整体性能的影响是非常有价值的。与三维网格修改相比，参数化数据的使用提供了显著的优势，因为它具有可访问性和易操作性。工程师可以快速调整设计参数，并立即观察对气动性能的影响，从而简化设计过程。

本研究在单个参数化汽车设计(快车)上训练模型，在另一个实验中，在所有参数化模型(有长坡度车顶的汽车、客货两用汽车和Estateback)上训练模型，以探索模型在不同设计中的泛化能力。对于这两个实验，本研究将数据集分成80 %用于训练，20 %用于测试。然后，本研究进一步将训练集按训练部分的20 %，40 %，60 %，80 %和100 %划分子集。

为了标准化参数研究，本研究关注26个参数，而不是50个参数，因为DrivAerNet [ 22 ]的50个几何参数模型仅基于一个汽车类别，特别是带有详细底座和开放车轮的快背车。结果如图5所示，AutoGluon在单个快背类别上表现较好，而LightGBM在组合数据集上表现较好。尽管如此，所有模型在组合数据集上的性能都有所下降。一个有意义的发现是，对于所有的模型，无论是单个类别还是组合类别，扩大数据集的大小都会带来性能的提升。例如，通过将训练集大小从640增加到3200，XGBoost的R2值从大约0.35增加到0.55。

![fig5](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/fig/fig5.jpg)

图5：基于参数化数据的不同车型的阻力系数预测。图中显示了R2得分的中位数和95 %置信区间作为训练数据百分比的函数。

**1.数据增强类：`DataAugmentation：**

用于对点云进行随机变换，包括平移、加噪声和随机丢点，以提升模型的泛化能力。

``` py linenums="15"
--8<--
ppsci/arch/regpointnet.py:15:78
--8<--
```

**2.数据集类：`DrivAerNetPlusPlusDataset`：**

用于加载 DrivAerNetPlusPlus 数据集，并处理点云数据（如采样、增强和归一化）。

``` py linenums="37"
--8<--
ppsci/data/dataset/drivaernetplusplus_dataset.py:37:266
--8<--
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

``` py linenums="35"
--8<--
examples/drivaernetplusplus/drivaernetplusplus.py:35:57
--8<--

```

### 3.4 优化器构建

优化器是模型训练中的关键部分，用于通过梯度下降法（或其他算法）调整模型参数。在本场景中，使用了`Adam`和`SGD`优化器，并通过学习率调度器来动态调整学习率。

``` py linenums="84"
--8<--
examples/drivaernetplusplus/drivaernetplusplus.py:84:107
--8<--
```

### 3.5 评估器构建

在训练过程中通常会按一定轮数间隔，用验证集（测试集）评估当前模型的训练情况，因此使用 `ppsci.validate.SupervisedValidator` 构建评估器。

``` py linenums="59"
--8<--
examples/drivaernetplusplus/drivaernetplusplus.py:59:82
--8<--
```

评价指标 `metric` 选择 `ppsci.metric.MSE` 即可,也可根据需求自己选择其他评估指标。

### 3.6 模型训练、评估

完成上述设置之后，只需要将上述实例化的对象按顺序传递给 `ppsci.solver.Solver`，然后启动训练、评估。

``` py linenums="109"
--8<--
examples/drivaernetplusplus/drivaernetplusplus.py:109:125
--8<--
```

## 4. 完整代码
``` py linenums="16"
--8<--
examples/drivaernetplusplus/drivaernetplusplus.py:16:199
--8<--
```

## 5. 结果展示

#### 5.1 局限性和未来工作

DrivAerNet + +数据集包含8000个汽车设计，涵盖了大多数传统汽车设计。然而，模拟的逼真度略低于工业上通常使用的( CFD网格为O ( 100M ))单元[ 4 ] )。此外，当使用稳态RANS模拟时，3D汽车周围的高度湍流和随时间变化的流动引入了误差。虽然k - ω - SST模型[ 45 ]提供了准确的结果，但它很难预测流动分离和再附[ 36、4 ]。未来的工作应采用混合RANS - LES方法，以更好地捕捉流场的时间依赖性。

此外，本研究训练的代理模型不够复杂，不足以学习复杂的几何和气动特征。更高级的模型，如几何信息神经算子[ 43 ]，卷积占位网络[ 51 ]和复杂的图模型，应该进行测试。本研究主要关注阻力的代理模型，因为它是初始设计阶段最关键的因素。然而，利用DrivAerNet + +进行其他任务，如加速CFD模拟，对于更全面的方法是必要的。

为了增强数据集，未来的工作将侧重于集成瞬态CFD模拟，并结合额外的模态，如二维图像绘制和多模态学习方法。这将提高模型的准确性和鲁棒性，推动汽车设计和优化的创新。

#### 5.2 结论与飞桨版结果

在本文中，本研究介绍了DrivAerNet + +，这是一个最大的、用于数据驱动的气动设计的多模态三维数据集，它包含了高保真的CFD模拟和各种汽车设计。本研究的数据集包括8000辆基于行业标准形状的汽车，提供了各种气动性能指标的广泛覆盖。该数据集需要39TB的存储量，比工程上可比的数据集要大得多，且公开可用。此外，生成DrivAerNet + +的计算成本比最近发表的CFD数据集[ 43 ]大一个数量级，该数据集使用了185，744个CPU小时，而本研究的数据集需要300万CPU小时。

该数据集支持广泛的机器学习任务，包括气动性能的代理模型建模、CFD模拟的加速、数据驱动的设计优化、生成式人工智能、形状和零件分类以及三维形状重建。本研究还展示了第一个对标结果，证明了几何深度学习模型和AutoML框架对阻力系数预测的有效性。此外，本研究还探讨了在不同类型的汽车中建立拖曳力代理模型的广义模型的挑战。虽然在单个汽车类别上训练的模型表现良好，但当应用于完全多样化的数据集时，它们的性能在R2方面从0.82 - 0.9显著降低到0.6。这强调了在不同设计之间实现稳健性能的复杂性。本研究的数据集可用于内燃机( ICE )汽车和电动汽车的数据驱动设计，涵盖美学/风格、气动效率和性能等主要设计方面。本研究相信该数据集将作为推进工程设计和CFD研究的基石，为开发更准确、更高效的预测模型提供丰富的资源。

下方展示实验结果：

|    Model    | **$MSE (10^{-5})$** | **$MAE (10^{-3})$** | **$Max$  $AE (10^{-3})$** | **$R^2$** | 备注 |
| :---------: | :-----------------: | :-----------------: | :-----------------------: | :-------: | :--: |
| RegPointNet |        11.2         |        8.10         |           14.9            |  0.9201   | BS16 |

## 6. 参考

参考代码：https://github.com/Mohamedelrefaie/DrivAerNet/tree/main/DeepSurrogates

参考文献列表

[1] Asad Abbas, Ashkan Rafiee, Max Haase, and Andrew Malcolm. Geometrical deep learning for performance prediction of high-speed craft. Ocean Engineering, 258:111716, 2022.

[2] S. R. Ahmed, G. Ramm, and G. Faltin. Some salient features of the time -averaged ground vehicle wake. SAE Transactions, 93:473–503, 1984. ISSN 0096736X. URL http://www.jstor.org/stable/ 44434262.

[3] Mubashara Akhtar, Omar Benjelloun, Costanza Conforti, Pieter Gijsbers, Joan Giner-Miguelez, Nitisha Jain, Michael Kuchnik, Quentin Lhoest, Pierre Marcenac, Manil Maskey, Peter Mattson, Luis Oala, Pierre Ruyssen, Rajat Shinde, Elena Simperl, Goeffry Thomas, Slava Tykhonov, Joaquin Vanschoren, Jos van der Velde, Steffen Vogler, and Carole-Jean Wu. Croissant: A metadata format for ml-ready datasets. DEEM ’24, page 1–6, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400706110. doi: 10.1145/3650203.3663326. URL https://doi.org/10.1145/3650203.3663326.

[4] Neil Ashton, A West, S Lardeau, and Alistair Revell. Assessment of rans and des methods for realistic automotive models. Computers & fluids, 128:1–15, 2016.

[5] Neil Ashton, Paul Batten, Andrew Cary, and Kevin Holst. Summary of the 4th high-lift prediction workshop hybrid rans/les technology focus group. Journal of Aircraft, pages 1–30, 2023.

[6] Pierre Baque, Edoardo Remelli, Francois Fleuret, and Pascal Fua. Geodesic convolutional shape optimization. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 472–481. PMLR, 10–15 Jul 2018. URL https://proceedings.mlr.press/v80/baque18a.html.

[7] Florent Bonnet, Jocelyn Mazari, Paola Cinnella, and Patrick Gallinari. Airfrans: High fidelity computational fluid dynamics dataset for approximating reynolds-averaged navier–stokes solutions. Advances in Neural Information Processing Systems, 35:23463–23478, 2022.

[8] Christian Brand, Jillian Anable, Ioanna Ketsopoulou, and Jim Watson. Road to zero or road to nowhere? disrupting transport and energy in a zero carbon world. Energy Policy, 139:111334, 2020.

[9] Adam Brandt, Henrik Berg, Michael Bolzon, and Linda Josefsson. The effects of wheel design on the aerodynamic drag of passenger vehicles. SAE International Journal of Advances and Current Practices in Mobility, 1(2019-01-0662):1279–1299, 2019.

[10] Leo Breiman. Random forests. Machine learning, 45:5–32, 2001.

[11] Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012, 2015.

[12] Tianqi Chen and Carlos Guestrin. Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining, pages 785–794, 2016.

[13] Wai Tong Chung, Bassem Akoush, Pushan Sharma, Alex Tamkin, Ki Sung Jung, Jacqueline Chen, Jack Guo, Davy Brouzet, Mohsen Talei, Bruno Savard, et al. Turbulence in focus: Benchmarking scaling behavior of 3d volumetric super-resolution with blastnet 2.0 data. Advances in Neural Information Processing Systems, 36, 2024.

[14] Adam Cobb, Anirban Roy, Daniel Elenius, Frederick Heim, Brian Swenson, Sydney Whittington, James Walker, Theodore Bapty, Joseph Hite, Karthik Ramani, et al. Aircraftverse: A large-scale multimodal dataset of aerial vehicle designs. Advances in Neural Information Processing Systems, 36:44524–44543, 2023.

[15] Blender Online Community. Blender - a 3D modelling and rendering package. Blender Foundation, Stichting Blender Foundation, Amsterdam, 2018. URL http://www.blender.org.

[16] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017.

[17] Guillaume Damblin, Mathieu Couplet, and Bertrand Iooss. Numerical studies of space-filling designs: optimization of latin hypercube samples and subprojection properties. Journal of Simulation, 7(4):276–289, 2013.

[18] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

[19] Open MPI Documentation. mpirun / mpiexec, 2024. URL https://docs.open-mpi.org/en/v5.0.x/ man-openmpi/man1/mpirun.1.html. Accessed: 2024-05-26.

[20] Benet Eiximeno, Arnau Miró, Ivette Rodríguez, and Oriol Lehmkuhl. Toward the usage of deep learning surrogate models in ground vehicle aerodynamics. Mathematics, 12(7):998, 2024.

[21] Mohamed Elrefaie, Tarek Ayman, Mayar A Elrefaie, Eman Sayed, Mahmoud Ayyad, and Mohamed M AbdelRahman. Surrogate modeling of the aerodynamic performance for airfoils in transonic regime. In AIAA SCITECH 2024 Forum, page 2220, 2024.

[22] Mohamed Elrefaie, Angela Dai, and Faez Ahmed. Drivaernet: A parametric car dataset for data-driven aerodynamic design and graph-based drag prediction. arXiv preprint arXiv:2403.08055, 2024.

[23] Mohamed Elrefaie, Steffen Hüttig, Mariia Gladkova, Timo Gericke, Daniel Cremers, and Christian Breitsamter. Real-time and on-site aerodynamics using stereoscopic piv and deep optical flow learning. arXiv preprint arXiv:2401.09932, 2024.

[24] Nick Erickson, Jonas Mueller, Alexander Shirkov, Hang Zhang, Pedro Larroy, Mu Li, and Alexander Smola. Autogluon-tabular: Robust and accurate automl for structured data. arXiv preprint arXiv:2003.06505, 2020.

[25] Matthias Fey and Jan Eric Lenssen. Fast graph representation learning with pytorch geometric. arXiv preprint arXiv:1903.02428, 2019.

[26] Jerome H Friedman. Greedy function approximation: a gradient boosting machine. Annals of statistics, pages 1189–1232, 2001.

[27] Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé Iii, and Kate Crawford. Datasheets for datasets. Communications of the ACM, 64(12): 86–92, 2021.

[28] Christopher Greenshields. OpenFOAM v11 User Guide. The OpenFOAM Foundation, London, UK, 2023. URL https://doc.cfd.direct/openfoam/user-guide-v11.

[29] Erkan Gunpinar, Umut Can Coskun, Mustafa Ozsipahi, and Serkan Gunpinar. A generative design and drag coefficient prediction system for sedan car side silhouettes based on computational fluid dynamics. CAD Computer Aided Design, 111:65–79, 6 2019. ISSN 00104485. doi: 10.1016/j.cad.2019.02.003.

[30] Sheikh Md Shakeel Hassan, Arthur Feeney, Akash Dhruv, Jihoon Kim, Youngjoon Suh, Jaiyoung Ryu, Yoonjin Won, and Aparna Chandramowlishwaran. Bubbleml: A multiphase multiphysics dataset and benchmarks for machine learning. Advances in Neural Information Processing Systems, 36, 2024.

[31] Angelina Heft, Thomas Indinger, and Nikolaus Adams. Investigation of unsteady flow structures in the wake of a realistic generic car model. In 29th AIAA applied aerodynamics conference, page 3669, 2011.

[32] Angelina I Heft, Thomas Indinger, and Nikolaus A Adams. Experimental and numerical investigation of the drivaer model. In Fluids Engineering Division Summer Meeting, volume 44755, pages 41–51. American Society of Mechanical Engineers, 2012.

[33] Angelina I Heft, Thomas Indinger, and Nikolaus A Adams. Introduction of a new realistic generic car model for aerodynamic investigations. Technical report, SAE Technical Paper, 2012.

[34] Wolf-Heinrich Hucho. Aerodynamik des Automobils: eine Brücke von der Strömungsmechanik zur Fahrzeugtechnik. Springer-Verlag, 2013.

[35] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning, pages 448–456. pmlr, 2015.

[36] M Islam, F Decker, E De Villiers, Aea Jackson, J Gines, T Grahs, A Gitt-Gehrke, and J Comas i Font. Application of detached-eddy simulation for automotive aerodynamics development. Technical report, SAE Technical Paper, 2009.

[37] Sam Jacob Jacob, Markus Mrosek, Carsten Othmer, and Harald Köstler. Deep learning for real-time aerodynamic evaluations of arbitrary vehicle shapes. SAE International Journal of Passenger Vehicle Systems, 15(2):77–90, mar 2022. ISSN 2770-3460. doi: https://doi.org/10.4271/15-15-02-0006. URL https://doi.org/10.4271/15-15-02-0006.

[38] Max Jaderberg, Karen Simonyan, Andrew Zisserman, et al. Spatial transformer networks. Advances in neural information processing systems, 28, 2015.

[39] Ali Kashefi and Tapan Mukerji. Physics-informed pointnet: A deep learning solver for steady-state incompressible flows and thermal fields on multiple sets of irregular geometries. Journal of Computational Physics, 468:111510, 2022.

[40] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. Lightgbm: A highly efficient gradient boosting decision tree. Advances in neural information processing systems, 30, 2017.

[41] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

[42] Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016.

[43] Zongyi Li, Nikola Borislavov Kovachki, Chris Choy, Boyi Li, Jean Kossaifi, Shourya Prakash Otta, Mohammad Amin Nabian, Maximilian Stadler, Christian Hundt, Kamyar Azizzadenesheli, and Anima Anandkumar. Geometry-informed neural operator for large-scale 3d pdes, 2023.

[44] H Martins, CO Henriques, JR Figueira, CS Silva, and AS Costa. Assessing policy interventions to stimulate the transition of electric vehicle technology in the european union. Socio-Economic Planning Sciences, 87: 101505, 2023.

[45] Florian R Menter, Martin Kuntz, Robin Langtry, et al. Ten years of industrial experience with the sst turbulence model. Turbulence, heat and mass transfer, 4(1):625–632, 2003.

[46] Peter Mock and Sonsoles Díaz. Pathways to decarbonization: the european passenger car market in the years 2021–2035. communications, 49:847129–848102, 2021.

[47] Chair of Aerodynamics and Technical University of Munich Fluid Mechanics. Drivaer model geometry. https://www.epc.ed.tum.de/en/aer/research-groups/automotive/drivaer/ geometry/, 2024. Accessed: 2024-05-21.

[48] OpenFOAM Foundation. Meshing with snappyHexMesh, 2023. URL https://www.openfoam.com/ documentation/guides/latest/doc/guide-meshing-snappyhexmesh.html. Accessed: 2024-0605.

[49] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.

[50] Giancarlo Pavia and Martin Passmore. Characterisation of wake bi-stability for a square-back geometry with rotating wheels. In Progress in Vehicle Aerodynamics and Thermal Management: 11th FKFS Conference, Stuttgart, September 26-27, 2017 11, pages 93–109. Springer, 2018.

[51] Songyou Peng, Michael Niemeyer, Lars Mescheder, Marc Pollefeys, and Andreas Geiger. Convolutional occupancy networks. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part III 16, pages 523–540. Springer, 2020.

[52] Leif E Peterson. K-nearest neighbor. Scholarpedia, 4(2):1883, 2009.

[53] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 652–660, 2017.

[54] Peng Qin, Alessio Ricci, and Bert Blocken. Cfd simulation of aerodynamic forces on the drivaer car model: Impact of computational parameters. Journal of Wind Engineering and Industrial Aerodynamics, 248:105711, 2024. ISSN 0167-6105. doi: https://doi.org/10.1016/j.jweia.2024.105711. URL https: //www.sciencedirect.com/science/article/pii/S0167610524000746.

[55] Edoardo Remelli, Artem Lukoianov, Stephan Richter, Benoit Guillard, Timur Bagautdinov, Pierre Baque, and Pascal Fua. Meshsdf: Differentiable iso-surface extraction. Advances in Neural Information Processing Systems, 33:22468–22478, 2020. URL https://proceedings.neurips.cc/paper_files/paper/ 2020/file/fe40fb944ee700392ed51bfe84dd4e3d-Paper.pdf.

[56] Thiago Rios, Patricia Wollstadt, Bas Van Stein, Thomas Back, Zhao Xu, Bernhard Sendhoff, and Stefan Menzel. Scalability of learning tasks on 3d cae models using point cloud autoencoders. pages 13671374. Institute of Electrical and Electronics Engineers Inc., 12 2019. ISBN 9781728124858. doi: 10.1109/SSCI44817.2019.9002982.

[57] Thiago Rios, Bas Van Stein, Thomas Back, Bernhard Sendhoff, and Stefan Menzel. Point2ffd: Learning shape representations of simulation-ready 3d models for engineering design optimization. pages 10241033. Institute of Electrical and Electronics Engineers Inc., 2021. ISBN 9781665426886. doi: 10.1109/ 3DV53792.2021.00110.

[58] Thiago Rios, Bas van Stein, Patricia Wollstadt, Thomas Bäck, Bernhard Sendhoff, and Stefan Menzel. Exploiting local geometric features in vehicle design optimization with 3d point cloud autoencoders. In 2021 IEEE Congress on Evolutionary Computation (CEC), pages 514–521, 2021. doi: 10.1109/CEC45853. 2021.9504746.

[59] Thomas Schütz. Hucho-Aerodynamik des Automobils: Strömungsmechanik, Wärmetechnik, Fahrdynamik, Komfort. Springer-Verlag, 2013.

[60] Shengrong Shen, Tian Han, and Jiachen Pang. Car drag coefficient prediction using long–short term memory neural network and lasso. Measurement, 225:113982, 2024.

[61] Binyang Song, Chenyang Yuan, Frank Permenter, Nikos Arechiga, and Faez Ahmed. Surrogate modeling of car drag coefficient with depth and normal renderings. arXiv preprint arXiv:2306.06110, 2023.

[62] D Brian Spalding. The numerical computation of turbulent flow. Comp. Methods Appl. Mech. Eng., 3:269, 1974.

[63] Guocheng Tao, Chengwei Fan, Wen Wang, Wenjun Guo, and Jiahuan Cui. Multi-fidelity deep learning for aerodynamic shape optimization using convolutional neural network. Physics of Fluids, 36(5), 2024.

[64] Nils Thuerey, Konstantin Weißenow, Lukas Prantl, and Xiangyu Hu. Deep learning methods for reynoldsaveraged navier–stokes simulations of airfoil flows. AIAA Journal, 58(1):25–36, 2020.

[65] Artur Toshev, Gianluca Galletti, Fabian Fritz, Stefan Adami, and Nikolaus Adams. Lagrangebench: A lagrangian fluid mechanics benchmarking suite. Advances in Neural Information Processing Systems, 36, 2024.

[66] Thanh Luan Trinh, Fangge Chen, Takuya Nanri, and Kei Akasaka. 3d super-resolution model for vehicle flow field enrichment. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 5826–5835, 2024.

[67] Nobuyuki Umetani and Bernd Bickel. Learning three-dimensional flow for interactive aerodynamic design. ACM Transactions on Graphics, 37, 2018. ISSN 15577368. doi: 10.1145/3197517.3201325.

[68] Muhammad Usama, Aqib Arif, Farhan Haris, Shahroz Khan, S. Kamran Afaq, and Shahrukh Rashid. A datadriven interactive system for aerodynamic and user-centred generative vehicle design. In 2021 International Conference on Artificial Intelligence (ICAI), pages 119–127, 2021. doi: 10.1109/ICAI52203.2021.9445243.

[69] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9(11), 2008.

[70] Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E Sarma, Michael M Bronstein, and Justin M Solomon. Dynamic graph cnn for learning on point clouds. ACM Transactions on Graphics (tog), 38(5):1–12, 2019.

[71] Mark D Wilkinson, Michel Dumontier, IJsbrand Jan Aalbersberg, Gabrielle Appleton, Myles Axton, Arie Baak, Niklas Blomberg, Jan-Willem Boiten, Luiz Bonino da Silva Santos, Philip E Bourne, et al. The fair guiding principles for scientific data management and stewardship. Scientific data, 3(1):1–9, 2016.

[72] Zhirong Wu, Shuran Song, Aditya Khosla, Fisher Yu, Linguang Zhang, Xiaoou Tang, and Jianxiong Xiao. 3d shapenets: A deep representation for volumetric shapes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1912–1920, 2015.

[73] Yu Xiang, Wonhui Kim, Wei Chen, Jingwei Ji, Christopher Choy, Hao Su, Roozbeh Mottaghi, Leonidas Guibas, and Silvio Savarese. Objectnet3d: A large scale database for 3d object recognition. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part VIII 14, pages 160–176. Springer, 2016.
