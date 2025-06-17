# Suzuki-Miyaura 交叉偶联反应产率预测

!!! note

    1. 开始训练、评估前，数据文件data_set.xlsx的存在，并对应修改 yaml 配置文件中的 `data_dir` 为数据文件路径。
    2. 如果需要使用预训练模型进行评估，请先下载预训练模型[chem_model.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/TADF/Est/Est_pretrained.pdparams), 并对应修改 yaml 配置文件中的 `load_model_path` 为模型参数路径。
    3. 开始训练、评估前，请安装 `rdkit` 等，相关依赖请执行`pip install -r requirements.txt`安装。

=== "模型训练命令"

    ``` sh
    # 训练:  
    python Chem.py mode=train
    ```

=== "模型评估命令"

    ``` sh
    # 评估：
    python Chem.py mode=eval
    ```

## 1. 背景简介

Suzuki-Miyaura 交叉偶联反应具有反应条件温和、转化率高的优点，在材料合成、药物研发等领域具有重要作用，但存在开发周期长，试错成本高的问题。本研究通过使用高通量实验数据分析反应底物（包括亲电试剂和亲核试剂），催化配体，碱基，溶剂对偶联反应产率的影响，从而建立预测模型。