# DoMINO: Decomposable Multi-scale Iterative Neural Operator for External Aerodynamics

DoMINO代码复现。

## 安装依赖

```shell
# 安装PaddlePaddle

cd /path/PaddleScience
pip install -e .

cd /path/PaddleScience/examples/domino
pip install -r requirements.txt
```

## 数据集下载与处理

1. 参考[DoMINO](https://github.com/NVIDIA/modulus/tree/main/examples/cfd/external_aerodynamics/domino#training-the-domino-model)数据下载和处理方式，执行`download_aws_dataset.sh`和`process_data.py`，获取数据。

2. 本仓库训练数据为`process_data.py`的后处理数据。

## 训练

1. 修改`conf/config.yaml`路径, 原始配置文件参考[DoMINO](https://github.com/NVIDIA/modulus/blob/main/examples/cfd/external_aerodynamics/domino/src/conf/config.yaml)。

2. 训练

```shell
cd /path/PaddleScience/examples/domino
python train.py
```

## 推理

1. 修改`conf/config.yaml`路径, 原始配置文件参考[DoMINO](https://github.com/NVIDIA/modulus/blob/main/examples/cfd/external_aerodynamics/domino/src/conf/config.yaml)。

2. 推理

```shell
cd /path/PaddleScience/examples/domino
python test.py
```
