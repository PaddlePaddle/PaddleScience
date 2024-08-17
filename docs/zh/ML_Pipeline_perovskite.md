# 模型简介：
新型钙钛矿太阳能电池材料的探索,
利用钙钛矿数据库项目的数据，使用XGBoost模型，预测高Jsc(短路电流密度)值的材料，能够在设计高效PSCs(钙钛矿太阳能电池)时提供有价值的指导


# 数据来源和选择：
数据来自钙钛矿数据库项目（Perovskite Database Project） [Perovskite Database](https://www.perovskitedatabase.com/Download) 。
从原始数据中选择相关的列。

# 安装要求：
    pip install -r requirements.txt

# 数据集文件:
用于我们的机器学习分析的数据集可以在终端中运行的脚本create_data.py，
python scripts/create_data.py

它从钙钛矿数据库中获取 CSV 文件，并将其转换为干净的 csv 文件，以备机器学习训练。

该流程是模块化的，可以用作具有可变特征集的不同预测目标的模板。它旨在处理钙钛矿数据库的 PDF 内容描述中概述的数据格式

该文件的结构如下：
1.指定所需的列
2.指定目标列
3.指明列数据类别和任何列的特殊要求
4.管道的辅助函数
5.主数据管道
# 运行主代码：
python notebooks/xg_optuna.py

# 文献来源：
DOI: 10.1002/apxr.202400060