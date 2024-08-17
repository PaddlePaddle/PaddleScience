### 1. 模型简介：
锂离子电池电极材料性能预测，利用从材料项目数据集中提取的两组特征来预测电池的平均电压、比容量和比能量等电化学性能。

### 2. 问题定义：
该 MLP 模型能够使用相应的电极材料实现对电池电化学性能的准确预测。

输入包括化学计量属性、晶体结构特性、电子结构属性和其他电池属性。输出包括平均电压、比能量和比容量。

数据集来自 Materials Project。


### 3. MLP-model-for-prediction-of-battery-electrochemical-performance
This MLP model is able to achieve accurate predictions of the battery electrochemical performance using the corresponding electrode materials. The input includes stoichiometric attributes, crystal structure properties, electronic structure attributes and other battery attributes. The output includes the average voltage, specific energy and specific capacity. The datasets are obtained from Materials Project.

### 4. 程序运行主代码

请参考 MLP.py

### 5. 数据集文件

来自网站https://next-gen.materialsproject.org/batteries

其中训练数据集已经准备好，如下：

MP_data_down_loading(train+validate).csv

MP_data_down_loading(train+validate+test).csv

### 6. 运行结果

请参考 MLP model for prediction of battery electrochemical performance - V8.ipynb

### 7. 来源参考

https://doi.org/10.1016/j.gee.2022.10.002
