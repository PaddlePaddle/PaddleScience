# Deep learning based denoising process

## 参考

Yoon, Taekeun, et al. "Deep learning-based denoising for fast time-resolved flame emission spectroscopy in high-pressure combustion environment." Combustion and Flame 248 (2023): 112583.
<https://doi.org/10.1016/j.combustflame.2022.112583>
<https://github.com/ytg7146/DU_CNN>

## 包含目录和文件

* data  : 数据集下载到 data 目录 [rawdataM.mat](https://drive.google.com/file/d/1yOuxJmI4tKYI3tJEJIWKf52T4SjAfaSB/view?usp=share_link)
* models  : 模型
* utils  : 需要的辅助函数
* main.py  : 主程序
* requirements.txt  : 安装的依赖
* config.yaml  : 配置

## 步骤

1. 选择工作目录
2. 下载数据集到目录 ./data/
3. conda create -n DUCNN
4. conda activate DUCNN
5. pip install -r requirements.txt
6. python main.py

### 注意

如果 GPU 内存不足，降低 batch_size
