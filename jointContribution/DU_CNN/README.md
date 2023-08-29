# Deep learning based denoising process

## Reference  

Yoon, Taekeun, et al. "Deep learning-based denoising for fast time-resolved flame emission spectroscopy in high-pressure combustion environment." Combustion and Flame 248 (2023): 112583.
<https://doi.org/10.1016/j.combustflame.2022.112583>

## Included folders and files

* data  : raw dataset (download following [file (url)](https://drive.google.com/file/d/1yOuxJmI4tKYI3tJEJIWKf52T4SjAfaSB/view?usp=share_link)
'rawdataM.mat' and locate in data folder)
* models   : model strucutre python code
* utils   : required functions
* main.py : main code
* enivornment.yaml  : conda environment
* config.yaml  : configuration

## Requirements

* python environment (anaconda)
* python version 3.10.4
* window 10 64bit

## Procedure

$ : command <br/>

1. Select working directory
2. Download [data](https://drive.google.com/file/d/1yOuxJmI4tKYI3tJEJIWKf52T4SjAfaSB/view?usp=share_link) in ./data/
3. $ conda create -n DUCNN
4. $ conda activate DUCNN
5. $ conda env create --file environment.yaml
6. $ python main.py

### Note1

If the GPU memory is not enough,
Reduce batch_size

-------------------
原 github 仓库 <https://github.com/ytg7146/DU_CNN>
使用 `conda env create --file environment.yaml` 安装依赖失败，所以删除 environment.yaml
增加 requirements.txt

安装依赖运行

``` shell
pip install -r requirements.txt
```

数据集下载
[rawdataM.mat](https://drive.google.com/file/d/1yOuxJmI4tKYI3tJEJIWKf52T4SjAfaSB/view?usp=share_link)
