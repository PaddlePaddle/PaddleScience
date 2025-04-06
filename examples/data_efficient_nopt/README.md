# Usage

## 1. Data Download

数据来源参考如下。

|dataset| model | size| link |
|---|---|---|---|
|possion_64|Poisson (FNO)|6.2GB|https://drive.google.com/drive/folders/1crIsTZGxZULWhrXkwGDiWF33W6RHxJkf|
|helmholtz_64|Helmholtz (FNO)|5.8GB|https://drive.google.com/drive/folders/1UjIaF6FsjmN_xlGGSUX-1K2V3EF2Zalw|

## 2. Pretraining

修改`config/operators_possion.yaml`或`config/operators_helmholtz.yaml`相关文件夹路径。

修改`run_pretrain.sh`中config路径，采用`possion_64`或`helmholtz_64`配置，同时指定`--config`预训练配置。

```shell
bash run_pretrain.py
```

## 3. Fine-tuning

修改`config/operators_possion.yaml`或`config/operators_helmholtz.yaml`相关文件夹路径。

修改`run_pretrain.sh`中config路径，采用`possion_64`或`helmholtz_64`配置，同时指定`--config`微调配置。

```shell
bash run_pretrain.py
```

## 4. In-context Learning

修改`config/inference_poisson.yaml`或`config/inference_helmholtz.yaml`相关文件夹路径。

修改`run_pretrain.sh`中config路径，采用`possion_64`或`helmholtz_64`配置，同时指定`--ckpt`权重路径。

```shell
bash run_inference_possion.py
```
