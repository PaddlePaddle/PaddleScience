
# FOURCASTNET

参考代码:
https://github.com/NVlabs/FourCastNet


## 下载数据集
从[此处](https://app.globus.org/file-manager?origin_id=945b3c9e-0f8c-11ed-8daf-9f359c660fbd&origin_path=%2F~%2Fdata%2F)下载数据集

## 模型训练
在配置文件中配置数据集路径:

```
  train_data_path:     # 训练集路径
  valid_data_path:     # 验证集路径
  inf_data_path:       # 测试集路径
  time_means_path:     # time_mean 文件路径（time_means.npy）
  global_means_path:   # mean 文件路径（global_means.npy）
  global_stds_path:    # std 文件路径（global_stds.npy）
  exp_dir:             # 输出路径
```
### 步骤 1 (风速模型预训练)

```
# 全局shuffle
python3 -m paddle.distributed.launch --gpus ${gpus} train.py --config afno_backbone
# 局部 shuffle
# python3 -m paddle.distributed.launch --gpus ${gpus} train_sampled.py --config afno_backbone
```

### 步骤 2 (风速模型微调)

```
# 需要先在配置文件中设置字段 pretrained_ckpt_path
# 全局shuffle
python3 -m paddle.distributed.launch --gpus ${gpus} train.py --config afno_backbone_finetune
# 局部shuffle
# python3 -m paddle.distributed.launch --gpus ${gpus} train_sampled.py --config afno_backbone_finetune
```


### 步骤 3 (降雨量模型训练)

```
# 需要先在配置文件中设置字段 model_wind_path
# 全局shuffle
python3 -m paddle.distributed.launch --gpus ${gpus} train.py --config precip
# 局部shuffle
# python3 -m paddle.distributed.launch --gpus ${gpus} train_sampled.py --config precip
```


## 模型推理

### 风速模型推理

    # batch方式推理
    python inference/inference_fast_batch.py --yaml_config 'your config file' --override_dir 'your output dir' --weights 'your training models'

### 降雨量模型推理

    python inference/inference_precip.py --yaml_config 'your config file' --override_dir 'your output dir' --weights 'your training models'
