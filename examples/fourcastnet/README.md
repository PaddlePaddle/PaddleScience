
# FOURCASTNET

This code is refer from:
https://github.com/NVlabs/FourCastNet


## Download dataset
Download the dataset from [here](https://app.globus.org/file-manager?origin_id=945b3c9e-0f8c-11ed-8daf-9f359c660fbd&origin_path=%2F~%2Fdata%2F)

## Training
The flolowing data paths need to be set in config file:

```
  train_data_path:     # path to train data
  valid_data_path:     # path to valid data
  inf_data_path:       # path to test data
  time_means_path:     # path to time_means.npy
  global_means_path:   # path to global_means.npy
  global_stds_path:    # path to global_stds.npy
  exp_dir:             # path to store outputs
```
### setp 1 (wind model pretraining)

```
# global shuffle
python3 -m paddle.distributed.launch --gpus ${gpus} train.py --config afno_backbone
# local shuffle
# python3 -m paddle.distributed.launch --gpus ${gpus} train_sampled.py --config afno_backbone
```

### setp 2 (wind model finetuning)

```
# set pretrained_ckpt_path in config file
# 全局shuffle
python3 -m paddle.distributed.launch --gpus ${gpus} train.py --config afno_backbone_finetune
# 局部shuffle
# python3 -m paddle.distributed.launch --gpus ${gpus} train_sampled.py --config afno_backbone_finetune
```


### setp 3 (precip model training)

```
# set model_wind_path in config file
# 全局shuffle
python3 -m paddle.distributed.launch --gpus ${gpus} train.py --config precip
# 局部shuffle
# python3 -m paddle.distributed.launch --gpus ${gpus} train_sampled.py --config precip
```


## Inference

### wind model inference

    # batch方式推理
    python inference/inference_fast_batch.py --yaml_config 'your config file' --override_dir 'your output dir' -- weights 'your training models'

### precip model inference

    python inference/inference_precip.py --yaml_config 'your config file' --override_dir 'your output dir' -- weights 'your training models'
