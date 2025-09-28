# 需要加载环境变量export CUSTOM_DEVICE_BLACK_LIST=top_k_v2,top_k,mask_select
import os
import numpy as np
import paddle
from omegaconf import DictConfig
from omegaconf import OmegaConf
import  ppsci
from ppsci.utils import save_load

CMIP6_SST_MAX = 10.198975563049316
CMIP6_SST_MIN = -16.549121856689453
CMIP5_SST_MAX = 8.991744995117188
CMIP5_SST_MIN = -9.33076286315918
CMIP6_NINO_MAX = 4.138188362121582
CMIP6_NINO_MIN = -3.5832221508026123
CMIP5_NINO_MAX = 3.8253555297851562
CMIP5_NINO_MIN = -2.691682815551758

SST_MAX = max(CMIP6_SST_MAX, CMIP5_SST_MAX)
SST_MIN = min(CMIP6_SST_MIN, CMIP5_SST_MIN)
def scale_sst(sst):
    return (sst - SST_MIN) / (SST_MAX - SST_MIN)




def inference(cfg: DictConfig):
    #  载入配置
    normalize_sst = cfg.DATASET.normalize_sst
    print('normalize_sst:', normalize_sst)
    in_len = cfg.DATASET.in_len
    print('in_len:', in_len)
    input_keys = cfg.MODEL.input_keys

    # 载入数据
    test_00001_06_05 = np.load('./data/weather_data/icar_enso_2021/enso_final_test_data_B/test_00001_06_05.npy')

    # 4为预测因子，并按照SST,T300,Ua,Va的顺序存放，所以只取sst
    test_sst = paddle.to_tensor(test_00001_06_05[...,0], dtype='float32')
    test_sst = test_sst[...,np.newaxis]
    # 取19：67的区间是为了对应训练时的经度范围95E-330E
    test_sst = test_sst[ :, :, 19:67, :]

    test_sst_in_tar = np.concatenate([test_sst, test_sst, test_sst], axis=0)[:26]
    print('test_sst:', test_sst_in_tar.shape)
    if normalize_sst:
        test_sst_in_tar = scale_sst(test_sst_in_tar)

    # 构建模型所需的数据集
    in_seq = paddle.unsqueeze(paddle.to_tensor(test_sst_in_tar[: in_len, ...], dtype='float32'), axis=0) # ( in_len, lat, lon, 1)
    target_seq = paddle.unsqueeze(paddle.to_tensor(test_sst_in_tar[in_len :, ...], dtype='float32'), axis=0)  # ( in_len, lat, lon, 1)
    input_item = {input_keys[0]: in_seq, "sst_target": target_seq}

    # 载入模型
    moe_config = OmegaConf.to_object(cfg.MOE)
    rnc_config = OmegaConf.to_object(cfg.RNC)
    model = ppsci.arch.ExtFormerMoECuboid(**cfg.MODEL, moe_config=moe_config, rnc_config=rnc_config)
    save_load.load_pretrain(model, "./pretrained/extformer_moe_pretrained.pdparams")

    # 预测
    # model.eval()
    pred = model(input_item)
    print({k: (None if v is None else v.shape) for k, v in pred.items()})



def main(cfg: DictConfig):
   if cfg.mode == 'inference':
        inference(cfg)
   else:
        raise ValueError("Invalid mode: for inference only, but got {}".format(cfg.mode))

    

if __name__ == '__main__':
    cfg = OmegaConf.load("./conf/extformer_moe_enso_inference.yaml")
    main(cfg)