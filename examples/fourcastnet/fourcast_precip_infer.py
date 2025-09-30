# 降雨量模型实现
import functools

import numpy as np
import paddle

import utils as fourcast_utils
import ppsci
from ppsci.utils import logger


# set wind dataset path
WIND_DATA_FILE_PATH = "./datasets/era5/test/2018-04-04_n6.npy"
WIND_MEAN_PATH = "./datasets/era5/stat/global_means.npy"
WIND_STD_PATH = "./datasets/era5/stat/global_stds.npy"
# set dataset path
DATA_FILE_PATH = "./datasets/era5/test/2018-04-04_n6_precip.npy"

# set hyper-parameters
NUM_TIMESTAMPS = 6
input_keys = ("input",)
output_keys = tuple(f"output_{i}" for i in range(NUM_TIMESTAMPS))
IMG_H, IMG_W = 720, 1440
# FourCastNet use 20 atmospheric variable，their index in the dataset is from 0 to 19.
# The variable name is 'u10', 'v10', 't2m', 'sp', 'msl', 't850', 'u1000', 'v1000', 'z000',
# 'u850', 'v850', 'z850',  'u500', 'v500', 'z500', 't500', 'z50', 'r500', 'r850', 'tcwv'.
# You can obtain detailed information about each variable from
# https://cds.climate.copernicus.eu/cdsapp#!/search?text=era5&type=dataset
VARS_CHANNEL = list(range(20))
# set output directory
OUTPUT_DIR = "./output/fourcastnet/inference_precip"

WIND_MODEL_PATH = "./model_wind/best_ckpt"
MODEL_PATH = "./model_precip/best_ckpt"

logger.init_logger("ppsci", f"{OUTPUT_DIR}/eval.log", "info")

# 根据设置的数据路径读取数据，并且需要读取数据集的均值、方差用于后续输入数据的归一化
def get_vis_datas(
    wind_file_path: str,
    file_path: str,
    num_timestamps: int,
    data_mean: np.ndarray,
    data_std: np.ndarray,
):
    wind_data = np.load(wind_file_path)
    data = np.load(file_path)

    vis_datas = {"input": (wind_data - data_mean) / data_std}
    for t in range(num_timestamps):
        hour = (t + 1) * 6
        data_t = data[:, t]
        vis_datas[f"target_{hour}h"] = np.asarray(data_t)
    return vis_datas

wind_data_mean, wind_data_std = fourcast_utils.get_mean_std(
    WIND_MEAN_PATH, WIND_STD_PATH, VARS_CHANNEL
)

# set set visualizer datas
vis_datas = get_vis_datas(
    WIND_DATA_FILE_PATH,
    DATA_FILE_PATH,
    NUM_TIMESTAMPS,
    wind_data_mean,
    wind_data_std,
)

# 根据设置的超参数构建模型
wind_model = ppsci.arch.AFNONet(input_keys, output_keys)
ppsci.utils.save_load.load_pretrain(wind_model, path=WIND_MODEL_PATH)
model = ppsci.arch.PrecipNet(
    input_keys, output_keys, num_timestamps=NUM_TIMESTAMPS, wind_model=wind_model
)

# 由于模型对降雨量进行了对数处理，因此需要将模型结果重新映射回线性空间
def output_precip_func(d, var_name):
    output = 1e-2 * paddle.expm1(d[var_name][0])
    return output

visu_output_expr = {}
for i in range(NUM_TIMESTAMPS):
    hour = (i + 1) * 6
    visu_output_expr[f"output_{hour}h"] = functools.partial(
        output_precip_func,
        var_name=f"output_{i}",
    )
    visu_output_expr[f"target_{hour}h"] = (
        lambda d, hour=hour: d[f"target_{hour}h"] * 1000
    )


# 最后，构建可视化器
visualizer = {
    "visulize_precip": ppsci.visualize.VisualizerWeather(
        vis_datas,
        visu_output_expr,
        xticks=np.linspace(0, 1439, 13),
        xticklabels=[str(i) for i in range(360, -1, -30)],
        yticks=np.linspace(0, 719, 7),
        yticklabels=[str(i) for i in range(90, -91, -30)],
        vmin=0.001,
        vmax=130,
        colorbar_label="mm",
        log_norm=True,
        batch_size=1,
        num_timestamps=NUM_TIMESTAMPS,
        prefix="precip",
    )
}

# directly evaluate pretrained model
solver = ppsci.solver.Solver(
    model,
    output_dir=OUTPUT_DIR,
    visualizer=visualizer,
    pretrained_model_path=MODEL_PATH,
    eval_with_no_grad=True,
)
# visualize prediction from pretrained_model_path
solver.visualize()
