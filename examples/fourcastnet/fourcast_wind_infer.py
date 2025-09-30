# 风速模型实现
import functools
import numpy as np
import paddle
import utils as fourcast_utils
import ppsci
from ppsci.utils import logger

paddle.set_device("sdaa")

# set dataset path
DATA_FILE_PATH = "./datasets/era5/test/2018-09-08_n32.npy"
DATA_MEAN_PATH = "./datasets/era5/stat/global_means.npy"
DATA_STD_PATH = "./datasets/era5/stat/global_stds.npy"

# set training hyper-parameters
NUM_TIMESTAMPS = 32
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
OUTPUT_DIR ="./output/fourcastnet/inference_wind"
MODEL_PATH = "./model_wind/best_ckpt.pdparams"

logger.init_logger("ppsci", f"{OUTPUT_DIR}/eval.log", "info")

# 然后我们需要根据设置的数据路径读取数据，并且需要读取数据集的均值、方差用于后续输入数据的归一化，代码如下：
def get_vis_datas(
    file_path: str,
    num_timestamps: int,
    data_mean: np.ndarray,
    data_std: np.ndarray,
):
    data = np.load(file_path)

    vis_datas = {"input": (data[:, 0] - data_mean) / data_std}
    for t in range(num_timestamps):
        hour = (t + 1) * 6
        data_t = data[:, t + 1]
        wind_data = []
        for i in range(data_t.shape[0]):
            wind_data.append((data_t[i][0] ** 2 + data_t[i][1] ** 2) ** 0.5)
        vis_datas[f"target_{hour}h"] = np.asarray(wind_data)
    return vis_datas

data_mean, data_std = fourcast_utils.get_mean_std(
    DATA_MEAN_PATH, DATA_STD_PATH, VARS_CHANNEL
)

# set visualizer datas
vis_datas = get_vis_datas(
    DATA_FILE_PATH,
    NUM_TIMESTAMPS,
    data_mean,
    data_std,
)

# 根据设置的超参数构建模型
model = ppsci.arch.AFNONet(input_keys, output_keys, num_timestamps=NUM_TIMESTAMPS)

# 由于模型对风速的纬向和经向分开预测，因此需要把这两个方向上的风速合成为真正的风速
def output_wind_func(d, var_name, data_mean, data_std):
    output = (d[var_name] * data_std) + data_mean
    wind_data = []
    for i in range(output.shape[0]):
        wind_data.append((output[i][0] ** 2 + output[i][1] ** 2) ** 0.5)
    return paddle.to_tensor(wind_data, paddle.get_default_dtype())

vis_output_expr = {}
for i in range(NUM_TIMESTAMPS):
    hour = (i + 1) * 6
    vis_output_expr[f"output_{hour}h"] = functools.partial(
        output_wind_func,
        var_name=f"output_{i}",
        data_mean=paddle.to_tensor(data_mean, paddle.get_default_dtype()),
        data_std=paddle.to_tensor(data_std, paddle.get_default_dtype()),
    )
    vis_output_expr[f"target_{hour}h"] = lambda d, hour=hour: d[f"target_{hour}h"]

# 最后，构建可视化器
visualizer = {
    "visulize_wind": ppsci.visualize.VisualizerWeather(
        vis_datas,
        vis_output_expr,
        xticks=np.linspace(0, 1439, 13),
        xticklabels=[str(i) for i in range(360, -1, -30)],
        yticks=np.linspace(0, 719, 7),
        yticklabels=[str(i) for i in range(90, -91, -30)],
        vmin=0,
        vmax=25,
        colorbar_label="m\s",
        batch_size=1,
        num_timestamps=NUM_TIMESTAMPS,
        prefix="wind",
    )
}

# 以上构建好的模型、可视化器将会传递给 `ppsci.solver.Solver` 用于在输入数据上进行可视化
solver = ppsci.solver.Solver(
    model,
    output_dir=OUTPUT_DIR,
    visualizer=visualizer,
    pretrained_model_path=MODEL_PATH,
    eval_with_no_grad=True,
)
# visualize prediction from pretrained_model_path
solver.visualize()
