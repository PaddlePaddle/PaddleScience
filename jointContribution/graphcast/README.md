# 飞桨黑客马拉松第五期 科学计算 GraphCast: Learning skillful medium-range global weather forecasting

## 1.简介

本项目基于paddle框架复现。

论文主要内容：
GraphCast这种方法是天气预报领域的一项重大进展，它利用机器学习的能力来提高预测的准确性和效率。GraphCast通过图神经网络（GNNs）建模复杂的天气动态，并在欧洲中期天气预报中心（ECMWF）的ERA5再分析数据集上进行训练。它在全球范围内以0.25°的高分辨率快速预测数百种天气变量，并在多项目标上超越了ECMWF的高分辨率预测系统（HRES）。这项研究表明，GraphCast不仅能提高标准天气预测的效率，还在预测严重天气事件方面显示出潜力，可能对依赖天气的决策过程产生重大影响。

本项目关键技术：

* 通过结合mesh和grid的节点和边特征，大大提升了图神经网络的预测性能；
* 通过精细化的训练数据预处理，提升了模型对特征学习的能力；
* 通过大量的训练数据（1979-2017年，40T左右）,减缓了图神经网络（16层）的过拟合问题。（个人认为这点对图神经网络的研究有启发性作用。）

实验结果要点：

* 完整复现数据处理过程和模型结构，推理过程误差在1e-5以下。
* 简单明确的网络结构和易于理解的数据处理流程便于后续研究工作推进。

论文信息：
Lam R, Sanchez-Gonzalez A, Willson M, et al. Learning skillful medium-range global weather forecasting[J]. Science, 2023: eadi2336.

参考Github地址：
<https://github.com/deepmind/graphcast>

项目aistudio地址：
<https://aistudio.baidu.com/projectdetail/7266127>

模型结构：
![](https://ai-studio-static-online.cdn.bcebos.com/a1bee2bc4a7548e69a2d37324c89b868f84da4ebf79b4542be4b18a815be7418)

## 2. 模型

* `GraphCast`，在GraphCast论文中使用的高分辨率模型（0.25度分辨率，37个压力层），在1979年至2017年的ERA5数据上训练，
* `GraphCast_small`，GraphCast的较小、低分辨率版本（1度分辨率，13个压力层和较小的网格），在1979年至2015年的ERA5数据上训练，适用于在内存和计算约束较低的情况下运行模型，
* `GraphCast_operational`，高分辨率模型（0.25度分辨率，1313个压力层), 该模型是在1979年至2017年的ERA5数据上进行预训练，并在2016年至2021年的HRES数据上进行微调的压力层级预测模型。该模型可以从HRES数据初始化（不需要降水输入）。

原作者描述如下：

>* `GraphCast`, the high-resolution model used in the GraphCast paper (0.25 degree
resolution, 37 pressure levels), trained on ERA5 data from 1979 to 2017,
>* `GraphCast_small`, a smaller, low-resolution version of GraphCast (1 degree
resolution, 13 pressure levels, and a smaller mesh), trained on ERA5 data from
1979 to 2015, useful to run a model with lower memory and compute constraints,
>* `GraphCast_operational`, a high-resolution model (0.25 degree resolution, 13
pressure levels) pre-trained on ERA5 data from 1979 to 2017 and fine-tuned on
HRES data from 2016 to 2021. This model can be initialized from HRES data (does
not require precipitation inputs).

## 3. 数据集

本项目数据集由作者提供用于运行测试，如需进行训练，需至ERA5网站进行下载整理。

原作者提供说明如下：
>The model weights, normalization statistics, and example inputs are available on [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/dm_graphcast).

本项目为用户提供3种数据（对应3种模型）和完整的模型参数用于测试运行。具体对应细节已在`config/graphcast*.json`中详细说明。通过下方**快速运行**章节说明，可快速得到对应数据和参数。

* GraphCast: data/dataset/source-era5_date-2022-01-01_res-0.25_levels-37_steps-01.nc
* GraphCast_small:data/dataset/source-era5_date-2022-01-01_res-1.0_levels-13_steps-01.nc
* GraphCast_operational:data/dataset/source-hres_date-2022-01-01_res-0.25_levels-13_steps-01.nc

## 4. 快速运行

本节提供环境依赖、数据准备、功能运行说明。

### 4.1 环境依赖

* paddlepaddle
* matpoltlib （用于图像绘制）
* pickle （用于存储和加载图模板）
* xarray （用于加载.nc数据）
* trimesh （用于制作mesh数据）

本项目在aistudio中仅缺少xarray和trimesh，运行下方指令进行安装。

```python
!pip install xarray trimesh
```

### 4.2 数据准备

本项目已经完整准备数据并绑定至项目，在运行前仅需解压即可。

```python
!unzip -q data/data252766/dataset.zip -d data/
!unzip -q data/data252766/params.zip -d data/
!unzip -q data/data252766/stats.zip -d data/
!unzip -q data/data252766/template_graph.zip -d data/
!cp data/data252766/graphcast-jax2paddle.csv data/
!cp data/data252766/jax_graphcast_small_output.npy data/
```

### 4.3 功能运行

主要功能如下：

* convert_parameters() ： 转换原始jax模型参数至paddle模型参数。（已经转换并保存，可以跳过运行）
* make_graph_template()：制作并保存图结构模板，可减少大规模数据训练时数据制作时间。（已经制作并保存，可以跳过运行）
* test_datasets()：测试训练数据制作流程。（可跳过运行）
* eval()：根据转换后模型参数、图模板、数据进行推理预测。
* visualize()：结果绘图展示。
* compare()：对比jax输出结果和paddle复现结果。

```python
import json
import os
import pickle

import numpy as np
import paddle

import args
import datasets
import graphcast
import graphtype
import vis

# isort: off
from graphtype import GraphGridMesh  # noqa: F401
from graphtype import TriangularMesh  # noqa: F401


def convert_parameters():
    def convert(
        jax_parameters_path,
        paddle_parameters_path,
        mapping_csv,
        model,
        output_size=False,
    ):
        model = graphcast.GraphCastNet(config)
        state_dict = model.state_dict()
        jax_data = np.load(jax_parameters_path)

        if output_size:
            for key in state_dict.keys():
                print(key, state_dict[key].shape)

            for param_name in jax_data.files:
                if jax_data[param_name].size == 1:
                    print(param_name, "\t", jax_data[param_name])
                else:
                    print(param_name, "\t", jax_data[param_name].shape)

        with open(mapping_csv, "r") as f:
            mapping = [line.strip().split(",") for line in f]
            for jax_key, paddle_key in mapping:
                state_dict[paddle_key].set_value(jax_data[jax_key])
        paddle.save(state_dict, paddle_parameters_path)

    params_path = "data/params"
    mapping_path = "data/graphcast-jax2paddle.csv"

    params_names = [p for p in os.listdir(params_path) if ".npz" in p]
    config_jsons = {
        "resolution 0.25 - pressure levels 37": "config/GraphCast.json",
        "resolution 0.25 - pressure levels 13": "config/GraphCast_operational.json",
        "resolution 1.0 - pressure levels 13": "config/GraphCast_small.json",
    }

    for params_type, config_json in config_jsons.items():
        params_name = [n for n in params_names if params_type in n]
        if len(params_name) > 1:
            raise ValueError("More one parameter files")
        params_name = params_name[0]

        print(f"Start convert '{params_type}' parameters...")
        config_json = config_jsons[params_type]
        jax_parameters_path = os.path.join(params_path, params_name)
        paddle_parameters_path = os.path.join(
            params_path,
            params_name.replace(".npz", ".pdparams").replace(" ", "-"),
        )
        with open(config_json, "r") as f:
            config = args.TrainingArguments(**json.load(f))
        convert(jax_parameters_path, paddle_parameters_path, mapping_path, config)
        print(f"Convert {params_type} parameters finished.")


def make_graph_template():
    config_jsons = {
        "resolution 0.25 - pressure levels 37": "config/GraphCast.json",
        "resolution 0.25 - pressure levels 13": "config/GraphCast_operational.json",
        "resolution 1.0 - pressure levels 13": "config/GraphCast_small.json",
    }

    for model_type, config_json in config_jsons.items():
        print(
            f"Make graph template for {model_type} and "
            "Save into data/template_graph folder"
        )

        with open(config_json, "r") as f:
            config = args.TrainingArguments(**json.load(f))
        graph = GraphGridMesh(config=config)

        graph_template_path = os.path.join(
            "data/template_graph",
            f"{config.type}.pkl",
        )
        with open(graph_template_path, "wb") as f:
            pickle.dump(graph, f)


def test_datasets():
    with open("config/GraphCast_small.json", "r") as f:
        config = args.TrainingArguments(**json.load(f))
    era5dataset = datasets.ERA5Data(config=config, data_type="train")
    print(era5dataset)


def eval():
    with open("config/GraphCast_small.json", "r") as f:
        config = args.TrainingArguments(**json.load(f))
    dataset = datasets.ERA5Data(config=config, data_type="train")
    model = graphcast.GraphCastNet(config)
    model.set_state_dict(paddle.load(config.param_path))
    graph = model(graphtype.convert_np_to_tensor(dataset.input_data[0]))
    pred = dataset.denormalize(graph.grid_node_feat.numpy())
    pred = graph.grid_node_outputs_to_prediction(pred, dataset.targets_template)
    print(pred)
    return (
        graph.grid_node_outputs_to_prediction(
            dataset.target_data[0], dataset.targets_template
        ),
        pred,
    )


def visualize(target, pred, variable_name, level, robust=True):
    plot_size = 5
    plot_max_steps = pred.dims["time"]

    data = {
        "Targets": vis.scale(
            vis.select(target, variable_name, level, plot_max_steps), robust=robust
        ),
        "Predictions": vis.scale(
            vis.select(pred, variable_name, level, plot_max_steps), robust=robust
        ),
        "Diff": vis.scale(
            (
                vis.select(target, variable_name, level, plot_max_steps)
                - vis.select(pred, variable_name, level, plot_max_steps)
            ),
            robust=robust,
            center=0,
        ),
    }
    fig_title = variable_name
    if "level" in pred[variable_name].coords:
        fig_title += f" at {level} hPa"

    vis.plot_data(data, fig_title, plot_size, robust, cols=1)

def compare(paddle_pred):
    with open("config/GraphCast_small.json", "r") as f:
        config = args.TrainingArguments(**json.load(f))
    dataset = datasets.ERA5Data(config=config, data_type="train")
    graph = graphtype.convert_np_to_tensor(dataset.input_data[0])

    jax_graphcast_small_pred_path = "data/jax_graphcast_small_output.npy"
    jax_graphcast_small_pred = np.load(jax_graphcast_small_pred_path).reshape(
        181 * 360, 1, 83
    )
    jax_graphcast_small_pred = graph.grid_node_outputs_to_prediction(
        jax_graphcast_small_pred, dataset.targets_template
    )

    paddle_graphcast_small_pred = paddle_pred

    for var_name in list(paddle_graphcast_small_pred):
        diff_var = np.average(
            jax_graphcast_small_pred[var_name].data
            - paddle_graphcast_small_pred[var_name].data
        )
        print(var_name, f"diff is {diff_var}")

    jax_graphcast_small_pred_np = datasets.dataset_to_stacked(
        jax_graphcast_small_pred
    )
    paddle_graphcast_small_pred_np = datasets.dataset_to_stacked(
        paddle_graphcast_small_pred
    )
    diff_all = np.average(
        jax_graphcast_small_pred_np.data - paddle_graphcast_small_pred_np.data
    )
    print(f"All diff is {diff_all}")

```

```python
# convert_parameters()  # step.1 pre-finished
# make_graph_template()  # step.2 pre-finished
# test_datasets()  # step.3 pre-finished
target, pred = eval()  # step.4
```

```python
visualize(target, pred, "2m_temperature", level=50) # 此处可修改变量及level(参考args)
```

```python
# 计算graphcast的jax输出结果和paddle复现结果差值
compare(pred)
```

## 5.模型信息

| 信息     | 说明                                                                  |
| -------- | --------------------------------------------------------------------- |
| 发布者   | 朱卫国 (DrownFish19)                                                  |
| 发布时间 | 2023.12                                                               |
| 框架版本 | paddle 2.5.2                                                          |
| 支持硬件 | GPU、CPU                                                              |
| aistudio | [notebook](https://aistudio.baidu.com/aistudio/projectdetail/7266127) |
