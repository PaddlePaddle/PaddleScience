# neo_geometry模块使用说明

本目录下的python文件主要支持STL几何、加密采样、布尔运算功能，同时重构了简单几何的代码，统一了简单几何和STL几何的基本功能和接口。
下面说明neo_geometry中的几何类如何使用

## 依赖安装
- open3d 安装

    ```shell
    pip install open3d
    ```
- pysdf 安装

    ```shell
    pip install pysdf
    ```

- pymesh 安装

    请参考 [官方文档](https://pymesh.readthedocs.io/en/latest/installation.html#) （使用Setuptools进行安装）

## 简单几何初始化与基础采样
```python
import paddlescience as psci
import numpy as np


# 初始化一个左下角端点是[0.0, 0.0], 右上角端点是[1.0, 1.0]的矩形
rect = psci.neo_geometry.Rectangle([0.0, 0.0], [1.0, 1.0])

# 给rect添加第1个采样配置，在其内部采样
rect.add_sample_config(
    "interior", # 采样数据名称
    10000, # 采样数据个数
    None, # 设置额外的criteria，用于在筛选出特定数据
    "pseudo", # 采样时的随机方式
    True # 采样数据是否在第一次采样后固定
)

# 给rect添加第2个采样配置，在其边界采样
rect.add_sample_config(
    "boundary", # 采样区域
    10000, # 采样点数
    None, # 设置额外的criteria，用于在筛选出特定点
    "pseudo", # 采样时的随机方式
    True # 采样数据是否在第一次采样后固定
)

# 给rect添加第3个采样配置，在其内部采样并指定criteria筛选条件，只采样左半边的点
rect.add_sample_config(
    "interior_left", # 采样区域
    10000, # 采样点数
    lambda x, y: x < 0.5, # 设置额外的criteria，用于在筛选出特定点
    "pseudo", # 采样时的随机方式
    True # 采样数据是否在第一次采样后固定
)

# 给rect添加第4个采样配置，在其边界采样并指定criteria筛选条件，只采样右半边的点
rect.add_sample_config(
    "boundary_right", # 采样区域
    10000, # 采样点数
    lambda x, y: x > 0.5, # 设置额外的criteria，用于在筛选出特定点
    "pseudo", # 采样时的随机方式
    True # 采样数据是否在第一次采样后固定
)

# 根据每个采样配置，在对应区域采样并返回数据
data_dict = rect.fetch_batch_data()

# 可视化
for name, data in data_dict.items():
    print(f"{name} {data.shape}")
    psci.visu.__save_vtk_raw(name, data, np.full([len(data), 1], 1))
```

## STL几何初始化与加密采样
```python
import numpy as np

import paddlescience as psci

# 初始化一个来自stl文件的几何形状（ball.stl是一个球心在原点的球）
ball = psci.neo_geometry.Mesh("./paddlescience/neo_geometry/ball.stl")

# 沿着ball的外法线方向0.3单位距离处，加密采样10000个点，并可视化
inflate_outer = ball.inflated_random_boundary_points(10000, 0.3)
print(inflate_outer.shape)
psci.visu.__save_vtk_raw("inflate_outer", inflate_outer, np.full([len(inflate_outer), 1], 1))

# 沿着ball的内法线方向0.5单位距离处，加密采样10000个点，并可视化
inflate_inner = ball.inflated_random_boundary_points(10000, -0.5)
print(inflate_outer.shape)
psci.visu.__save_vtk_raw("inflate_inner", inflate_inner, np.full([len(inflate_inner), 1], 1))
```
