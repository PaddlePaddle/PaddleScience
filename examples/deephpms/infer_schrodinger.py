import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import paddle
from matplotlib import cm

import ppsci
from ppsci.utils import logger

# 设置随机种子
ppsci.utils.misc.set_random_seed(42)

# 初始化日志
logger.init_logger("ppsci", "./output/schrodinger_infer.log", "info")

# 设置边界
T_LB = 0.0
T_UB = 2.0
X_LB = -5.0
X_UB = 5.0

# 初始化模型
model_idn_u = ppsci.arch.MLP(
    input_keys=("t", "x"),  # 使用元组而不是列表
    output_keys=("u_idn",),  # 使用元组而不是列表
    num_layers=4,
    hidden_size=50,
    activation="sin",
)

model_idn_v = ppsci.arch.MLP(
    input_keys=("t", "x"),  # 使用元组而不是列表
    output_keys=("v_idn",),  # 使用元组而不是列表
    num_layers=4,
    hidden_size=50,
    activation="sin",
)

# 初始化transform
t_lb = paddle.to_tensor(T_LB)
t_ub = paddle.to_tensor(T_UB)
x_lb = paddle.to_tensor(X_LB)
x_ub = paddle.to_tensor(X_UB)


def transform_uv(_in):
    t, x = _in["t"], _in["x"]
    t = 2.0 * (t - t_lb) * paddle.pow((t_ub - t_lb), -1) - 1.0
    x = 2.0 * (x - x_lb) * paddle.pow((x_ub - x_lb), -1) - 1.0
    input_trans = {"t": t, "x": x}
    return input_trans


# 注册transform
model_idn_u.register_input_transform(transform_uv)
model_idn_v.register_input_transform(transform_uv)

# 初始化model list
model_list = ppsci.arch.ModelList((model_idn_u, model_idn_v))

# 加载预训练模型
pretrained_path = "./pretrained_models/schrodinger_pretrained.pdparams"
ppsci.utils.save_load.load_pretrain(model_list, pretrained_path)

# 创建输出目录
output_dir = "./output/schrodinger_infer"
os.makedirs(output_dir, exist_ok=True)

# 创建网格点
t_points = 100
x_points = 100
t_min, t_max = T_LB, T_UB
x_min, x_max = X_LB, X_UB

t_points = np.linspace(t_min, t_max, t_points)
x_points = np.linspace(x_min, x_max, x_points)
t_mesh, x_mesh = np.meshgrid(t_points, x_points)

# 准备输入数据
t_flat = paddle.to_tensor(t_mesh.flatten()[:, None].astype(np.float32))
x_flat = paddle.to_tensor(x_mesh.flatten()[:, None].astype(np.float32))

# 运行预测
with paddle.no_grad():
    output_dict = model_list({"t": t_flat, "x": x_flat})

# 提取结果
u_pred = output_dict["u_idn"].numpy().reshape(x_mesh.shape)
v_pred = output_dict["v_idn"].numpy().reshape(x_mesh.shape)

# 计算波函数振幅
uv_amplitude = np.sqrt(u_pred**2 + v_pred**2)

# 保存结果
result_dict = {
    "t": t_mesh,
    "x": x_mesh,
    "u": u_pred,
    "v": v_pred,
    "amplitude": uv_amplitude,
}

# 保存为numpy文件
np.savez(osp.join(output_dir, "schrodinger_inference.npz"), **result_dict)

# 绘图函数
def draw_and_save(
    figname, data_learned, boundary, griddata_points, griddata_xi, save_path
):
    plt.figure(figsize=(18, 5))

    # 绘制预测结果
    plt.subplot(1, 1, 1)
    plt.pcolor(griddata_xi[0], griddata_xi[1], data_learned, cmap=cm.jet)
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Predicted |u|")

    # 保存图像
    plt.tight_layout()
    plt.savefig(osp.join(save_path, f"{figname}.png"), dpi=300)
    plt.close()


# 绘图并保存可视化结果
draw_and_save(
    figname="schrodinger_inference",
    data_learned=uv_amplitude,
    boundary=[t_min, t_max, x_min, x_max],
    griddata_points=np.column_stack((t_flat.numpy(), x_flat.numpy())),
    griddata_xi=(t_mesh, x_mesh),
    save_path=output_dir,
)

logger.info(f"Inference results saved to {output_dir}")
