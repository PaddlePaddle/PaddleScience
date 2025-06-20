import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import paddle
from matplotlib import cm

import ppsci
from ppsci.utils import logger


def evaluate(cfg=None):
    # 设置随机种子
    ppsci.utils.misc.set_random_seed(42)

    # 初始化日志
    output_dir = "./output/schrodinger_eval"
    os.makedirs(output_dir, exist_ok=True)
    logger.init_logger("ppsci", osp.join(output_dir, "eval.log"), "info")

    # 设置边界
    T_LB = 0.0
    T_UB = 2.0
    X_LB = -5.0
    X_UB = 5.0

    logger.info("初始化模型...")
    # 初始化模型
    model_idn_u = ppsci.arch.MLP(
        input_keys=("t", "x"),
        output_keys=("u_idn",),
        num_layers=4,
        hidden_size=50,
        activation="sin",
    )

    model_idn_v = ppsci.arch.MLP(
        input_keys=("t", "x"),
        output_keys=("v_idn",),
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

    # 初始化solver
    solver = ppsci.solver.Solver(
        model=model_list,
        output_dir=output_dir,
        pretrained_model_path="./pretrained_models/schrodinger_pretrained.pdparams",
    )

    # 可视化预测结果
    logger.info("生成可视化结果...")
    with solver.no_grad_context_manager(True):
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
    npz_path = osp.join(output_dir, "schrodinger_eval.npz")
    np.savez(npz_path, **result_dict)
    logger.info(f"保存数据到: {npz_path}")

    # 绘图
    plt.figure(figsize=(18, 5))
    plt.pcolor(t_mesh, x_mesh, uv_amplitude, cmap=cm.jet)
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Predicted |u|")
    plt.tight_layout()
    fig_path = osp.join(output_dir, "schrodinger_eval.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    logger.info(f"评估结果已保存到: {output_dir}")
    print(f"\n评估完成！结果已保存到: {output_dir}")
    print(f"生成的图片: {fig_path}")


if __name__ == "__main__":
    evaluate()
