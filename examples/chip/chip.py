# 顶部表面随机的2D功率图
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import paddle.nn as nn
from scipy.fftpack import idctn

import ppsci
from ppsci.arch import activation as act_mod
from ppsci.arch import base
from ppsci.arch import mlp
from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.utils import logger

rng = np.random.default_rng()

# set random seed for reproducibility
ppsci.utils.misc.set_random_seed(42)

# set output directory
OUTPUT_DIR = "./output1"

# initialize logger
logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

# 设置无量纲计算参数
DL = 1.0
DW = 1.0
DH = 0.5

# 设置无量纲热源参数
qs_type = 2


class DeepONet(base.Arch):
    def __init__(
        self,
        branch_input_keys: Tuple[str, ...],
        trunk_input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        num_loc: int,
        num_features: int,
        branch_num_layers: int,
        trunk_num_layers: int,
        branch_hidden_size: Union[int, Tuple[int, ...]],
        trunk_hidden_size: Union[int, Tuple[int, ...]],
        branch_skip_connection: bool = False,
        trunk_skip_connection: bool = False,
        branch_activation: str = "tanh",
        trunk_activation: str = "tanh",
        branch_weight_norm: bool = False,
        trunk_weight_norm: bool = False,
        use_bias: bool = True,
    ):
        super().__init__()
        self.trunk_input_keys = trunk_input_keys
        self.branch_input_keys = branch_input_keys
        self.input_keys = self.trunk_input_keys + self.branch_input_keys
        self.output_keys = output_keys

        self.branch_net = mlp.MLP(
            self.branch_input_keys,
            ("b",),
            branch_num_layers,
            branch_hidden_size,
            branch_activation,
            branch_skip_connection,
            branch_weight_norm,
            input_dim=num_loc,
            output_dim=num_features,
        )

        self.trunk_net = mlp.MLP(
            self.trunk_input_keys,
            ("t",),
            trunk_num_layers,
            trunk_hidden_size,
            trunk_activation,
            trunk_skip_connection,
            trunk_weight_norm,
            input_dim=len(self.trunk_input_keys),
            output_dim=num_features,
        )
        self.trunk_act = act_mod.get_activation(trunk_activation)
        self.branch_act = act_mod.get_activation(branch_activation)

        self.use_bias = use_bias
        if use_bias:
            # register bias to parameter for updating in optimizer and storage
            self.b = self.create_parameter(
                shape=(1,),
                attr=nn.initializer.Constant(0.0),
            )

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)

        # Branch net to encode the input function
        u_features = self.branch_net(x)[self.branch_net.output_keys[0]]
        # u_features = self.branch_act(u_features)

        # Trunk net to encode the domain of the output function
        y_features = self.trunk_net(x)[self.trunk_net.output_keys[0]]
        y_features = self.trunk_act(y_features)

        # Dot product
        # G_u = paddle.einsum("bi,bi->b", u_features, y_features)  # [batch_size, ]
        # G_u = paddle.reshape(G_u, [-1, 1])  # reshape [batch_size, ] to [batch_size, 1]
        G_u = paddle.sum(u_features * y_features, axis=1, keepdim=True)
        # Add bias
        if self.use_bias:
            G_u += self.b

        result_dict = {
            self.output_keys[0]: G_u,
        }
        if self._output_transform is not None:
            result_dict = self._output_transform(x, result_dict)

        return result_dict


# set model
NUM_SENSORS = 400
NUM_FEATURES = 128
model = DeepONet(
    ("u",),
    (
        "x",
        "y",
        "z",
    ),
    ("T",),
    NUM_SENSORS,
    NUM_FEATURES,
    9,
    6,
    256,
    128,
    branch_activation="swish",
    trunk_activation="swish",
    use_bias=True,
)


def GRF(alpha, tau, s):
    """
    通过使用 GRF (Gaussian Random Field) 模型对信号进行建模，然后对其进行 IDCT (反离散余弦变换) 得到结果

    Args:
    alpha (float): 参数 alpha 控制信号的平滑性
    tau (float): 参数 tau 控制信号的长度
    s (int): 信号的长度

    Returns:
    np.ndarray: IDCT 后的信号数组
    """
    xi = np.random.randn(s, s)
    K1, K2 = np.meshgrid(np.arange(s), np.arange(s))
    coef = tau ** (alpha - 1) * (np.pi**2 * (K1**2 + K2**2) + tau**2) ** (
        -alpha / 2
    )
    L = s * coef * xi
    L[0, 0] = 0
    U = idctn(L, norm="ortho")
    return U


NL, NW, NH, NU = 20, 20, 10, 50
NPOINT = NL * NW * NH

geom = {"rect": ppsci.geometry.Cuboid((0, 0, 0), (DL, DW, DH))}
points = geom["rect"].sample_interior(NPOINT, evenly=True)

data_u = GRF(alpha=5, tau=4, s=NL).reshape([1, -1])
test_u = GRF(alpha=5, tau=4, s=NL).reshape([1, -1])
for i in range(NU - 1):
    data_u = np.vstack((data_u, GRF(alpha=5, tau=4, s=NL).reshape([1, -1])))
    test_u = np.vstack((test_u, GRF(alpha=5, tau=4, s=NL).reshape([1, -1])))
data_u = data_u.astype("float32")
test_u = test_u.astype("float32")

points["x"] = np.repeat(points["x"], NU, axis=0)
points["y"] = np.repeat(points["y"], NU, axis=0)
points["z"] = np.repeat(points["z"], NU, axis=0)
points["u"] = np.tile(data_u, (NPOINT, 1))
points["test_u"] = np.tile(test_u, (NPOINT, 1))

top_indices = points["z"] == DH
down_indices = points["z"] == 0
left_indices = (points["y"] == 0) & (points["z"] != 0) & (points["z"] != DH)
right_indices = (points["y"] == DL) & (points["z"] != 0) & (points["z"] != DH)
front_indices = (
    (points["x"] == DW)
    & (points["z"] != 0)
    & (points["z"] != DH)
    & (points["y"] != 0)
    & (points["y"] != DL)
)
back_indices = (
    (points["x"] == 0)
    & (points["z"] != 0)
    & (points["z"] != DH)
    & (points["y"] != 0)
    & (points["y"] != DL)
)
interior_indices = (
    (points["x"] != 0)
    & (points["x"] != DW)
    & (points["z"] != 0)
    & (points["z"] != DH)
    & (points["y"] != 0)
    & (points["y"] != DL)
)
# 使用numpy的where函数获取对应的索引
top_indices = np.where(top_indices)
down_indices = np.where(down_indices)
left_indices = np.where(left_indices)
right_indices = np.where(right_indices)
front_indices = np.where(front_indices)
back_indices = np.where(back_indices)
interior_indices = np.where(interior_indices)

top_data = {
    "x": points["x"][top_indices[0]],
    "y": points["y"][top_indices[0]],
    "z": points["z"][top_indices[0]],
    "u": points["u"][top_indices[0]],
    "u_one": data_u.T.reshape([-1, 1]),
}
down_data = {
    "x": points["x"][down_indices[0]],
    "y": points["y"][down_indices[0]],
    "z": points["z"][down_indices[0]],
    "u": points["u"][down_indices[0]],
}
left_data = {
    "x": points["x"][left_indices[0]],
    "y": points["y"][left_indices[0]],
    "z": points["z"][left_indices[0]],
    "u": points["u"][left_indices[0]],
}
right_data = {
    "x": points["x"][right_indices[0]],
    "y": points["y"][right_indices[0]],
    "z": points["z"][right_indices[0]],
    "u": points["u"][right_indices[0]],
}
front_data = {
    "x": points["x"][front_indices[0]],
    "y": points["y"][front_indices[0]],
    "z": points["z"][front_indices[0]],
    "u": points["u"][front_indices[0]],
}
back_data = {
    "x": points["x"][back_indices[0]],
    "y": points["y"][back_indices[0]],
    "z": points["z"][back_indices[0]],
    "u": points["u"][back_indices[0]],
}
interior_data = {
    "x": points["x"][interior_indices[0]],
    "y": points["y"][interior_indices[0]],
    "z": points["z"][interior_indices[0]],
    "u": points["u"][interior_indices[0]],
}
test_data = {
    "x": points["x"][interior_indices[0]],
    "y": points["y"][interior_indices[0]],
    "z": points["z"][interior_indices[0]],
    "u": points["test_u"][interior_indices[0]],
}
test_top_data = {
    "x": points["x"][top_indices[0]],
    "y": points["y"][top_indices[0]],
    "z": points["z"][top_indices[0]],
    "u": points["test_u"][top_indices[0]],
    "u_one": test_u.T.reshape([-1, 1]),
}

top_down_label = {"neumann": np.zeros([NL * NW * NU, 1], dtype="float32")}
left_right_label = {"neumann": np.zeros([NL * (NH - 2) * NU, 1], dtype="float32")}
front_back_label = {"neumann": np.zeros([(NW - 2) * (NH - 2) * NU, 1], dtype="float32")}
interior_label = {
    "thermal_condution": np.zeros([interior_data["x"].shape[0], 1], dtype="float32")
}

NI = NL * NW * NH - 2 * NL * NW - 2 * NL * (NH - 2) - 2 * (NW - 2) * (NH - 2)

top_sup_constraint = ppsci.constraint.SupervisedConstraint(
    {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": top_data,
            "label": top_down_label,
        },
        "batch_size": NL * NW * NU,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
    },
    ppsci.loss.MSELoss("mean"),
    output_expr={
        "neumann": lambda out: jacobian(out["T"], out["z"]) * 0.1 + out["u_one"]
    },
    name="top_sup",
)
down_sup_constraint = ppsci.constraint.SupervisedConstraint(
    {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": down_data,
            "label": top_down_label,
        },
        "batch_size": NL * NW * NU,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
    },
    ppsci.loss.MSELoss("mean"),
    output_expr={
        "neumann": lambda out: jacobian(out["T"], out["z"]) * 0.1
        + 500 * (out["T"] - 298.15)
    },
    name="down_sup",
)
left_sup_constraint = ppsci.constraint.SupervisedConstraint(
    {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": left_data,
            "label": left_right_label,
        },
        "batch_size": NL * (NH - 2) * NU,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
    },
    ppsci.loss.MSELoss("mean"),
    output_expr={"neumann": lambda out: jacobian(out["T"], out["y"])},
    name="left_sup",
)
right_sup_constraint = ppsci.constraint.SupervisedConstraint(
    {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": right_data,
            "label": left_right_label,
        },
        "batch_size": NL * (NH - 2) * NU,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
    },
    ppsci.loss.MSELoss("mean"),
    output_expr={"neumann": lambda out: jacobian(out["T"], out["y"])},
    name="right_sup",
)
front_sup_constraint = ppsci.constraint.SupervisedConstraint(
    {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": front_data,
            "label": front_back_label,
        },
        "batch_size": (NW - 2) * (NH - 2) * NU,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
    },
    ppsci.loss.MSELoss("mean"),
    output_expr={"neumann": lambda out: jacobian(out["T"], out["x"])},
    name="front_sup",
)
back_sup_constraint = ppsci.constraint.SupervisedConstraint(
    {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": back_data,
            "label": front_back_label,
        },
        "batch_size": (NW - 2) * (NH - 2) * NU,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
    },
    ppsci.loss.MSELoss("mean"),
    output_expr={"neumann": lambda out: jacobian(out["T"], out["x"])},
    name="back_sup",
)
interior_sup_constraint = ppsci.constraint.SupervisedConstraint(
    {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": interior_data,
            "label": interior_label,
        },
        "batch_size": NI * NU,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
    },
    ppsci.loss.MSELoss("mean"),
    output_expr={
        "thermal_condution": lambda out: hessian(out["T"], out["x"])
        + hessian(out["T"], out["y"])
        + hessian(out["T"], out["z"])
    },
    name="interior_sup",
)

# wrap constraints together
constraint = {
    top_sup_constraint.name: top_sup_constraint,
    down_sup_constraint.name: down_sup_constraint,
    left_sup_constraint.name: left_sup_constraint,
    right_sup_constraint.name: right_sup_constraint,
    front_sup_constraint.name: front_sup_constraint,
    back_sup_constraint.name: back_sup_constraint,
    interior_sup_constraint.name: interior_sup_constraint,
}

# set training hyper-parameters
EPOCHS = 2000
EVAL_FREQ = 1000
ITERS_PER_EPOCH = 1
# set optimizer
lr_scheduler = ppsci.optimizer.lr_scheduler.Step(
    EPOCHS, ITERS_PER_EPOCH, 0.001, 500, 0.1, by_epoch=True
)()
optimizer = ppsci.optimizer.Adam(lr_scheduler)((model,))

# set validator
interior_validator = ppsci.validate.SupervisedValidator(
    {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": test_data,
            "label": interior_label,
        },
        "batch_size": 1000,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    },
    ppsci.loss.MSELoss("mean"),
    output_expr={
        "thermal_condution": lambda out: hessian(out["T"], out["x"])
        + hessian(out["T"], out["y"])
        + hessian(out["T"], out["z"])
    },
    metric={"MSE": ppsci.metric.MSE()},
    name="interior_mse",
)
top_validator = ppsci.validate.SupervisedValidator(
    {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": test_top_data,
            "label": top_down_label,
        },
        "batch_size": 1000,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    },
    ppsci.loss.MSELoss("mean"),
    output_expr={
        "neumann": lambda out: jacobian(out["T"], out["z"]) * 0.1 + out["u_one"]
    },
    metric={"MSE": ppsci.metric.MSE()},
    name="top_mse",
)
validator = {
    interior_validator.name: interior_validator,
    top_validator.name: top_validator,
}

# initialize solver
solver = ppsci.solver.Solver(
    model,
    constraint,
    OUTPUT_DIR,
    optimizer,
    None,
    EPOCHS,
    ITERS_PER_EPOCH,
    eval_during_train=True,
    eval_freq=EVAL_FREQ,
    # equation=equation,
    validator=validator,
)
# train model
solver.train()
# evaluate after finished training
solver.eval()

solver.plot_loss_history()


# directly evaluate model from pretrained_model_path(optional)
# logger.init_logger("ppsci", f"{OUTPUT_DIR}/eval.log", "info")
# solver = ppsci.solver.Solver(
#     model,
#     constraint,
#     OUTPUT_DIR,
#     equation=equation,
#     geom=geom,
#     validator=validator,
#     pretrained_model_path=f"{OUTPUT_DIR}/checkpoints/latest",
# )
# solver.eval()
# visualize prediction from pretrained_model_path(optional)

pred_u1 = np.ones(400).reshape([1, -1])
a1 = np.ones([10, 10])
a2 = np.zeros([10, 10])
array_20x20 = np.zeros((20, 20))
array_20x20[-5:, -5:] = 1
pred_u2 = np.vstack((np.hstack((a2, a2)), np.hstack((a2, a1)))).reshape([1, -1])
pred_u3 = np.zeros(400).reshape([1, -1])
pred_u4 = array_20x20.reshape([1, -1])
pred_u = pred_u4
pred_points = geom["rect"].sample_interior(NPOINT, evenly=True)
pred_u = pred_u.astype("float32")

pred_points["u"] = np.tile(pred_u, (NPOINT, 1))

pred = solver.predict(pred_points)
ppsci.visualize.save_vtu_from_dict(
    "result1.vtu",
    {
        "x": pred_points["x"],
        "y": pred_points["y"],
        "z": pred_points["z"],
        "T": pred["T"],
    },
    ("x", "y", "z"),
    ("T"),
)
