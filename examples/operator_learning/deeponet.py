"""
Reference: https://deepxde.readthedocs.io/en/latest/demos/operator/antiderivative_unaligned.html
"""

import os
from typing import Callable
from typing import Tuple

import numpy as np
import paddle
from matplotlib import pyplot as plt

import ppsci
from ppsci.utils import config
from ppsci.utils import logger

if __name__ == "__main__":
    args = config.parse_args()
    # set random seed for reproducibility
    SEED = 2023
    ppsci.utils.misc.set_random_seed(SEED)
    # set output directory
    OUTPUT_DIR = "./output_DeepONet" if not args.output_dir else args.output_dir
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # set model
    NUM_SENSORS = 100
    NUM_FEATURES = 40
    model = ppsci.arch.DeepONet(
        "u",
        "y",
        "G",
        NUM_SENSORS,
        NUM_FEATURES,
        1,
        1,
        40,
        40,
        branch_activation="relu",
        trunk_activation="relu",
        use_bias=True,
    )

    # set dataloader config
    ITERS_PER_EPOCH = 1
    train_dataloader_cfg = {
        "dataset": {
            "name": "IterableNPZDataset",
            "file_path": "./antiderivative_unaligned_train.npz",
            "input_keys": ("u", "y"),
            "label_keys": ("G",),
            "alias_dict": {"u": "X_train0", "y": "X_train1", "G": "y_train"},
        },
    }

    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        ppsci.loss.MSELoss(),
        {"G": lambda out: out["G"]},
    )
    # wrap constraints together
    constraint = {sup_constraint.name: sup_constraint}

    # set training hyper-parameters
    EPOCHS = 10000 if not args.epochs else args.epochs

    # set optimizer
    optimizer = ppsci.optimizer.Adam(1e-3)(model)

    # set validator
    eval_dataloader_cfg = {
        "dataset": {
            "name": "IterableNPZDataset",
            "file_path": "./antiderivative_unaligned_test.npz",
            "input_keys": ("u", "y"),
            "label_keys": ("G",),
            "alias_dict": {"u": "X_test0", "y": "X_test1", "G": "y_test"},
        },
    }
    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        ppsci.loss.MSELoss(),
        {"G": lambda out: out["G"]},
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="G_eval",
    )
    validator = {sup_validator.name: sup_validator}

    # initialize solver
    EVAL_FREQ = 500
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        optimizer,
        None,
        EPOCHS,
        ITERS_PER_EPOCH,
        save_freq=EVAL_FREQ,
        log_freq=20,
        eval_during_train=True,
        eval_freq=EVAL_FREQ,
        seed=SEED,
        validator=validator,
        eval_with_no_grad=True,
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()

    # directly evaluate pretrained model(optional)
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/eval.log", "info")
    solver = ppsci.solver.Solver(
        model,
        None,
        OUTPUT_DIR,
        validator=validator,
        pretrained_model_path=f"{OUTPUT_DIR}/checkpoints/latest",
        eval_with_no_grad=True,
    )
    solver.eval()

    # visualize prediction for different functions u and corresponding G(u)
    NUM_Y = 1000  # number of y point for G(u) to be visualized
    dtype = paddle.get_default_dtype()

    def generate_y_u_G_ref(
        u_func: Callable, G_u_func: Callable
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate discretized data of given function u and corresponding G(u).

        Args:
            u_func (Callable): Function u.
            G_u_func (Callable): Function G(u).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Discretized data of u, y and G(u).
        """
        x = np.linspace(0, 1, NUM_SENSORS, dtype=dtype).reshape([1, NUM_SENSORS])
        u = u_func(x)
        u = np.tile(u, [NUM_Y, 1])

        y = np.linspace(0, 1, NUM_Y, dtype=dtype).reshape([NUM_Y, 1])
        G_ref = G_u_func(y)
        return u, y, G_ref

    func_u_G_pair = [
        # (title_string, func_u, func_G(u)), s.t. dG/dx == u and G(u)(0) = 0
        (r"$u=\cos(x), G(u)=sin(x$)", lambda x: np.cos(x), lambda y: np.sin(y)),  # 1
        (
            r"$u=sec^2(x), G(u)=tan(x$)",
            lambda x: (1 / np.cos(x)) ** 2,
            lambda y: np.tan(y),
        ),  # 2
        (
            r"$u=sec(x)tan(x), G(u)=sec(x) - 1$",
            lambda x: (1 / np.cos(x) * np.tan(x)),
            lambda y: 1 / np.cos(y) - 1,
        ),  # 3
        (
            r"$u=1.5^x\ln{1.5}, G(u)=1.5^x-1$",
            lambda x: 1.5**x * np.log(1.5),
            lambda y: 1.5**y - 1,
        ),  # 4
        (r"$u=3x^2, G(u)=x^3$", lambda x: 3 * x**2, lambda y: y**3),  # 5
        (r"$u=4x^3, G(u)=x^4$", lambda x: 4 * x**3, lambda y: y**4),  # 6
        (r"$u=5x^4, G(u)=x^5$", lambda x: 5 * x**4, lambda y: y**5),  # 7
        (r"$u=6x^5, G(u)=x^6$", lambda x: 5 * x**4, lambda y: y**5),  # 8
        (r"$u=e^x, G(u)=e^x-1$", lambda x: np.exp(x), lambda y: np.exp(y) - 1),  # 9
    ]

    os.makedirs(os.path.join(OUTPUT_DIR, "visual"), exist_ok=True)
    for i, (title, u_func, G_func) in enumerate(func_u_G_pair):
        u, y, G_ref = generate_y_u_G_ref(u_func, G_func)
        G_pred = solver.predict({"u": u, "y": y})["G"].numpy()
        plt.plot(y, G_pred, label=r"$G(u)(y)_{ref}$")
        plt.plot(y, G_ref, label=r"$G(u)(y)_{pred}$")
        plt.legend()
        plt.title(title)
        plt.savefig(os.path.join(OUTPUT_DIR, "visual", f"func_{i}_result.png"))
        plt.clf()
