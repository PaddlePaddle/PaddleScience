"""
Reference: https://deepxde.readthedocs.io/en/latest/demos/operator/antiderivative_unaligned.html
"""

import os
from os import path as osp
from typing import Callable
from typing import Tuple

import hydra
import numpy as np
import paddle
from matplotlib import pyplot as plt
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger


def train(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # set model
    model = ppsci.arch.DeepONet(**cfg.MODEL)

    # set dataloader config
    train_dataloader_cfg = {
        "dataset": {
            "name": "IterableNPZDataset",
            "file_path": cfg.TRAIN_FILE_PATH,
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

    # set optimizer
    optimizer = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model)

    # set validator
    eval_dataloader_cfg = {
        "dataset": {
            "name": "IterableNPZDataset",
            "file_path": cfg.VALID_FILE_PATH,
            "input_keys": ("u", "y"),
            "label_keys": ("G",),
            "alias_dict": {"u": "X_test0", "y": "X_test1", "G": "y_test"},
        },
    }
    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        ppsci.loss.MSELoss(),
        {"G": lambda out: out["G"]},
        metric={"MeanL2Rel": ppsci.metric.MeanL2Rel()},
        name="G_eval",
    )
    validator = {sup_validator.name: sup_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        None,
        cfg.TRAIN.epochs,
        cfg.TRAIN.iters_per_epoch,
        save_freq=cfg.TRAIN.save_freq,
        eval_freq=cfg.TRAIN.eval_freq,
        log_freq=cfg.log_freq,
        seed=cfg.seed,
        validator=validator,
        eval_during_train=cfg.TRAIN.eval_during_train,
        checkpoint_path=cfg.TRAIN.checkpoint_path,
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()

    # visualize prediction for different functions u and corresponding G(u)
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
        x = np.linspace(0, 1, cfg.MODEL.num_loc, dtype=dtype).reshape(
            [1, cfg.MODEL.num_loc]
        )
        u = u_func(x)
        u = np.tile(u, [cfg.NUM_Y, 1])

        y = np.linspace(0, 1, cfg.NUM_Y, dtype=dtype).reshape([cfg.NUM_Y, 1])
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

    os.makedirs(os.path.join(cfg.output_dir, "visual"), exist_ok=True)
    for i, (title, u_func, G_func) in enumerate(func_u_G_pair):
        u, y, G_ref = generate_y_u_G_ref(u_func, G_func)
        G_pred = solver.predict({"u": u, "y": y}, return_numpy=True)["G"]
        plt.plot(y, G_pred, label=r"$G(u)(y)_{ref}$")
        plt.plot(y, G_ref, label=r"$G(u)(y)_{pred}$")
        plt.legend()
        plt.title(title)
        plt.savefig(os.path.join(cfg.output_dir, "visual", f"func_{i}_result.png"))
        plt.clf()


def evaluate(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # set model
    model = ppsci.arch.DeepONet(**cfg.MODEL)

    # set validator
    eval_dataloader_cfg = {
        "dataset": {
            "name": "IterableNPZDataset",
            "file_path": cfg.VALID_FILE_PATH,
            "input_keys": ("u", "y"),
            "label_keys": ("G",),
            "alias_dict": {"u": "X_test0", "y": "X_test1", "G": "y_test"},
        },
    }
    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        ppsci.loss.MSELoss(),
        {"G": lambda out: out["G"]},
        metric={"MeanL2Rel": ppsci.metric.MeanL2Rel()},
        name="G_eval",
    )
    validator = {sup_validator.name: sup_validator}

    solver = ppsci.solver.Solver(
        model,
        None,
        cfg.output_dir,
        validator=validator,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )
    solver.eval()

    # visualize prediction for different functions u and corresponding G(u)
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
        x = np.linspace(0, 1, cfg.MODEL.num_loc, dtype=dtype).reshape(
            [1, cfg.MODEL.num_loc]
        )
        u = u_func(x)
        u = np.tile(u, [cfg.NUM_Y, 1])

        y = np.linspace(0, 1, cfg.NUM_Y, dtype=dtype).reshape([cfg.NUM_Y, 1])
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

    os.makedirs(os.path.join(cfg.output_dir, "visual"), exist_ok=True)
    for i, (title, u_func, G_func) in enumerate(func_u_G_pair):
        u, y, G_ref = generate_y_u_G_ref(u_func, G_func)
        G_pred = solver.predict({"u": u, "y": y}, return_numpy=True)["G"]
        plt.plot(y, G_pred, label=r"$G(u)(y)_{ref}$")
        plt.plot(y, G_ref, label=r"$G(u)(y)_{pred}$")
        plt.legend()
        plt.title(title)
        plt.savefig(os.path.join(cfg.output_dir, "visual", f"func_{i}_result.png"))
        plt.clf()


@hydra.main(version_base=None, config_path="./conf", config_name="deeponet.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
