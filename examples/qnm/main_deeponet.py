"""
Reference: https://github.com/PredictiveIntelligenceLab/jaxpi/tree/main/examples/allen_cahn
"""

import copy
import os
from os import path as osp

import hydra
import numpy as np
import paddle
from initial_eigenvalues import Kerrfinit

# import scipy.io as sio
from matplotlib import pyplot as plt
from omegaconf import DictConfig

import ppsci
from ppsci.autodiff import hessian as hes
from ppsci.autodiff import jacobian as jac

# from ppsci.utils import misc
from ppsci.utils import logger
from ppsci.utils import save_load
from ppsci.utils.reader import load_csv_file

# import sympy as sp

dtype = "float64"
paddle.set_default_dtype(dtype)


def F_terms(a, w, A, s, m, x):
    """
    All these values were calculated by Mathematica.
    Calculates The F_i terms defined in the Appendix A, each one with shape (N_x,1).
    Receives as arguments:
    - a - the spin parameter, value between 0 and 0.5 (float);
    - w - the frequency of the QNM (parameter of the Neural Network);
    - A - the separation constant of thr Teukolsky equation (parameter of the Neural Network);
    - s - spin-weight (for tensor pertubations s = -2);
    - m - Spherical harmonic indicies l and m;
    - x: vector with dimensions (N_x,1) that defines the radial space.
    """

    # Important intermediate values:
    r_plus = (1 + np.sqrt(1 - 4 * a**2)) / 2

    # F0 term:
    F0 = (
        -(a**4) * x**2 * w**2
        - 2 * a**3 * m * x**2 * w
        + a**2
        * (
            -A * x**2
            + x**2
            * (
                4 * (r_plus + 1) * w**2
                + 2j * (r_plus + 2) * w
                + 2j * s * (w + 1j)
                - 2
            )
            + x * w**2
            - w**2
        )
        + 2 * a * m * (r_plus * x**2 * (2 * w + 1j) - x * (w + 1j) - w)
        + A * (x - 1)
        - 1j
        * r_plus
        * (2 * w + 1j)
        * (x**2 * (s - 2j * w + 1) - 2 * (s + 1) * x + 2j * w)
        + (s + 1) * (x - 2j * w)
    )

    # F1 term:
    F1 = (
        2 * a**4 * x**4 * (x - 1j * w)
        - 2j * a**3 * m * x**4
        + a**2
        * x**2
        * (
            2 * r_plus * x**2 * (-1 + 2j * w)
            - (s + 3) * x**2
            + 2 * x * (s + 1j * w + 2)
            - 4j * w
        )
        + 2j * a * m * (x - 1) * x**2
        + (x - 1)
        * (
            2 * r_plus * x**2 * (1 - 2j * w)
            + (s + 1) * x**2
            - 2 * (s + 1) * x
            + 2j * w
        )
    )

    # F2 term:
    F2 = a**4 * x**6 - 2 * a**2 * (x - 1) * x**4 + (x - 1) ** 2 * x**2
    return F0, F1, F2


def G_terms(a, w, A, s, m, u):
    """
    Calculates The G_i terms defined in the Appendix A, each one with shape (u_x,1).
    Receives as arguments:
    - a - the spin parameter, value between 0 and 0.5 (float);
    - w - the frequency of the QNM (parameter of the Neural Network);
    - A - the separation constant of thr Teukolsky equation (parameter of the Neural Network).
    - s - spin-weight (for tensor pertubations s = -2);
    - m - Spherical harmonic indicies l and m;
    - u: vector with dimensions (N_u,1) that defines the angular space.
    """

    G0 = (
        4 * a**2 * (u**2 - 1) * w**2
        - 4
        * a
        * (u**2 - 1)
        * w
        * ((u - 1) * np.abs(m - s) + (u + 1) * np.abs(m + s) + 2 * (s + 1) * u)
        + 4 * (A * (u**2 - 1) + m**2 + 2 * m * s * u + s * ((s + 1) * u**2 - 1))
        - 2 * (u**2 - 1) * np.abs(m + s)
        - 2 * (u**2 - 1) * np.abs(m - s) * (np.abs(m + s) + 1)
        - (u - 1) ** 2 * (np.abs(m - s)) ** 2
        - (u + 1) ** 2 * (np.abs(m + s)) ** 2
    )

    G1 = -8 * a * (u**2 - 1) ** 2 * w - 4 * (u**2 - 1) * (
        (u - 1) * np.abs(m - s) + (u + 1) * np.abs(m + s) + 2 * u
    )

    G2 = -4 * (u**2 - 1) ** 2
    return G0, G1, G2


def train(cfg: DictConfig):
    class QNM(ppsci.equation.PDE):
        def __init__(self, m, s):
            super().__init__()
            self.m = m
            self.s = s

            def ode1(out) -> paddle.Tensor:
                x = out["x"]
                a = out["a"]
                # a = out["a"].unsqueeze(-1)
                w_real_param = out["w_r"]
                w_img_param = out["w_i"]
                A_real_param = out["A_r"]
                A_img_param = out["A_i"]
                f_real = out["f_r"]
                f_img = out["f_i"]

                F0, F1, F2 = F_terms(
                    a.numpy(),
                    paddle.as_complex(
                        paddle.stack([w_real_param, w_img_param], axis=-1)
                    ),
                    paddle.as_complex(
                        paddle.stack([A_real_param, A_img_param], axis=-1)
                    ),
                    self.s,
                    self.m,
                    x,
                )

                F0_r = F0.real()
                F0_i = F0.imag()
                F1_r = F1.real()
                F1_i = F1.imag()

                out["F0"] = F0
                out["F1"] = F1
                out["F2"] = F2
                out["F0_r"] = F0_r
                out["F0_i"] = F0_i
                out["F1_r"] = F1_r
                out["F1_i"] = F1_i

                tmp = (
                    F2 * hes(f_real, x)
                    + F1_r * jac(f_real, x)
                    - F1_i * jac(f_img, x)
                    + F0_r * f_real
                    - F0_i * f_img
                )
                return tmp

            self.add_equation("QNM1", ode1)  # Real part of F(x)

            def ode2(out) -> paddle.Tensor:
                x = out["x"]

                f_real = out["f_r"]
                f_img = out["f_i"]

                F2 = out["F2"]
                F0_r = out["F0_r"]
                F0_i = out["F0_i"]
                F1_r = out["F1_r"]
                F1_i = out["F1_i"]

                tmp = (
                    F2 * hes(f_img, x)
                    + F1_r * jac(f_img, x)
                    + F1_i * jac(f_real, x)
                    + F0_r * f_img
                    + F0_i * f_real
                )
                return tmp

            self.add_equation("QNM2", ode2)  # Imag part of F(x)

            def ode3(out) -> paddle.Tensor:
                u = out["u"]
                a = out["a"]
                # a = out["a"].unsqueeze(-1)
                w_real_param = out["w_r"]
                w_img_param = out["w_i"]
                A_real_param = out["A_r"]
                A_img_param = out["A_i"]
                g_real = out["g_r"]
                g_img = out["g_i"]

                G0, G1, G2 = G_terms(
                    a.numpy(),
                    paddle.as_complex(
                        paddle.stack([w_real_param, w_img_param], axis=-1)
                    ),
                    paddle.as_complex(
                        paddle.stack([A_real_param, A_img_param], axis=-1)
                    ),
                    self.s,
                    self.m,
                    u,
                )

                G0_r = G0.real()
                G0_i = G0.imag()
                G1_r = G1.real()
                G1_i = G1.imag()

                out["G0"] = G0
                out["G1"] = G1
                out["G2"] = G2
                out["G0_r"] = G0_r
                out["G0_i"] = G0_i
                out["G1_r"] = G1_r
                out["G1_i"] = G1_i

                tmp = (
                    G2 * hes(g_real, u)
                    + G1_r * jac(g_real, u)
                    - G1_i * jac(g_img, u)
                    + G0_r * g_real
                    - G0_i * g_img
                )
                return tmp

            self.add_equation("QNM3", ode3)  # Real part of G(u)

            def ode4(out) -> paddle.Tensor:
                u = out["u"]

                g_real = out["g_r"]
                g_img = out["g_i"]

                G2 = out["G2"]
                G0_r = out["G0_r"]
                G0_i = out["G0_i"]
                G1_r = out["G1_r"]
                G1_i = out["G1_i"]

                tmp = (
                    G2 * hes(g_img, u)
                    + G1_r * jac(g_img, u)
                    + G1_i * jac(g_real, u)
                    + G0_r * g_img
                    + G0_i * g_real
                )
                jac._clear()
                hes._clear()
                return tmp

            self.add_equation("QNM4", ode4)  # Imag part of G(u)

    ### Load eigen value data

    param_data = load_csv_file(
        file_path="data/qnm_100_row.csv",
        keys=(
            "n",
            "l",
            "m",
            "a",
            "w_r",
            "w_i",
            "A_r",
            "A_i",
        ),
    )
    # logger.message(f"train n list = {cfg.TRAIN.n_list}, train m list = {cfg.TRAIN.m_list}")
    selected_mask = np.isin(param_data["m"], cfg.TRAIN.m_list)

    param_data = {k: v[selected_mask].reshape(-1, 1) for k, v in param_data.items()}
    selected_mask2 = np.isin(param_data["n"], cfg.TRAIN.n_list)
    param_data = {k: v[selected_mask2].reshape(-1, 1) for k, v in param_data.items()}
    logger.info(
        f"Selected {selected_mask.astype('int32').sum()} lines data from {len(param_data['m'])} lines data"
    )

    a_list = param_data["a"]  # [a_num, 1]
    logger.info(f"a_list.shape = {a_list.shape}")
    w_real_list = param_data["w_r"]  # [a_num, 1]
    logger.info(f"w_real_list.shape = {w_real_list.shape}")
    w_img_list = param_data["w_i"]  # [a_num, 1]
    logger.info(f"w_img_list.shape = {w_img_list.shape}")
    A_real_list = param_data["A_r"]  # [a_num, 1]
    logger.info(f"A_real_list.shape = {A_real_list.shape}")
    A_img_list = param_data["A_i"]  # [a_num, 1]
    logger.info(f"A_img_list.shape = {A_img_list.shape}")
    n_list = param_data["n"]  # [a_num, 1]
    logger.info(f"n_list.shape = {n_list.shape}")
    m_list = param_data["m"]  # [a_num, 1]
    logger.info(f"m_list.shape = {m_list.shape}")

    #### Load supervised data
    fr_label = np.load("data/fr.npy").astype(dtype)
    fi_label = np.load("data/fi.npy").astype(dtype)
    gr_label = np.load("data/gr.npy").astype(dtype)
    gi_label = np.load("data/gi.npy").astype(dtype)
    print(f"loaded f label shape:{fr_label.shape}")

    slices1 = []
    slices2 = []
    slices3 = []
    slices4 = []
    if 0 in m_list:
        slices1.append(fr_label[0:88])
        slices2.append(fi_label[0:88])
        slices3.append(gr_label[0:88])
        slices4.append(gi_label[0:88])
    if 1 in m_list:
        slices1.append(fr_label[88:176])
        slices2.append(fi_label[88:176])
        slices3.append(gr_label[88:176])
        slices4.append(gi_label[88:176])
    if 2 in m_list:
        slices1.append(fr_label[176:])
        slices2.append(fi_label[176:])
        slices3.append(gr_label[176:])
        slices4.append(gi_label[176:])

    fr_label = np.concatenate(slices1, axis=0)
    fi_label = np.concatenate(slices2, axis=0)
    gr_label = np.concatenate(slices3, axis=0)
    gi_label = np.concatenate(slices4, axis=0)

    print(f"selected f label shape:{fr_label.shape}")
    # squeeze the supervised data to 2 dim to use autodiff_hes
    # shape: [a_num * sv_bs, 1], by default [264 * 128, 1]
    fr_label = fr_label.reshape([-1, 1])
    fi_label = fi_label.reshape([-1, 1])
    gr_label = gr_label.reshape([-1, 1])
    gi_label = gi_label.reshape([-1, 1])

    #### set model
    if cfg.model == "deeponet":
        from model import DeepONet

        model = DeepONet(**cfg.MODEL.DEEPONET)

    elif cfg.model == "deepopirate":
        from model import DeepOPirate

        model = DeepOPirate(**cfg.MODEL.PIRATE)

    elif cfg.model == "deepopirakan":
        from model import DeepOPirakan

        model = DeepOPirakan(**cfg.MODEL.PIRAKAN)
        for k, v in model.named_parameters():
            assert dtype in str(v.dtype)

    #### set supervised_data and pde constraint

    equation = {
        "QNM": QNM(
            cfg.m,
            cfg.s,
        )
    }

    # generate sv_input and label batch
    def gen_input_batch(a_list=a_list, n_list=n_list, m_list=m_list):

        indices = np.arange(len(a_list))
        cond1 = indices % 10 == 0
        cond2 = (indices + 1) % 100 == 0
        conds = cond1 | cond2
        a_list = a_list[conds]
        n_list = n_list[conds]
        m_list = m_list[conds]
        # logger.info(f"supervised a_list shape: {a_list.shape}")
        # logger.info(f"supervised n_list shape: {n_list.shape}")
        # logger.info(f"supervised m_list shape: {m_list.shape}")

        N_u = 128  # sv_data bs, by default = 128
        N_x = 128
        X = []
        U = []
        u_0 = -1.0
        u_1 = 1.0
        # u = paddle.uniform([N_u, 1], min=u_0, max=u_1)
        u = np.linspace(u_0, u_1, N_u, dtype=dtype).reshape(
            [-1, 1]
        )  # [sv_bs, 1] = [128, 1]
        x_0 = 0.0
        x_1 = 1.0
        # x = paddle.uniform([N_x, 1], min=x_0, max=x_1)
        x = np.linspace(x_0, x_1, N_x, dtype=dtype).reshape(
            [-1, 1]
        )  # [sv_bs, 1] = [128, 1]

        for a in a_list:
            X.append(x)  # repeat x and u for a_num times
            U.append(u)

        X = np.array(X)  # shape: [a_num, sv_bs, 1]
        U = np.array(U)
        a = np.tile(
            a_list[:, np.newaxis, :],
            (1, 128, cfg.MODEL.DEEPONET.branch_input_dim),
        )  # [a_num, sv_bs, 1]
        n = np.tile(
            n_list[:, np.newaxis, :],
            (1, 128, cfg.MODEL.DEEPONET.branch_input_dim),
        )  # [a_num, sv_bs, 1]
        m = np.tile(
            m_list[:, np.newaxis, :],
            (1, 128, cfg.MODEL.DEEPONET.branch_input_dim),
        )  # [a_num, sv_bs, 1]
        X = X.reshape([-1, 1])  # [a_num * sv_bs, 1] = [6400, 1]
        U = U.reshape([-1, 1])  # [a_num * sv_bs, 1] = [6400, 1]
        a = a.reshape([-1, a.shape[-1]])  # [a_num * sv_bs, 1] = [6400, 1]
        n = n.reshape([-1, n.shape[-1]])
        m = m.reshape([-1, m.shape[-1]])

        return {
            "x": X,
            "u": U,
            "a": a,
            "n": n,
            "m": m,
        }

    def gen_label_batch(input_batch):
        return {
            # shape: [a_num * sv_bs, 1], by default [6400, 1]
            "fr_label": fr_label,
            "fi_label": fi_label,
            "gr_label": gr_label,
            "gi_label": gi_label,
        }

    solution_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "ContinuousNamedArrayDataset",
                "input": gen_input_batch,
                "label": gen_label_batch,
            },
        },
        output_expr={
            "fr_label": lambda out: out["f_r"],
            "fi_label": lambda out: out["f_i"],
            "gr_label": lambda out: out["g_r"],
            "gi_label": lambda out: out["g_i"],
        },
        loss=ppsci.loss.MSELoss(
            "mean",
            weight={
                "fr_label": 2.0,
                "fi_lable": 2.0,
                "gr_label": 1.0,
                "gi_label": 1.0,
            },
        ),
        name="sv_solution",
    )

    # generate pde input and label batch
    def gen_pde_input_batch(a_list: np.ndarray = a_list):
        N_u = int(cfg.TRAIN.pde_batch_size)
        N_x = int(cfg.TRAIN.pde_batch_size)
        X = []
        U = []
        indices: np.ndarray = np.random.choice(
            len(a_list), cfg.TRAIN.a_num, replace=False
        )
        a_list_ = a_list[indices]
        for a in a_list_:
            u_0 = -1.0
            u_1 = 1.0
            u = np.random.uniform(low=u_0, high=u_1, size=[N_u, 1])  # [pde_bs, 1]
            # u = np.linspace(u_0, u_1, N_u, dtype=dtype).reshape([-1, 1])
            x_0 = 0.0
            x_1 = 1
            x = np.random.uniform(low=x_0, high=x_1, size=[N_x, 1])  # [pde_bs, 1]
            # x = np.linspace(x_0, x_1, N_x, dtype=dtype).reshape([-1, 1])
            X.append(x)
            U.append(u)
        X = np.array(X)  # [a_num, pde_bs, 1]
        U = np.array(U)  # [a_num, pde_bs, 1]

        a = np.tile(
            a_list_[:, np.newaxis, :],
            (1, cfg.TRAIN.pde_batch_size, cfg.MODEL.DEEPONET.branch_input_dim),
        )  # [a_num, pde_bs, num_sensors], by default num_sensors = 1

        w_r = np.tile(
            w_real_list[indices][:, np.newaxis, :], (1, cfg.TRAIN.pde_batch_size, 1)
        )  # [a_num, pde_bs, 1]
        w_i = np.tile(
            w_img_list[indices][:, np.newaxis, :], (1, cfg.TRAIN.pde_batch_size, 1)
        )  # [a_num, pde_bs, 1]
        A_r = np.tile(
            A_real_list[indices][:, np.newaxis, :], (1, cfg.TRAIN.pde_batch_size, 1)
        )  # [a_num, pde_bs, 1]
        A_i = np.tile(
            A_img_list[indices][:, np.newaxis, :], (1, cfg.TRAIN.pde_batch_size, 1)
        )  # [a_num, pde_bs, 1]
        n = np.tile(
            n_list[indices][:, np.newaxis, :], (1, cfg.TRAIN.pde_batch_size, 1)
        )  # [a_num, pde_bs, 1]
        m = np.tile(
            m_list[indices][:, np.newaxis, :], (1, cfg.TRAIN.pde_batch_size, 1)
        )  # [a_num, pde_bs, 1]

        X = X.reshape([-1, 1])  # [a_num * pde_bs, 1]
        U = U.reshape([-1, 1])  # [a_num * pde_bs, 1]
        a = a.reshape([-1, a.shape[-1]])  # [a_num * pde_bs, num_sensors]
        w_r = w_r.reshape([-1, 1])  # [a_num * pde_bs, 1]
        w_i = w_i.reshape([-1, 1])  # [a_num * pde_bs, 1]
        A_r = A_r.reshape([-1, 1])  # [a_num * pde_bs, 1]
        A_i = A_i.reshape([-1, 1])  # [a_num * pde_bs, 1]
        n = n.reshape([-1, 1])  # [a_num * pde_bs, 1]
        m = m.reshape([-1, 1])  # [a_num * pde_bs, 1]

        return {
            "x": X,
            "u": U,
            "a": a,
            "w_r": w_r,
            "w_i": w_i,
            "A_r": A_r,
            "A_i": A_i,
            "n": n,
            "m": m,
        }

    def gen_pde_label_batch(input_batch):
        return {
            "QNM1": np.zeros(
                [cfg.TRAIN.a_num * cfg.TRAIN.pde_batch_size, 1], dtype=dtype
            ),  # [a_num * pde_bs, 1]
            "QNM2": np.zeros(
                [cfg.TRAIN.a_num * cfg.TRAIN.pde_batch_size, 1], dtype=dtype
            ),  # [a_num * pde_bs, 1]
            "QNM3": np.zeros(
                [cfg.TRAIN.a_num * cfg.TRAIN.pde_batch_size, 1], dtype=dtype
            ),  # [a_num * pde_bs, 1]
            "QNM4": np.zeros(
                [cfg.TRAIN.a_num * cfg.TRAIN.pde_batch_size, 1], dtype=dtype
            ),  # [a_num * pde_bs, 1]
        }

    pde_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "ContinuousNamedArrayDataset",
                "input": gen_pde_input_batch,
                "label": gen_pde_label_batch,
            },
        },
        output_expr={
            **equation["QNM"].equations,
        },
        loss=ppsci.loss.MAELoss(
            "mean",
            weight={
                "ode1": 2.0,
                "ode2": 2.0,
                "ode3": 1.0,
                "ode4": 1.0,
            },
        ),
        name="PDE",
    )

    """
        the Boundary Conditions had been hard-enforced in the forward process of the network,
        the following bc constraint can be removed
    """
    # wrap constraints together
    constraint = {
        solution_constraint.name: solution_constraint,
        pde_constraint.name: pde_constraint,
    }

    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        **cfg.TRAIN.lr_scheduler
    )()

    if cfg.TRAIN.optim == "adam":
        optimizer_warmup = ppsci.optimizer.Adam(lr_scheduler)((model,))
    elif cfg.TRAIN.optim == "soap":
        optimizer_warmup = ppsci.optimizer.SOAP(lr_scheduler)((model,))
    else:
        raise ValueError(
            f"cfg.TRAIN.optim should be in ['adam','soap'], but got '{cfg.TRAIN.optim}'."
        )

    cfg_warmup = copy.deepcopy(cfg)
    cfg_warmup.TRAIN.epochs = 5
    solver_warmup = ppsci.solver.Solver(
        model,
        constraint,
        optimizer=optimizer_warmup,
        equation=equation,
        cfg=cfg_warmup,
    )

    def convert_to_paddle_tensor(in_dict):
        paddle_dict = {}
        for key, value in in_dict.items():
            paddle_dict[key] = paddle.to_tensor(value, dtype=dtype)
        return paddle_dict

    def plot_solutions_per_epoch(slv: ppsci.solver.Solver):
        if (slv.epoch_id == 1) or (slv.epoch_id % 100 == 0):
            in_dict = gen_pde_input_batch()
            paddle_dict = convert_to_paddle_tensor(in_dict)
            pred_dict = model(paddle_dict)
            a = in_dict["a"]
            m = in_dict["m"]
            n = in_dict["n"]
            w_r = in_dict["w_r"]
            w_i = in_dict["w_i"]
            x = in_dict["x"]
            u = in_dict["u"]
            f_r = pred_dict["f_r"].numpy()
            f_i = pred_dict["f_i"].numpy()
            g_r = pred_dict["g_r"].numpy()
            g_i = pred_dict["g_i"].numpy()

            fig, axs = plt.subplots(2, 2, figsize=(12, 10))

            # f_r vs x
            axs[0, 0].scatter(x[:128], f_r[:128], c="b", s=6)
            axs[0, 0].set_title("Real part of f(x)")
            axs[0, 0].set_xlabel("x")
            axs[0, 0].set_ylabel("f_r(x)")
            axs[0, 0].grid(True)

            # f_i vs x
            axs[0, 1].scatter(x[:128], f_i[:128], c="r", s=6)
            axs[0, 1].set_title("Imaginary part of f(x)")
            axs[0, 1].set_xlabel("x")
            axs[0, 1].set_ylabel("f_i(x)")
            axs[0, 1].grid(True)

            # g_r vs u
            axs[1, 0].scatter(u[:128], g_r[:128], c="g", s=6)
            axs[1, 0].set_title("Real part of g(u)")
            axs[1, 0].set_xlabel("u")
            axs[1, 0].set_ylabel("g_r(u)")
            axs[1, 0].grid(True)

            # g_i vs u
            axs[1, 1].scatter(u[:128], g_i[:128], c="m", s=6)
            axs[1, 1].set_title("Imaginary part of g(u)")
            axs[1, 1].set_xlabel("u")
            axs[1, 1].set_ylabel("g_i(u)")
            axs[1, 1].grid(True)

            plt.savefig(
                osp.join(
                    cfg.output_dir,
                    f"epoch{slv.epoch_id}_m_n_a={m[0,0]}_{n[0,0]}_{a[0,0]:.4f}_w={w_r[0,0]:.4f}+{w_i[0,0]:.4f}*i_f(x)g(u).png",
                ),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    # solver_warmup.register_callback_on_epoch_end(plot_solutions_per_epoch)

    solver_warmup.train()


def evaluate(cfg: DictConfig):
    import pandas as pd

    class QNM(ppsci.equation.PDE):
        def __init__(
            self, A_real_param, A_img_param, w_real_param, w_img_param, a, m, s
        ):
            super().__init__()
            self.w = (w_real_param, w_img_param)
            self.A = (A_real_param, A_img_param)
            self.a = a
            self.m = m
            self.s = s
            # self.train_params = False
            self.learnable_parameters.append(w_real_param)
            self.learnable_parameters.append(w_img_param)
            self.learnable_parameters.append(A_real_param)
            self.learnable_parameters.append(A_img_param)
            self.iter = 0

            def ode1(out) -> paddle.Tensor:
                x = out["x"]  # .squeeze(0)
                f_real = out["f_r"]  # .squeeze(0)
                f_img = out["f_i"]  # .squeeze(0)

                F0, F1, F2 = F_terms(
                    self.a,
                    paddle.as_complex(paddle.stack([w_real_param, w_img_param])),
                    paddle.as_complex(paddle.stack([A_real_param, A_img_param])),
                    self.s,
                    self.m,
                    x,
                )

                F0_r = F0.real()
                F0_i = F0.imag()
                F1_r = F1.real()
                F1_i = F1.imag()

                out["F0"] = F0
                out["F1"] = F1
                out["F2"] = F2
                out["F0_r"] = F0_r
                out["F0_i"] = F0_i
                out["F1_r"] = F1_r
                out["F1_i"] = F1_i

                tmp = (
                    F2 * hes(f_real, x)
                    + F1_r * jac(f_real, x)
                    - F1_i * jac(f_img, x)
                    + F0_r * f_real
                    - F0_i * f_img
                )
                return tmp

            self.add_equation("QNM1", ode1)  # Real part of F(x)

            def ode2(out) -> paddle.Tensor:
                x = out["x"]
                f_real = out["f_r"]
                f_img = out["f_i"]

                F2 = out["F2"]
                F0_r = out["F0_r"]
                F0_i = out["F0_i"]
                F1_r = out["F1_r"]
                F1_i = out["F1_i"]

                self.iter += 1

                if self.iter % 100 == 0:
                    logger.message(
                        f"a = {self.a}, m = {self.m}, Learned: w = {self.w[0]:.5f} + j*{self.w[1]:.5f}, A = {self.A[0]:.5f} + j*{self.A[1]:.5f}"
                    )
                tmp = (
                    F2 * hes(f_img, x)
                    + F1_r * jac(f_img, x)
                    + F1_i * jac(f_real, x)
                    + F0_r * f_img
                    + F0_i * f_real
                )
                return tmp

            self.add_equation("QNM2", ode2)  # Imag part of F(x)

            def ode3(out) -> paddle.Tensor:
                u = out["u"]
                g_real = out["g_r"]
                g_img = out["g_i"]

                G0, G1, G2 = G_terms(
                    self.a,
                    paddle.as_complex(paddle.stack([w_real_param, w_img_param])),
                    paddle.as_complex(paddle.stack([A_real_param, A_img_param])),
                    self.s,
                    self.m,
                    u,
                )

                G0_r = G0.real()
                G0_i = G0.imag()
                G1_r = G1.real()
                G1_i = G1.imag()

                out["G0"] = G0
                out["G1"] = G1
                out["G2"] = G2
                out["G0_r"] = G0_r
                out["G0_i"] = G0_i
                out["G1_r"] = G1_r
                out["G1_i"] = G1_i

                tmp = (
                    G2 * hes(g_real, u)
                    + G1_r * jac(g_real, u)
                    - G1_i * jac(g_img, u)
                    + G0_r * g_real
                    - G0_i * g_img
                )
                return tmp

            self.add_equation("QNM3", ode3)  # Real part of G(u)

            def ode4(out) -> paddle.Tensor:
                u = out["u"]
                g_img = out["g_i"]
                g_real = out["g_r"]

                G2 = out["G2"]
                G0_r = out["G0_r"]
                G0_i = out["G0_i"]
                G1_r = out["G1_r"]
                G1_i = out["G1_i"]

                tmp = (
                    G2 * hes(g_img, u)
                    + G1_r * jac(g_img, u)
                    + G1_i * jac(g_real, u)
                    + G0_r * g_img
                    + G0_i * g_real
                )
                jac._clear()
                hes._clear()
                return tmp

            self.add_equation("QNM4", ode4)  # Imag part of G(u)

    l = cfg.l
    s = cfg.s
    A_real_param = paddle.create_parameter(
        [],
        dtype=dtype,
        default_initializer=paddle.nn.initializer.Constant(
            float(l * (l + 1) - s * (s + 1))
        ),
    )
    A_img_param = paddle.create_parameter(
        [], dtype=dtype, default_initializer=paddle.nn.initializer.Constant(cfg.A_img)
    )
    w_real_param = paddle.create_parameter(
        [], dtype=dtype, default_initializer=paddle.nn.initializer.Constant(cfg.w_real)
    )
    w_img_param = paddle.create_parameter(
        [],
        dtype=dtype,
        default_initializer=paddle.nn.initializer.Constant(cfg.w_img),
    )
    w_img_param = paddle.create_parameter(
        [], dtype=dtype, default_initializer=paddle.nn.initializer.Constant(cfg.w_img)
    )

    if cfg.model == "deeponet":
        from model import DeepONet

        model = DeepONet(**cfg.MODEL.DEEPONET)
    elif cfg.model == "deepopirate":
        from model import DeepOPirate

        model = DeepOPirate(**cfg.MODEL.PIRATE)
    elif cfg.model == "deepopirakan":
        from model import DeepOPirakan

        model = DeepOPirakan(**cfg.MODEL.PIRAKAN)

    # load pretrained model
    save_load.load_pretrain(
        model=model,
        path=cfg.EVAL.pretrained_model_path,
    )
    model.freeze()
    # model.train()

    results = []
    output_csv = osp.join(cfg.output_dir, "eval_results.csv")
    os.makedirs(
        os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".",
        exist_ok=True,
    )

    ### Load eigen value data
    param_data = load_csv_file(
        file_path="data/qnm_100_row.csv",
        keys=(
            "n",
            "l",
            "m",
            "a",
            "w_r",
            "w_i",
            "A_r",
            "A_i",
        ),
    )
    # logger.message(f"train n list = {cfg.TRAIN.n_list}, train m list = {cfg.TRAIN.m_list}")
    selected_mask = np.isin(param_data["m"], cfg.TRAIN.m_list)

    param_data = {k: v[selected_mask].reshape(-1, 1) for k, v in param_data.items()}
    selected_mask2 = np.isin(param_data["n"], cfg.TRAIN.n_list)
    param_data = {k: v[selected_mask2].reshape(-1, 1) for k, v in param_data.items()}
    logger.info(
        f"Selected {selected_mask.astype('int32').sum()} lines data from {len(param_data['m'])} lines data"
    )

    a_list = param_data["a"]  # [a_num, 1]
    logger.info(f"a_list.shape = {a_list.shape}")
    w_real_list = param_data["w_r"]  # [a_num, 1]
    logger.info(f"w_real_list.shape = {w_real_list.shape}")
    w_img_list = param_data["w_i"]  # [a_num, 1]
    logger.info(f"w_img_list.shape = {w_img_list.shape}")
    A_real_list = param_data["A_r"]  # [a_num, 1]
    logger.info(f"A_real_list.shape = {A_real_list.shape}")
    A_img_list = param_data["A_i"]  # [a_num, 1]
    logger.info(f"A_img_list.shape = {A_img_list.shape}")
    n_list = param_data["n"]  # [a_num, 1]
    logger.info(f"n_list.shape = {n_list.shape}")
    m_list = param_data["m"]  # [a_num, 1]
    logger.info(f"m_list.shape = {m_list.shape}")

    # solve eigenvalues of input list of a with curriculum learning
    w_real_pred = []
    w_img_pred = []
    A_real_pred = []
    A_img_pred = []

    def train_curriculum(cfg, idx):
        cfg_t = copy.deepcopy(cfg)
        a = a_list[idx]
        n = n_list[idx]
        m = m_list[idx]
        l = cfg_t.l
        s = cfg_t.s

        # if idx == 0:
        if a == 0:
            w_init = Kerrfinit(l=l, m=int(m), a=0, n=int(n), s=s)
            print(f"l={l},m={m},n={n} w init {w_init}")
            w_img = np.imag(w_init)
            w_real_param.set_value(paddle.to_tensor(0.7, dtype=dtype))
            w_img_param.set_value(paddle.to_tensor(w_img, dtype=dtype))
            A_real_param.set_value(paddle.to_tensor(4.0, dtype=dtype))
            A_img_param.set_value(paddle.to_tensor(0.0, dtype=dtype))

        # set pde constraint
        equation = {
            "QNM": QNM(
                A_real_param,
                A_img_param,
                w_real_param,
                w_img_param,
                a,
                m,
                cfg_t.s,
            )
        }

        def gen_input_batch(a=a, n=n, m=m):
            N_u = cfg_t.EVAL.pde_batch_size
            N_x = cfg_t.EVAL.pde_batch_size
            u_0 = -1.0
            u_1 = 1.0
            # u = np.random.uniform(low=u_0, high=u_1, size=[N_u, 1])
            u = np.linspace(u_0, u_1, N_u, dtype=dtype).reshape([-1, 1])

            x_0 = 0.0
            x_1 = 1
            # x = np.random.uniform(low=x_0, high=x_1, size=[N_x, 1])
            x = np.linspace(x_0, x_1, N_x, dtype=dtype).reshape([-1, 1])

            a = np.tile(a, (cfg_t.TRAIN.pde_batch_size, cfg_t.EVAL.num_sensors))
            n = np.tile(float(n), (cfg_t.TRAIN.pde_batch_size, cfg_t.EVAL.num_sensors))
            m = np.tile(float(m), (cfg_t.TRAIN.pde_batch_size, cfg_t.EVAL.num_sensors))

            return {
                "x": x,
                "u": u,
                "a": a,
                "n": n,
                "m": m,
            }

        def gen_label_batch(input_batch):
            return {
                "QNM1": np.zeros([cfg_t.TRAIN.pde_batch_size, 1], dtype=dtype),
                "QNM2": np.zeros([cfg_t.TRAIN.pde_batch_size, 1], dtype=dtype),
                "QNM3": np.zeros([cfg_t.TRAIN.pde_batch_size, 1], dtype=dtype),
                "QNM4": np.zeros([cfg_t.TRAIN.pde_batch_size, 1], dtype=dtype),
                # "f_panal": np.zeros([], dtype=dtype),
            }

        def convert_to_paddle_tensor(in_dict):
            paddle_dict = {}
            for key, value in in_dict.items():

                paddle_dict[key] = paddle.to_tensor(value, dtype=dtype)
            return paddle_dict

        pde_constraint = ppsci.constraint.SupervisedConstraint(
            {
                "dataset": {
                    "name": "ContinuousNamedArrayDataset",
                    "input": gen_input_batch,
                    "label": gen_label_batch,
                },
            },
            output_expr={
                **equation["QNM"].equations,
                # "wr_panal": lambda out: 0.01 * 1 / (w_real_param**2)
            },
            loss=ppsci.loss.MAELoss(
                "mean",
                weight={
                    "ode1": 10.0,
                    "ode2": 10.0,
                    "ode3": 1.0,
                    "ode4": 1.0,
                },
            ),
            name="PDE",
        )

        """
            the Boundary Conditions had been hard-enforced in the forward process of the network,
            the following bc constraint can be removed
        """
        # wrap constraints together
        constraint = {
            pde_constraint.name: pde_constraint,
        }

        # set optimizer
        lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
            **cfg.EVAL.lr_scheduler
        )()

        logger.message(
            f"n_l_m_a = {n}_{l}_{m}_{a}, Initial: w = {w_real_param.item()} + j*{w_img_param.item()}, A = {A_real_param.item()} + j*{A_img_param.item()}"
        )

        if cfg_t.EVAL.optim == "adam":
            optimizer = ppsci.optimizer.Adam(lr_scheduler)(
                # (model,) +
                tuple(equation.values())
            )
        elif cfg_t.EVAL.optim == "soap":
            optimizer = ppsci.optimizer.SOAP(lr_scheduler)(
                # (model,) +
                tuple(equation.values())
            )
        else:
            raise ValueError(
                f"cfg.TRAIN.optim should be in ['adam','soap'], but got '{cfg_t.TRAIN.optim}'."
            )

        output_dir = f"{cfg_t.output_dir}/"
        solver = ppsci.solver.Solver(
            model,
            constraint,
            output_dir=output_dir,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epochs=cfg_t.EVAL.epochs,
            iters_per_epoch=cfg_t.EVAL.iters_per_epoch,
            equation=equation,
            log_freq=cfg_t.log_freq,
            save_freq=0,
        )

        w_real_pred_last_epoch = []
        w_img_pred_last_epoch = []
        A_real_pred_last_epoch = []
        A_img_pred_last_epoch = []

        def save_params_per_iter(slv: ppsci.solver.Solver):
            if slv.epoch_id == slv.epochs:
                w_real_pred_last_epoch.append(w_real_param.item())
                w_img_pred_last_epoch.append(w_img_param.item())
                A_real_pred_last_epoch.append(A_real_param.item())
                A_img_pred_last_epoch.append(A_img_param.item())

        solver.register_callback_on_iter_begin(save_params_per_iter)

        def plot_solutions_per_epoch(slv: ppsci.solver.Solver):
            if slv.epoch_id == slv.epochs:
                in_dict = gen_input_batch()
                paddle_dict = convert_to_paddle_tensor(in_dict)
                pred_dict = model(paddle_dict)

                x = in_dict["x"]
                u = in_dict["u"]
                f_r = pred_dict["f_r"].numpy()
                f_i = pred_dict["f_i"].numpy()
                g_r = pred_dict["g_r"].numpy()
                g_i = pred_dict["g_i"].numpy()

                fig, axs = plt.subplots(2, 2, figsize=(12, 10))

                # f_r vs x
                axs[0, 0].plot(x, f_r, "b")
                axs[0, 0].set_title("Real part of f(x)")
                axs[0, 0].set_xlabel("x")
                axs[0, 0].set_ylabel("f_r(x)")
                axs[0, 0].grid(True)
                axs[0, 0].set_xlim(0.0, 1.0)

                # f_i vs x
                axs[0, 1].plot(x, f_i, "r")
                axs[0, 1].set_title("Imaginary part of f(x)")
                axs[0, 1].set_xlabel("x")
                axs[0, 1].set_ylabel("f_i(x)")
                axs[0, 1].grid(True)
                axs[0, 1].set_xlim(0.0, 1.0)

                # g_r vs u
                axs[1, 0].plot(u, g_r, "g")
                axs[1, 0].set_title("Real part of g(u)")
                axs[1, 0].set_xlabel("u")
                axs[1, 0].set_ylabel("g_r(u)")
                axs[1, 0].grid(True)

                # g_i vs u
                axs[1, 1].plot(u, g_i, "m")
                axs[1, 1].set_title("Imaginary part of g(u)")
                axs[1, 1].set_xlabel("u")
                axs[1, 1].set_ylabel("g_i(u)")
                axs[1, 1].grid(True)

                plt.savefig(
                    f"{cfg_t.output_dir}/epoch{slv.epoch_id}_a={a}_w={w_real_param.item():.5f}+{w_img_param.item():.5f}*i_f(x)g(u).png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

        # solver.register_callback_on_epoch_end(plot_solutions_per_epoch)

        solver.train()

        w_real_pred.append(float(np.mean(w_real_pred_last_epoch)))
        w_img_pred.append(float(np.mean(w_img_pred_last_epoch)))
        A_real_pred.append(float(np.mean(A_real_pred_last_epoch)))
        A_img_pred.append(float(np.mean(A_img_pred_last_epoch)))
        logger.info(
            f"a = {a}, m = {m}, n = {n}, w_real = {float(np.mean(w_real_pred_last_epoch)):.4f}±{float(np.std(w_real_pred_last_epoch)):.4f}"
        )
        logger.info(
            f"a = {a}, m = {m}, n = {n}, w_img = {float(np.mean(w_img_pred_last_epoch)):.4f}±{float(np.std(w_img_pred_last_epoch)):.4f}"
        )
        logger.info(
            f"a = {a}, m = {m}, n = {n}, A_real = {float(np.mean(A_real_pred_last_epoch)):.4f}±{float(np.std(A_real_pred_last_epoch)):.4f}"
        )
        logger.info(
            f"a = {a}, m = {m}, n = {n}, A_img = {float(np.mean(A_img_pred_last_epoch)):.4f}±{float(np.std(A_img_pred_last_epoch)):.4f}"
        )
        return (
            a,
            n,
            m,
            float(np.mean(w_real_pred_last_epoch)),
            float(np.std(w_real_pred_last_epoch)),
            float(np.mean(w_img_pred_last_epoch)),
            float(np.std(w_img_pred_last_epoch)),
            float(np.mean(A_real_pred_last_epoch)),
            float(np.std(A_real_pred_last_epoch)),
            float(np.mean(A_img_pred_last_epoch)),
            float(np.std(A_img_pred_last_epoch)),
        )

    for idx in range(len(a_list)):
        a, n, m, w_r, err_wr, w_i, err_wi, A_r, err_Ar, A_i, err_Ai = train_curriculum(
            cfg, idx
        )

        w_r_label = w_real_list[idx]
        w_i_label = w_img_list[idx]
        A_r_label = A_real_list[idx]
        A_i_label = A_img_list[idx]

        results.append(
            {
                "a": a,
                "n": n,
                "m": m,
                "w_r": w_r,
                "err_wr": err_wr,
                "w_i": w_i,
                "err_wi": err_wi,
                "A_r": A_r,
                "err_Ar": err_Ar,
                "A_i": A_i,
                "err_Ai": err_Ai,
                "w_r_label": w_r_label,
                "w_i_label": w_i_label,
                "A_r_label": A_r_label,
                "A_i_label": A_i_label,
            }
        )

        if (idx + 1) % 10 == 0 or idx == len(a_list) - 1:
            df_results = pd.DataFrame(results)
            df_results.to_csv(output_csv, index=False)

    w_real_pred = np.array(w_real_pred, dtype=dtype)
    w_img_pred = np.array(w_img_pred, dtype=dtype)
    A_real_pred = np.array(A_real_pred, dtype=dtype)
    A_img_pred = np.array(A_img_pred, dtype=dtype)

    assert w_real_pred.shape == w_real_list.shape
    assert w_img_pred.shape == w_img_list.shape

    w_real_mre_avg = np.mean(abs(w_real_pred - w_real_list) / (abs(w_real_list)))
    w_img_mre_avg = np.mean((abs(w_img_pred - w_img_list) / abs(w_img_list)))
    w_real_mre_avg = float(w_real_mre_avg)
    w_img_mre_avg = float(w_img_mre_avg)
    logger.message(
        f"mre_avg: w_real = {w_real_mre_avg:.5f}, w_img = {w_img_mre_avg:.5f}"
    )

    # Save Results to CSV
    result_dict = {
        "n": n_list.flatten(),  # [N,]
        "l": np.full_like(n_list, cfg.l).flatten(),
        "m": m_list.flatten(),
        "a": a_list.flatten(),
        # Ground truth
        "w_r_true": w_real_list.flatten(),
        "w_i_true": w_img_list.flatten(),
        "A_r_true": A_real_list.flatten(),
        "A_i_true": A_img_list.flatten(),
        # PINNs results
        "w_r_pred": w_real_pred.flatten(),
        "w_i_pred": w_img_pred.flatten(),
        "A_r_pred": A_real_pred.flatten(),
        "A_i_pred": A_img_pred.flatten(),
    }

    df_result = pd.DataFrame(result_dict)

    output_path = cfg.EVAL.get("output_csv_path", "eval_results.csv")
    df_result.to_csv(output_path, index=False)
    logger.info(f"Saved evaluation results to {output_path}")


@hydra.main(version_base=None, config_path="./conf", config_name="main_curriculum.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        return train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(
            f"cfg.mode should in ['train', 'eval', 'export', 'infer'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()
