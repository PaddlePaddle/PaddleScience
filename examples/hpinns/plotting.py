# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import ticker

from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian

np.random.seed(2023)


class Plot:
    """All plotting functions."""

    def __init__(
        self, figname, OUTPUT_DIR, DATASET_PATH, DATASET_PATH_VALID, funcs_obj
    ):
        self.figname = figname
        self.save_dir = OUTPUT_DIR + "figure/"
        self.DATASET_PATH = DATASET_PATH
        self.DATASET_PATH_VALID = DATASET_PATH_VALID
        self.f = funcs_obj
        self.font = {"weight": "normal", "size": 10}
        self.input_name = ("x", "y")
        self.field_name = [
            "Fig7_E",
            "Fig7_eps",
            "Fig_6C_lambda_re_1",
            "Fig_6C_lambda_im_1",
            "Fig_6C_lambda_re_4",
            "Fig_6C_lambda_im_4",
            "Fig_6C_lambda_re_9",
            "Fig_6C_lambda_im_9",
        ]

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def prepare_data(self, solver):
        # train data
        train_dict = self.f.load_data_from_mat(self.DATASET_PATH)

        bound = int(train_dict["bound"])
        x_train = train_dict["x"][bound:]
        y_train = train_dict["y"][bound:]
        self.input_train = np.stack((x_train, y_train), axis=-1).reshape([-1, 2])
        # valid data
        N = ((self.f.l_BOX[1] - self.f.l_BOX[0]) / 0.05).astype(int)

        valid_dict = self.f.load_data_from_mat(self.DATASET_PATH_VALID)
        in_dict_val = {"x": valid_dict["x_val"], "y": valid_dict["y_val"]}
        self.f.init_lambda(in_dict_val, int(valid_dict["bound"]))

        expr_dict = {
            "x": lambda out: out["x"],
            "y": lambda out: out["y"],
            "e_real": lambda out: out["e_real"],
            "e_imaginary": lambda out: out["e_imaginary"],
            "epsilon": lambda out: out["epsilon"],
            "de_re_x": lambda out: jacobian(out["e_real"], out["x"]),
            "de_re_y": lambda out: jacobian(out["e_real"], out["y"]),
            "de_re_xx": lambda out: hessian(out["e_real"], out["x"]),
            "de_re_yy": lambda out: hessian(out["e_real"], out["y"]),
            "de_im_x": lambda out: jacobian(out["e_imaginary"], out["x"]),
            "de_im_y": lambda out: jacobian(out["e_imaginary"], out["y"]),
            "de_im_xx": lambda out: hessian(out["e_imaginary"], out["x"]),
            "de_im_yy": lambda out: hessian(out["e_imaginary"], out["y"]),
        }
        pred_dict_val = solver.predict(
            in_dict_val,
            expr_dict,
            batch_size=np.shape(valid_dict["x_val"])[0],
            no_grad=False,
        )

        self.input_valid = np.stack(
            (valid_dict["x_val"], valid_dict["y_val"]), axis=-1
        ).reshape([N[0], N[1], 2])
        self.output_valid = np.array(
            [
                pred_dict_val["e_real"].detach().cpu().numpy(),
                pred_dict_val["e_imaginary"].detach().cpu().numpy(),
                pred_dict_val["epsilon"].detach().cpu().numpy(),
            ]
        ).T.reshape([N[0], N[1], 3])

    def plot_6a(self, log_loss):
        plt.figure(300, figsize=(8, 6))
        smooth_step = 1  # how many steps of loss are squeezed to one point, num_points is epoch/smooth_step
        if log_loss.shape[0] % smooth_step is not 0:
            vis_loss_ = log_loss[: -(log_loss.shape[0] % smooth_step), :].reshape(
                [-1, smooth_step, log_loss.shape[1]]
            )
        else:
            vis_loss_ = log_loss.reshape([-1, smooth_step, log_loss.shape[1]])

        vis_loss = vis_loss_.mean(axis=1).reshape([-1, 3])
        vis_loss_total = vis_loss[:, :].sum(axis=1)
        vis_loss[:, 1] = vis_loss[:, 2]
        vis_loss[:, 2] = vis_loss_total
        for i in range(vis_loss.shape[1]):
            plt.semilogy(np.arange(vis_loss.shape[0]) * smooth_step, vis_loss[:, i])
        plt.legend(
            ["PDE loss", "Objective loss", "Total loss"],
            loc="lower left",
            prop=self.font,
        )
        plt.xlabel("Iteration ", fontdict=self.font)
        plt.ylabel("Loss ", fontdict=self.font)
        plt.grid()
        plt.yticks(size=10)
        plt.xticks(size=10)
        plt.savefig(os.path.join(self.save_dir, self.figname + "_Fig6_A.jpg"))

    def plot_6b(self, log_loss_obj):
        plt.figure(400, figsize=(10, 6))
        plt.clf()
        plt.plot(np.arange(len(log_loss_obj)), log_loss_obj, "bo-")
        plt.xlabel("k", fontdict=self.font)
        plt.ylabel("Objective", fontdict=self.font)
        plt.grid()
        plt.yticks(size=10)
        plt.xticks(size=10)
        plt.savefig(os.path.join(self.save_dir, self.figname + "_Fig6_B.jpg"))

    def plot_6c7c(self, log_lambda):
        input_valid = self.input_valid
        output_valid = self.output_valid
        input_train = self.input_train

        field_lambda = np.concatenate(
            [log_lambda[1], log_lambda[4], log_lambda[9]], axis=0
        ).T
        v_visual = output_valid[..., 0] ** 2 + output_valid[..., 1] ** 2
        field_visual = np.stack((v_visual, output_valid[..., -1]), axis=-1)
        coord_visual = input_valid
        coord_lambda = input_train
        self.plot_field_horo(coord_visual, field_visual, coord_lambda, field_lambda)

    def plot_6d(self, log_lambda):
        ############################################# lambda/mu #####################################################
        mu_ = 2 ** np.arange(1, 11)
        log_lambda = np.array(log_lambda) / mu_[:, None, None]
        ########################################### lambda/mu Fig 6 D #################################################
        # 随机挑选3个点 表示 lambdaRe lambdaIm 的迭代1-9的过程， 结论：lambda 收敛
        ind = np.random.randint(low=0, high=np.shape(log_lambda)[-1], size=3)
        la_mu_ind = log_lambda[:, :, ind]
        marker = [
            "ro-",
            "bo:",
            "r*-",
            "b*:",
            "rp-",
            "bp:",
        ]
        plt.figure(500, figsize=(7, 5))
        plt.clf()
        for i in range(6):
            plt.plot(
                np.arange(0, 10),
                la_mu_ind[:, int(i % 2), int(i / 2)],
                marker[i],
                linewidth=2,
            )
        plt.legend(
            [
                "Re, 1",
                "Im, 1",
                "Re, 2",
                "Im, 2",
                "Re, 3",
                "Im, 3",
            ],
            loc="upper right",
            prop=self.font,
        )
        plt.grid()
        plt.xlabel("k", fontdict=self.font)
        plt.ylabel(r"$ \lambda^k / \mu^k_F$", fontdict=self.font)
        plt.yticks(size=12)
        plt.xticks(size=12)
        plt.savefig(os.path.join(self.save_dir, self.figname + "_Fig6_D_lambda.jpg"))

    def plot_6ef(self, log_lambda):
        ############################################# lambda/mu #####################################################
        mu_ = 2 ** np.arange(1, 11)
        log_lambda = np.array(log_lambda) / mu_[:, None, None]
        ########################################### lamda/mu Fig 6E & Fig 6F #################################################
        # 5个迭代步中的lambda 分布情况
        iter_ind = [1, 4, 6, 9]  # 即1th, 4th, 6th, 9th 迭代
        # la_mu = np.array([log_lambda[1], log_lambda[4], log_lambda[6], log_lambda[9]])
        plt.figure(600, figsize=(5, 5))
        plt.clf()
        # for i in range(len(iter_ind)):
        # sns.kdeplot(
        #     la_mu[i, 0, :], label="k = " + str(iter_ind[i]), cut=0, linewidth=2
        # )
        for i in iter_ind:
            sns.kdeplot(log_lambda[i, 0, :], label="k = " + str(i), cut=0, linewidth=2)
        plt.legend(prop=self.font)
        plt.grid()
        plt.xlim([-0.1, 0.1])
        plt.xlabel(r"$ \lambda^k_{Re} / \mu^k_F$", fontdict=self.font)
        plt.ylabel("Frequency", fontdict=self.font)
        plt.yticks(size=12)
        plt.xticks(size=12)
        plt.savefig(os.path.join(self.save_dir, self.figname + "_Fig6_E.jpg"))

        plt.figure(700, figsize=(5, 5))
        plt.clf()
        # for i in range(len(iter_ind)):
        #     sns.kdeplot(
        #         la_mu[i, 1, :], label="k = " + str(iter_ind[i]), cut=0, linewidth=2
        #     )
        for i in iter_ind:
            sns.kdeplot(log_lambda[i, 1, :], label="k = " + str(i), cut=0, linewidth=2)
        plt.legend(prop=self.font)
        plt.grid()
        plt.xlim([-0.1, 0.1])
        plt.xlabel(r"$ \lambda^k_{Im} / \mu^k_F$", fontdict=self.font)
        plt.ylabel("Frequency", fontdict=self.font)
        # plt.rcParams['font.size'] = 2
        plt.yticks(size=12)
        plt.xticks(size=12)
        plt.savefig(os.path.join(self.save_dir, self.figname + "_Fig6_F.jpg"))

    def plot_field_horo(
        self, coord_visual, field_visual, coord_lambda, field_lambda, title=None
    ):
        fmin, fmax = np.array([0, 1.0]), np.array([0.6, 12])
        cmin, cmax = coord_visual.min(axis=(0, 1)), coord_visual.max(axis=(0, 1))
        emin, emax = np.array([-3, -1]), np.array([3, 0])
        x_pos = coord_visual[:, :, 0]
        y_pos = coord_visual[:, :, 1]

        for fi in range(len(self.field_name)):
            ########      Exact f(t,x,y)     ###########
            # plt.subplot(1, Num_fields,  0 * Num_fields + fi + 1)
            # plt.contour(x_pos, y_pos, f_true, levels=20, linestyles='-', linewidths=0.4, colors='k')
            if fi == 0:
                plt.figure(101, figsize=(8, 6))
                plt.clf()
                plt.rcParams["font.size"] = 20
                f_true = field_visual[..., fi]
                plt.pcolormesh(
                    x_pos,
                    y_pos,
                    f_true,
                    cmap="rainbow",
                    shading="gouraud",
                    antialiased=True,
                    snap=True,
                )
                cb = plt.colorbar()
                plt.axis((cmin[0], cmax[0], cmin[1], cmax[1]))
                plt.clim(vmin=fmin[fi], vmax=fmax[fi])
                # plt.axis('equal')
            elif fi == 1:
                plt.figure(201, figsize=(8, 1.5))
                plt.clf()
                plt.rcParams["font.size"] = 20
                f_true = field_visual[..., fi]
                plt.pcolormesh(
                    x_pos,
                    y_pos,
                    f_true,
                    cmap="rainbow",
                    shading="gouraud",
                    antialiased=True,
                    snap=True,
                )
                cb = plt.colorbar()
                # plt.axis('equal')
                plt.axis((emin[0], emax[0], emin[1], emax[1]))
                plt.clim(vmin=fmin[fi], vmax=fmax[fi])
            else:
                plt.figure(fi * 100 + 101, figsize=(8, 6))
                plt.clf()
                plt.rcParams["font.size"] = 20
                f_true = field_lambda[..., fi - 2]
                plt.scatter(
                    coord_lambda[..., 0],
                    coord_lambda[..., 1],
                    c=f_true,
                    cmap="rainbow",
                    alpha=0.6,
                )
                cb = plt.colorbar()
                plt.axis((cmin[0], cmax[0], cmin[1], cmax[1]))
                #

            # cb.set_label('$' + self.field_name[fi] + '$', rotation=0, fontdict=self.font, y=1.12)
            # 设置图例字体和大小
            cb.ax.tick_params(labelsize=20)
            # for l in cb.ax.yaxis.get_ticklabels():
            #     l.set_family("Times New Roman")
            tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
            cb.locator = tick_locator
            cb.update_ticks()

            plt.xlabel("$" + str(self.input_name[0]) + "$", fontdict=self.font)
            plt.ylabel("$" + str(self.input_name[1]) + "$", fontdict=self.font)
            plt.yticks(size=10)
            plt.xticks(size=10)
            plt.savefig(
                os.path.join(
                    self.save_dir,
                    self.figname + "_" + str(self.field_name[fi]) + ".jpg",
                )
            )
