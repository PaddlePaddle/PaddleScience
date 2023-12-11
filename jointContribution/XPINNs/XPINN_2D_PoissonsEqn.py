import os
import time

import matplotlib.pyplot as plt
import numpy as np
import paddle
import plotting
import scipy.io
from matplotlib import gridspec
from matplotlib import patches
from matplotlib import tri

import ppsci

# For the use of the second derivative: paddle.cos, paddle.exp
paddle.framework.core.set_prim_eager_enabled(True)

np.random.seed(1234)
paddle.seed(1234)


class XPINN(paddle.nn.Layer):
    # Initialize the class
    def __init__(self, layer_list):
        super().__init__()
        # Initialize NNs
        self.weights1, self.biases1, self.amplitudes1 = self.initialize_nn(
            layer_list[0], "layers1"
        )
        self.weights2, self.biases2, self.amplitudes2 = self.initialize_nn(
            layer_list[1], "layers2"
        )
        self.weights3, self.biases3, self.amplitudes3 = self.initialize_nn(
            layer_list[2], "layers3"
        )

    def preprocess_data(self, dataset):
        x_ub, ub, x_f1, x_f2, x_f3, x_fi1, x_fi2 = dataset
        self.x_ub = paddle.to_tensor(x_ub[:, 0:1], dtype=paddle.float64)
        self.y_ub = paddle.to_tensor(x_ub[:, 1:2], dtype=paddle.float64)
        self.ub = paddle.to_tensor(ub, dtype=paddle.float64)
        self.x_f1 = paddle.to_tensor(x_f1[:, 0:1], dtype=paddle.float64)
        self.y_f1 = paddle.to_tensor(x_f1[:, 1:2], dtype=paddle.float64)
        self.x_f2 = paddle.to_tensor(x_f2[:, 0:1], dtype=paddle.float64)
        self.y_f2 = paddle.to_tensor(x_f2[:, 1:2], dtype=paddle.float64)
        self.x_f3 = paddle.to_tensor(x_f3[:, 0:1], dtype=paddle.float64)
        self.y_f3 = paddle.to_tensor(x_f3[:, 1:2], dtype=paddle.float64)
        self.x_fi1 = paddle.to_tensor(x_fi1[:, 0:1], dtype=paddle.float64)
        self.y_fi1 = paddle.to_tensor(x_fi1[:, 1:2], dtype=paddle.float64)
        self.x_fi2 = paddle.to_tensor(x_fi2[:, 0:1], dtype=paddle.float64)
        self.y_fi2 = paddle.to_tensor(x_fi2[:, 1:2], dtype=paddle.float64)

    def forward(self, dataset):
        self.preprocess_data(dataset)
        self.ub1_pred = self.net_u1(self.x_ub, self.y_ub)
        self.ub2_pred = self.net_u2(self.x_f2, self.y_f2)
        self.ub3_pred = self.net_u3(self.x_f3, self.y_f3)

        (
            self.f1_pred,
            self.f2_pred,
            self.f3_pred,
            self.fi1_pred,
            self.fi2_pred,
            self.uavgi1_pred,
            self.uavgi2_pred,
            self.u1i1_pred,
            self.u1i2_pred,
            self.u2i1_pred,
            self.u3i2_pred,
        ) = self.net_f(
            self.x_f1,
            self.y_f1,
            self.x_f2,
            self.y_f2,
            self.x_f3,
            self.y_f3,
            self.x_fi1,
            self.y_fi1,
            self.x_fi2,
            self.y_fi2,
        )

        self.loss1 = (
            20 * paddle.mean(paddle.square(self.ub - self.ub1_pred))
            + paddle.mean(paddle.square(self.f1_pred))
            + 1 * paddle.mean(paddle.square(self.fi1_pred))
            + 1 * paddle.mean(paddle.square(self.fi2_pred))
            + 20 * paddle.mean(paddle.square(self.u1i1_pred - self.uavgi1_pred))
            + 20 * paddle.mean(paddle.square(self.u1i2_pred - self.uavgi2_pred))
        )

        self.loss2 = (
            paddle.mean(paddle.square(self.f2_pred))
            + 1 * paddle.mean(paddle.square(self.fi1_pred))
            + 20 * paddle.mean(paddle.square(self.u2i1_pred - self.uavgi1_pred))
        )

        self.loss3 = (
            paddle.mean(paddle.square(self.f3_pred))
            + 1 * paddle.mean(paddle.square(self.fi2_pred))
            + 20 * paddle.mean(paddle.square(self.u3i2_pred - self.uavgi2_pred))
        )
        return [self.loss1, self.loss2, self.loss3]

    def predict(self, x_star1, x_star2, x_star3):
        x_star1 = paddle.to_tensor(x_star1, dtype=paddle.float64)
        x_star2 = paddle.to_tensor(x_star2, dtype=paddle.float64)
        x_star3 = paddle.to_tensor(x_star3, dtype=paddle.float64)
        self.ub1_pred = self.net_u1(x_star1[:, 0:1], x_star1[:, 1:2])
        self.ub2_pred = self.net_u2(x_star2[:, 0:1], x_star2[:, 1:2])
        self.ub3_pred = self.net_u3(x_star3[:, 0:1], x_star3[:, 1:2])
        return [self.ub1_pred.numpy(), self.ub2_pred.numpy(), self.ub3_pred.numpy()]

    def initialize_nn(self, layers, name_prefix):
        # The weight used in neural_net
        weights = []
        # The bias used in neural_net
        biases = []
        # The amplitude used in neural_net
        amplitudes = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            weight = self.create_parameter(
                shape=[layers[l], layers[l + 1]],
                dtype="float64",
                default_initializer=self.w_init((layers[l], layers[l + 1])),
            )
            bias = self.create_parameter(
                shape=[1, layers[l + 1]],
                dtype="float64",
                is_bias=True,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            amplitude = self.create_parameter(
                shape=[1],
                dtype="float64",
                is_bias=True,
                default_initializer=paddle.nn.initializer.Constant(0.05),
            )

            self.add_parameter(name_prefix + "_w_" + str(l), weight)
            self.add_parameter(name_prefix + "_b_" + str(l), bias)
            self.add_parameter(name_prefix + "_a_" + str(l), amplitude)
            weights.append(weight)
            biases.append(bias)
            amplitudes.append(amplitude)
        return weights, biases, amplitudes

    def w_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        param = paddle.empty(size, "float64")
        param = ppsci.utils.initializer.trunc_normal_(param, 0.0, xavier_stddev)
        # TODO: Truncated normal and assign support float64
        return lambda p_ten, _: p_ten.set_value(param)

    def neural_net_tanh(self, x, weights, biases, amplitudes):
        num_layers = len(weights) + 1

        h = x
        for l in range(0, num_layers - 2):
            w = weights[l]
            b = biases[l]
            h = paddle.tanh(20 * amplitudes[l] * paddle.add(paddle.matmul(h, w), b))
        w = weights[-1]
        b = biases[-1]
        y = paddle.add(paddle.matmul(h, w), b)
        return y

    def neural_net_sin(self, x, weights, biases, amplitudes):
        num_layers = len(weights) + 1

        h = x
        for l in range(0, num_layers - 2):
            w = weights[l]
            b = biases[l]
            h = paddle.sin(20 * amplitudes[l] * paddle.add(paddle.matmul(h, w), b))
        w = weights[-1]
        b = biases[-1]
        y = paddle.add(paddle.matmul(h, w), b)
        return y

    def neural_net_cos(self, x, weights, biases, amplitudes):
        num_layers = len(weights) + 1

        h = x
        for l in range(0, num_layers - 2):
            w = weights[l]
            b = biases[l]
            h = paddle.cos(20 * amplitudes[l] * paddle.add(paddle.matmul(h, w), b))
        w = weights[-1]
        b = biases[-1]
        y = paddle.add(paddle.matmul(h, w), b)
        return y

    def net_u1(self, x, y):
        return self.neural_net_tanh(
            paddle.concat([x, y], 1), self.weights1, self.biases1, self.amplitudes1
        )

    def net_u2(self, x, y):
        return self.neural_net_sin(
            paddle.concat([x, y], 1), self.weights2, self.biases2, self.amplitudes2
        )

    def net_u3(self, x, y):
        return self.neural_net_cos(
            paddle.concat([x, y], 1), self.weights3, self.biases3, self.amplitudes3
        )

    def get_grad(self, outputs, inputs):
        grad = paddle.grad(outputs, inputs, retain_graph=True, create_graph=True)
        return grad[0]

    def net_f(self, x1, y1, x2, y2, x3, y3, xi1, yi1, xi2, yi2):
        # Gradients need to be calculated
        x1.stop_gradient = False
        y1.stop_gradient = False
        x2.stop_gradient = False
        y2.stop_gradient = False
        x3.stop_gradient = False
        y3.stop_gradient = False
        xi1.stop_gradient = False
        yi1.stop_gradient = False
        xi2.stop_gradient = False
        yi2.stop_gradient = False

        # Sub-Net1
        u1 = self.net_u1(x1, y1)
        u1_x = self.get_grad(u1, x1)
        u1_y = self.get_grad(u1, y1)
        u1_xx = self.get_grad(u1_x, x1)
        u1_yy = self.get_grad(u1_y, y1)

        # Sub-Net2
        u2 = self.net_u2(x2, y2)
        u2_x = self.get_grad(u2, x2)
        u2_y = self.get_grad(u2, y2)
        u2_xx = self.get_grad(u2_x, x2)
        u2_yy = self.get_grad(u2_y, y2)

        # Sub-Net3
        u3 = self.net_u3(x3, y3)
        u3_x = self.get_grad(u3, x3)
        u3_y = self.get_grad(u3, y3)
        u3_xx = self.get_grad(u3_x, x3)
        u3_yy = self.get_grad(u3_y, y3)

        # Sub-Net1, Interface 1
        u1i1 = self.net_u1(xi1, yi1)
        u1i1_x = self.get_grad(u1i1, xi1)
        u1i1_y = self.get_grad(u1i1, yi1)
        u1i1_xx = self.get_grad(u1i1_x, xi1)
        u1i1_yy = self.get_grad(u1i1_y, yi1)

        # Sub-Net2, Interface 1
        u2i1 = self.net_u2(xi1, yi1)
        u2i1_x = self.get_grad(u2i1, xi1)
        u2i1_y = self.get_grad(u2i1, yi1)
        u2i1_xx = self.get_grad(u2i1_x, xi1)
        u2i1_yy = self.get_grad(u2i1_y, yi1)

        # Sub-Net1, Interface 2
        u1i2 = self.net_u1(xi2, yi2)
        u1i2_x = self.get_grad(u1i2, xi2)
        u1i2_y = self.get_grad(u1i2, yi2)
        u1i2_xx = self.get_grad(u1i2_x, xi2)
        u1i2_yy = self.get_grad(u1i2_y, yi2)

        # Sub-Net3, Interface 2
        u3i2 = self.net_u3(xi2, yi2)
        u3i2_x = self.get_grad(u3i2, xi2)
        u3i2_y = self.get_grad(u3i2, yi2)
        u3i2_xx = self.get_grad(u3i2_x, xi2)
        u3i2_yy = self.get_grad(u3i2_y, yi2)

        # Average value (Required for enforcing the average solution along the interface)
        uavgi1 = (u1i1 + u2i1) / 2
        uavgi2 = (u1i2 + u3i2) / 2

        # Residuals
        f1 = u1_xx + u1_yy - (paddle.exp(x1) + paddle.exp(y1))
        f2 = u2_xx + u2_yy - (paddle.exp(x2) + paddle.exp(y2))
        f3 = u3_xx + u3_yy - (paddle.exp(x3) + paddle.exp(y3))

        # Residual continuity conditions on the interfaces
        fi1 = (u1i1_xx + u1i1_yy - (paddle.exp(xi1) + paddle.exp(yi1))) - (
            u2i1_xx + u2i1_yy - (paddle.exp(xi1) + paddle.exp(yi1))
        )
        fi2 = (u1i2_xx + u1i2_yy - (paddle.exp(xi2) + paddle.exp(yi2))) - (
            u3i2_xx + u3i2_yy - (paddle.exp(xi2) + paddle.exp(yi2))
        )

        return f1, f2, f3, fi1, fi2, uavgi1, uavgi2, u1i1, u1i2, u2i1, u3i2


class Trainer:
    def __init__(self, layer_list, dataset):
        self.model = XPINN(layer_list)
        self.optimizer = paddle.optimizer.Adam(
            learning_rate=0.0008, parameters=self.model.parameters()
        )
        self.dataset = dataset

    def train(self, n_iter, x_star1, x_star2, x_star3, u_exact2, u_exact3):
        mse_history1 = []
        mse_history2 = []
        mse_history3 = []
        l2_err2 = []
        l2_err3 = []

        for it in range(n_iter):
            loss1_value, loss2_value, loss3_value = self.model(self.dataset)
            loss = loss1_value + loss2_value + loss3_value
            loss.backward()
            self.optimizer.step()
            self.optimizer.clear_grad()

            if it % 20 == 0:
                # Predicted solution
                _, u_pred2, u_pred3 = self.model.predict(x_star1, x_star2, x_star3)

                # Relative L2 error in subdomains 2 and 3
                l2_error2 = np.linalg.norm(u_exact2 - u_pred2, 2) / np.linalg.norm(
                    u_exact2, 2
                )
                l2_error3 = np.linalg.norm(u_exact3 - u_pred3, 2) / np.linalg.norm(
                    u_exact3, 2
                )

                print(
                    "It: %d, Loss1: %.3e, Loss2: %.3e, Loss3: %.3e, L2_err2: %.3e, L2_err3: %.3e"
                    % (it, loss1_value, loss2_value, loss3_value, l2_error2, l2_error3)
                )

                mse_history1.append(loss1_value)
                mse_history2.append(loss2_value)
                mse_history3.append(loss3_value)
                l2_err2.append(l2_error2)
                l2_err3.append(l2_error3)

        return mse_history1, mse_history2, mse_history3, l2_err2, l2_err3

    def predict(self, x_star1, x_star2, x_star3):
        return self.model.predict(x_star1, x_star2, x_star3)


if __name__ == "__main__":
    # Boundary points from subdomain 1
    n_ub = 200

    # Residual points in three subdomains
    n_f1 = 5000
    n_f2 = 1800
    n_f3 = 1200

    # Interface points along the two interfaces
    n_i1 = 100
    n_i2 = 100

    # NN architecture in each subdomain
    layers1 = [2, 30, 30, 1]
    layers2 = [2, 20, 20, 20, 20, 1]
    layers3 = [2, 25, 25, 25, 1]

    max_iter = 501

    # Load training data (boundary points), residual and interface points from .mat file
    # All points are generated in Matlab
    data = scipy.io.loadmat("./data/XPINN_2D_PoissonEqn.mat")

    x_f1 = data["x_f1"].flatten()[:, None]
    y_f1 = data["y_f1"].flatten()[:, None]
    x_f2 = data["x_f2"].flatten()[:, None]
    y_f2 = data["y_f2"].flatten()[:, None]
    x_f3 = data["x_f3"].flatten()[:, None]
    y_f3 = data["y_f3"].flatten()[:, None]
    xi1 = data["xi1"].flatten()[:, None]
    yi1 = data["yi1"].flatten()[:, None]
    xi2 = data["xi2"].flatten()[:, None]
    yi2 = data["yi2"].flatten()[:, None]
    xb = data["xb"].flatten()[:, None]
    yb = data["yb"].flatten()[:, None]

    ub_train = data["ub"].flatten()[:, None]
    u_exact = data["u_exact"].flatten()[:, None]
    u_exact2 = data["u_exact2"].flatten()[:, None]
    u_exact3 = data["u_exact3"].flatten()[:, None]

    x_f1_train = np.hstack((x_f1.flatten()[:, None], y_f1.flatten()[:, None]))
    x_f2_train = np.hstack((x_f2.flatten()[:, None], y_f2.flatten()[:, None]))
    x_f3_train = np.hstack((x_f3.flatten()[:, None], y_f3.flatten()[:, None]))

    x_fi1_train = np.hstack((xi1.flatten()[:, None], yi1.flatten()[:, None]))
    x_fi2_train = np.hstack((xi2.flatten()[:, None], yi2.flatten()[:, None]))

    x_ub_train = np.hstack((xb.flatten()[:, None], yb.flatten()[:, None]))

    # Points in the whole domain
    x_total = data["x_total"].flatten()[:, None]
    y_total = data["y_total"].flatten()[:, None]

    x_star1 = np.hstack((x_f1.flatten()[:, None], y_f1.flatten()[:, None]))
    x_star2 = np.hstack((x_f2.flatten()[:, None], y_f2.flatten()[:, None]))
    x_star3 = np.hstack((x_f3.flatten()[:, None], y_f3.flatten()[:, None]))

    # Randomly select the residual points from sub-domains
    idx1 = np.random.choice(x_f1_train.shape[0], n_f1, replace=False)
    x_f1_train = x_f1_train[idx1, :]

    idx2 = np.random.choice(x_f2_train.shape[0], n_f2, replace=False)
    x_f2_train = x_f2_train[idx2, :]

    idx3 = np.random.choice(x_f3_train.shape[0], n_f3, replace=False)
    x_f3_train = x_f3_train[idx3, :]

    # Randomly select boundary points
    idx4 = np.random.choice(x_ub_train.shape[0], n_ub, replace=False)
    x_ub_train = x_ub_train[idx4, :]
    ub_train = ub_train[idx4, :]

    # Randomly select the interface points along two interfaces
    idxi1 = np.random.choice(x_fi1_train.shape[0], n_i1, replace=False)
    x_fi1_train = x_fi1_train[idxi1, :]

    idxi2 = np.random.choice(x_fi2_train.shape[0], n_i2, replace=False)
    x_fi2_train = x_fi2_train[idxi2, :]

    layer_list = (
        layers1,
        layers2,
        layers3,
    )
    dataset = (
        x_ub_train,
        ub_train,
        x_f1_train,
        x_f2_train,
        x_f3_train,
        x_fi1_train,
        x_fi2_train,
    )

    trainer_obj = Trainer(
        layer_list,
        dataset,
    )

    # Training
    start_time = time.time()
    mse_hist1, mse_hist2, mse_hist3, l2_err2, l2_err3 = trainer_obj.train(
        max_iter, x_star1, x_star2, x_star3, u_exact2, u_exact3
    )
    elapsed = time.time() - start_time
    print("Training time: %.4f" % (elapsed))

    # Solution prediction
    u_pred1, u_pred2, u_pred3 = trainer_obj.predict(x_star1, x_star2, x_star3)

    # Needed for plotting
    x1, y1 = x_star1[:, 0:1], x_star1[:, 1:2]
    triang_1 = tri.Triangulation(x1.flatten(), y1.flatten())
    x2, y2 = x_star2[:, 0:1], x_star2[:, 1:2]
    triang_2 = tri.Triangulation(x2.flatten(), y2.flatten())
    x3, y3 = x_star3[:, 0:1], x_star3[:, 1:2]
    triang_3 = tri.Triangulation(x3.flatten(), y3.flatten())
    x_tot = np.concatenate([x1, x2, x3])
    y_tot = np.concatenate([y1, y2, y3])
    triang_total = tri.Triangulation(x_tot.flatten(), y_tot.flatten())

    # Concatenating the solution from subdomains
    u_pred = np.concatenate([u_pred1, u_pred2, u_pred3])

    error_u_total = np.linalg.norm(
        np.squeeze(u_exact) - u_pred.flatten(), 2
    ) / np.linalg.norm(np.squeeze(u_exact), 2)
    print("Error u_total: %e" % (error_u_total))

    ############################# Plotting ###############################
    if not os.path.exists("./target"):
        os.mkdir("./target")
    fig, ax = plotting.newfig(1.0, 1.1)
    plt.plot(range(1, max_iter + 1, 20), mse_hist1, "r-", linewidth=1, label="Sub-Net1")
    plt.plot(
        range(1, max_iter + 1, 20), mse_hist2, "b-.", linewidth=1, label="Sub-Net2"
    )
    plt.plot(
        range(1, max_iter + 1, 20), mse_hist3, "g--", linewidth=1, label="Sub-Net3"
    )
    plt.xlabel("$\#$ iterations")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend(loc="upper right")
    plotting.savefig("./target/XPINN_PoissonMSEhistory")

    fig, ax = plotting.newfig(1.0, 1.1)
    plt.plot(
        range(1, max_iter + 1, 20), l2_err2, "r-", linewidth=1, label="Subdomain 2"
    )
    plt.plot(
        range(1, max_iter + 1, 20), l2_err3, "b--", linewidth=1, label="Subdomain 3"
    )
    plt.xlabel("$\#$ iterations")
    plt.ylabel("Rel. $L_2$ error")
    plt.yscale("log")
    plt.legend(loc="upper right")
    plotting.savefig("./target/XPINN_PoissonErrhistory")

    aa1 = np.array([[np.squeeze(xb[-1]), np.squeeze(yb[-1])]])
    aa2 = np.array(
        [
            [1.8, np.squeeze(yb[-1])],
            [+1.8, -1.7],
            [-1.6, -1.7],
            [-1.6, 1.55],
            [1.8, 1.55],
            [1.8, np.squeeze(yb[-1])],
        ]
    )
    x_domain1 = np.squeeze(xb.flatten()[:, None])
    y_domain1 = np.squeeze(yb.flatten()[:, None])
    aa3 = np.array([x_domain1, y_domain1]).T
    xx = np.vstack((aa3, aa2, aa1))
    triang_total = tri.Triangulation(x_tot.flatten(), y_tot.flatten())

    x_fi1_train_plot = np.hstack((xi1.flatten()[:, None], yi1.flatten()[:, None]))
    x_fi2_train_plot = np.hstack((xi2.flatten()[:, None], yi2.flatten()[:, None]))

    fig, ax = plotting.newfig(1.0, 1.1)
    gridspec.GridSpec(1, 1)
    ax = plt.subplot2grid((1, 1), (0, 0))
    tcf = ax.tricontourf(triang_total, np.squeeze(u_exact), 100, cmap="jet")
    ax.add_patch(patches.Polygon(xx, closed=True, fill=True, color="w", edgecolor="w"))
    tcbar = fig.colorbar(tcf)
    tcbar.ax.tick_params(labelsize=28)
    ax.set_xlabel("$x$", fontsize=32)
    ax.set_ylabel("$y$", fontsize=32)
    ax.set_title("$u$ (Exact)", fontsize=34)
    ax.tick_params(axis="x", labelsize=28)
    ax.tick_params(axis="y", labelsize=28)
    plt.plot(
        x_fi1_train_plot[:, 0:1],
        x_fi1_train_plot[:, 1:2],
        "w-",
        markersize=2,
        label="Interface Pts",
    )
    plt.plot(
        x_fi2_train_plot[:, 0:1],
        x_fi2_train_plot[:, 1:2],
        "w-",
        markersize=2,
        label="Interface Pts",
    )
    fig.set_size_inches(w=12, h=9)
    plotting.savefig("./target/XPINN_PoissonEq_ExSol")
    plt.show()

    fig, ax = plotting.newfig(1.0, 1.1)
    gridspec.GridSpec(1, 1)
    ax = plt.subplot2grid((1, 1), (0, 0))
    tcf = ax.tricontourf(triang_total, u_pred.flatten(), 100, cmap="jet")
    ax.add_patch(patches.Polygon(xx, closed=True, fill=True, color="w", edgecolor="w"))
    tcbar = fig.colorbar(tcf)
    tcbar.ax.tick_params(labelsize=28)
    ax.set_xlabel("$x$", fontsize=32)
    ax.set_ylabel("$y$", fontsize=32)
    ax.set_title("$u$ (Predicted)", fontsize=34)
    ax.tick_params(axis="x", labelsize=28)
    ax.tick_params(axis="y", labelsize=28)
    plt.plot(
        x_fi1_train_plot[:, 0:1],
        x_fi1_train_plot[:, 1:2],
        "w-",
        markersize=2,
        label="Interface Pts",
    )
    plt.plot(
        x_fi2_train_plot[:, 0:1],
        x_fi1_train_plot[:, 1:2],
        "w-",
        markersize=2,
        label="Interface Pts",
    )
    fig.set_size_inches(w=12, h=9)
    plotting.savefig("./target/XPINN_PoissonEq_Sol")
    plt.show()

    fig, ax = plotting.newfig(1.0, 1.1)
    gridspec.GridSpec(1, 1)
    ax = plt.subplot2grid((1, 1), (0, 0))
    tcf = ax.tricontourf(
        triang_total, abs(np.squeeze(u_exact) - u_pred.flatten()), 100, cmap="jet"
    )
    ax.add_patch(patches.Polygon(xx, closed=True, fill=True, color="w", edgecolor="w"))
    tcbar = fig.colorbar(tcf)
    tcbar.ax.tick_params(labelsize=28)
    ax.set_xlabel("$x$", fontsize=32)
    ax.set_ylabel("$y$", fontsize=32)
    ax.set_title("Point-wise Error", fontsize=34)
    ax.tick_params(axis="x", labelsize=28)
    ax.tick_params(axis="y", labelsize=28)
    plt.plot(
        x_fi1_train_plot[:, 0:1],
        x_fi1_train_plot[:, 1:2],
        "w-",
        markersize=2,
        label="Interface Pts",
    )
    plt.plot(
        x_fi2_train_plot[:, 0:1],
        x_fi2_train_plot[:, 1:2],
        "w-",
        markersize=2,
        label="Interface Pts",
    )
    fig.set_size_inches(w=12, h=9)
    plotting.savefig("./target/XPINN_PoissonEq_Err")
    plt.show()

    fig, ax = plotting.newfig(1.0, 1.1)
    gridspec.GridSpec(1, 1)
    ax = plt.subplot2grid((1, 1), (0, 0))
    plt.plot(
        x_f1_train[:, 0:1],
        x_f1_train[:, 1:2],
        "r*",
        markersize=4,
        label="Residual Pts  (sub-domain 1)",
    )
    plt.plot(
        x_f2_train[:, 0:1],
        x_f2_train[:, 1:2],
        "yo",
        markersize=4,
        label="Residual Pts (sub-domain 2)",
    )
    plt.plot(
        x_f3_train[:, 0:1],
        x_f3_train[:, 1:2],
        "gs",
        markersize=4,
        label="Residual Pts (sub-domain 3)",
    )
    plt.plot(
        x_fi1_train[:, 0:1],
        x_fi1_train[:, 1:2],
        "bs",
        markersize=7,
        label="Interface Pts 1",
    )
    plt.plot(
        x_fi2_train[:, 0:1],
        x_fi2_train[:, 1:2],
        "bs",
        markersize=7,
        label="Interface Pts 1",
    )
    plt.plot(
        x_ub_train[:, 0:1],
        x_ub_train[:, 1:2],
        "kx",
        markersize=9,
        label="Interface Pts 1",
    )
    ax.set_xlabel("$x$", fontsize=30)
    ax.set_ylabel("$y$", fontsize=30)
    ax.tick_params(axis="x", labelsize=26)
    ax.tick_params(axis="y", labelsize=26)
    fig.set_size_inches(w=12, h=12)
    plotting.savefig("./target/XPINN_Poisson_dataPts")
    plt.show()
