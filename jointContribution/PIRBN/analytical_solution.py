import os

import matplotlib.pyplot as plt
import numpy as np
import paddle
import scipy.io


def output_fig(train_obj, mu, b):
    plt.figure(figsize=(15, 9))
    rbn = train_obj.pirbn.rbn

    target_dir = os.path.join(os.path.dirname(__file__), "../target")
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    # Comparisons between the PINN predictions and the ground truth.
    plt.subplot(2, 3, 1)
    ns = 1001
    dx = 1 / (ns - 1)
    xy = np.zeros((ns, 1)).astype(np.float32)
    for i in range(0, ns):
        xy[i, 0] = i * dx
    y = rbn(paddle.to_tensor(xy))
    y = y.numpy()
    y_true = np.sin(2 * mu * np.pi * xy)
    plt.plot(xy, y_true)
    plt.plot(xy, y, linestyle="--")
    plt.legend(["ground truth", "PINN"])

    # Point-wise absolute error plot.
    plt.subplot(2, 3, 2)
    plt.plot(xy, np.abs(y_true - y))
    plt.ylim(top=1e-3)
    plt.legend(["Absolute Error"])

    # Loss history of the PINN during the training process.
    plt.subplot(2, 3, 3)
    his_l1 = train_obj.his_l1
    x = range(len(his_l1))
    plt.yscale("log")
    plt.plot(x, his_l1)
    plt.plot(x, train_obj.his_l2)
    plt.legend(["Lg", "Lb"])

    # Visualise NTK after initialisation, The normalised Kg at 0th iteration.
    plt.subplot(2, 3, 4)
    jac = train_obj.ntk_list[0]
    a = np.dot(jac, np.transpose(jac))
    plt.imshow(a / (np.max(abs(a))), cmap="bwr", vmax=1, vmin=-1)
    plt.colorbar()

    # Visualise NTK after training, The normalised Kg at 2000th iteration.
    plt.subplot(2, 3, 5)
    if 2000 in train_obj.ntk_list:
        jac = train_obj.ntk_list[2000]
        a = np.dot(jac, np.transpose(jac))
        plt.imshow(a / (np.max(abs(a))), cmap="bwr", vmax=1, vmin=-1)
        plt.colorbar()

    # The normalised Kg at 20000th iteration.
    plt.subplot(2, 3, 6)
    if 20000 in train_obj.ntk_list:
        jac = train_obj.ntk_list[20000]
        a = np.dot(jac, np.transpose(jac))
        plt.imshow(a / (np.max(abs(a))), cmap="bwr", vmax=1, vmin=-1)
        plt.colorbar()

    plt.savefig(os.path.join(target_dir, f"sine_function_{mu}_{b}.png"))

    # Save data
    scipy.io.savemat(os.path.join(target_dir, "out.mat"), {"NTK": a, "x": xy, "y": y})
