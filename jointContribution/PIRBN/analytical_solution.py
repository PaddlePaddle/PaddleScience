import os

import matplotlib.pyplot as plt
import numpy as np
import paddle


def output_fig(train_obj, mu, b, right_by, activation_function, output_Kgg):
    plt.figure(figsize=(15, 9))
    rbn = train_obj.pirbn.rbn

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Comparisons between the network predictions and the ground truth.
    plt.subplot(2, 3, 1)
    ns = 1001
    dx = 1 / (ns - 1)
    xy = np.zeros((ns, 1)).astype(np.float32)
    for i in range(0, ns):
        xy[i, 0] = i * dx + right_by
    y = rbn(paddle.to_tensor(xy))
    y = y.numpy()
    y_true = np.sin(2 * mu * np.pi * xy)
    plt.plot(xy, y_true)
    plt.plot(xy, y, linestyle="--")
    plt.legend(["ground truth", "predict"])
    plt.xlabel("x")

    # Point-wise absolute error plot.
    plt.subplot(2, 3, 2)
    xy_y = np.abs(y_true - y)
    plt.plot(xy, xy_y)
    plt.ylim(top=np.max(xy_y))
    plt.ylabel("Absolute Error")
    plt.xlabel("x")

    # Loss history of the network during the training process.
    plt.subplot(2, 3, 3)
    loss_g = train_obj.loss_g
    x = range(len(loss_g))
    plt.yscale("log")
    plt.plot(x, loss_g)
    plt.plot(x, train_obj.loss_b)
    plt.legend(["Lg", "Lb"])
    plt.ylabel("Loss")
    plt.xlabel("Iteration")

    # Visualise NTK after initialisation, The normalised Kg at 0th iteration.
    plt.subplot(2, 3, 4)
    index = str(output_Kgg[0])
    K = train_obj.ntk_list[index].numpy()
    plt.imshow(K / (np.max(abs(K))), cmap="bwr", vmax=1, vmin=-1)
    plt.colorbar()
    plt.title(f"Kg at {index}-th iteration")
    plt.xlabel("Sample point index")

    # Visualise NTK after training, The normalised Kg at 2000th iteration.
    plt.subplot(2, 3, 5)
    index = str(output_Kgg[1])
    K = train_obj.ntk_list[index].numpy()
    plt.imshow(K / (np.max(abs(K))), cmap="bwr", vmax=1, vmin=-1)
    plt.colorbar()
    plt.title(f"Kg at {index}-th iteration")
    plt.xlabel("Sample point index")

    # The normalised Kg at 20000th iteration.
    plt.subplot(2, 3, 6)
    index = str(output_Kgg[2])
    K = train_obj.ntk_list[index].numpy()
    plt.imshow(K / (np.max(abs(K))), cmap="bwr", vmax=1, vmin=-1)
    plt.colorbar()
    plt.title(f"Kg at {index}-th iteration")
    plt.xlabel("Sample point index")

    plt.savefig(
        os.path.join(
            output_dir, f"sine_function_{mu}_{b}_{right_by}_{activation_function}.png"
        )
    )

    # Save data
    # scipy.io.savemat(os.path.join(output_dir, "out.mat"), {"NTK": a, "x": xy, "y": y})
