import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import paddle
from matplotlib import gridspec
from matplotlib import patches
from matplotlib import tri


def figsize(scale: float, nplots: float = 1):
    fig_width_pt = 390.0  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = nplots * fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


def newfig(width: float, nplots: float = 1):
    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax


def savefig(filename: str, crop: bool = True):
    if crop:
        plt.savefig(f"{filename}.pdf", bbox_inches="tight", pad_inches=0)
        plt.savefig(f"{filename}.eps", bbox_inches="tight", pad_inches=0)
    else:
        plt.savefig(f"{filename}.pdf")
        plt.savefig(f"{filename}.eps")


def log_image(
    residual1_x: paddle.Tensor,
    residual1_y: paddle.Tensor,
    residual2_x: paddle.Tensor,
    residual2_y: paddle.Tensor,
    residual3_x: paddle.Tensor,
    residual3_y: paddle.Tensor,
    interface1_x: paddle.Tensor,
    interface1_y: paddle.Tensor,
    interface2_x: paddle.Tensor,
    interface2_y: paddle.Tensor,
    boundary_x: paddle.Tensor,
    boundary_y: paddle.Tensor,
    residual_u_pred: paddle.Tensor,
    residual_u_exact: paddle.Tensor,
):
    save_path = "./result"
    os.makedirs(save_path, exist_ok=True)

    interface1_x = interface1_x.numpy()
    interface1_y = interface1_y.numpy()
    interface2_x = interface2_x.numpy()
    interface2_y = interface2_y.numpy()
    x_tot = np.concatenate([residual1_x, residual2_x, residual3_x])
    y_tot = np.concatenate([residual1_y, residual2_y, residual3_y])

    aa1 = np.array([[np.squeeze(boundary_x[-1]), np.squeeze(boundary_y[-1])]])
    aa2 = np.array(
        [
            [1.8, np.squeeze(boundary_y[-1])],
            [+1.8, -1.7],
            [-1.6, -1.7],
            [-1.6, 1.55],
            [1.8, 1.55],
            [1.8, np.squeeze(boundary_y[-1])],
        ]
    )
    x_domain1 = np.squeeze(boundary_x.flatten()[:, None])
    y_domain1 = np.squeeze(boundary_y.flatten()[:, None])
    aa3 = np.array([x_domain1, y_domain1]).T
    xx = np.vstack((aa3, aa2, aa1))
    triang_total = tri.Triangulation(x_tot.flatten(), y_tot.flatten())

    fig, ax = newfig(1.0, 1.1)
    gridspec.GridSpec(1, 1)
    ax = plt.subplot2grid((1, 1), (0, 0))
    tcf = ax.tricontourf(triang_total, np.squeeze(residual_u_exact), 100, cmap="jet")
    ax.add_patch(
        patches.Polygon(xx, closed=True, fill=True, facecolor="w", edgecolor="w")
    )
    tcbar = fig.colorbar(tcf)
    tcbar.ax.tick_params(labelsize=28)
    ax.set_xlabel("$x$", fontsize=32)
    ax.set_ylabel("$y$", fontsize=32)
    ax.set_title("$u$ (Exact)", fontsize=34)
    ax.tick_params(axis="x", labelsize=28)
    ax.tick_params(axis="y", labelsize=28)
    plt.plot(
        interface1_x,
        interface1_y,
        "w-",
        markersize=2,
        label="Interface Pts",
    )
    plt.plot(
        interface2_x,
        interface2_y,
        "w-",
        markersize=2,
        label="Interface Pts",
    )
    fig.set_size_inches(w=12, h=9)
    savefig(os.path.join(save_path, "XPINN_PoissonEq_ExSol"))
    plt.show()

    fig, ax = newfig(1.0, 1.1)
    gridspec.GridSpec(1, 1)
    ax = plt.subplot2grid((1, 1), (0, 0))
    tcf = ax.tricontourf(triang_total, residual_u_pred.flatten(), 100, cmap="jet")
    ax.add_patch(
        patches.Polygon(xx, closed=True, fill=True, facecolor="w", edgecolor="w")
    )
    tcbar = fig.colorbar(tcf)
    tcbar.ax.tick_params(labelsize=28)
    ax.set_xlabel("$x$", fontsize=32)
    ax.set_ylabel("$y$", fontsize=32)
    ax.set_title("$u$ (Predicted)", fontsize=34)
    ax.tick_params(axis="x", labelsize=28)
    ax.tick_params(axis="y", labelsize=28)
    plt.plot(
        interface1_x,
        interface1_y,
        "w-",
        markersize=2,
        label="Interface Pts",
    )
    plt.plot(
        interface2_x,
        interface2_y,
        "w-",
        markersize=2,
        label="Interface Pts",
    )
    fig.set_size_inches(w=12, h=9)
    savefig(os.path.join(save_path, "XPINN_PoissonEq_Sol"))
    plt.show()

    fig, ax = newfig(1.0, 1.1)
    gridspec.GridSpec(1, 1)
    ax = plt.subplot2grid((1, 1), (0, 0))
    tcf = ax.tricontourf(
        triang_total,
        paddle.abs(residual_u_exact.flatten() - residual_u_pred.flatten()),
        100,
        cmap="jet",
    )
    ax.add_patch(
        patches.Polygon(xx, closed=True, fill=True, facecolor="w", edgecolor="w")
    )
    tcbar = fig.colorbar(tcf)
    tcbar.ax.tick_params(labelsize=28)
    ax.set_xlabel("$x$", fontsize=32)
    ax.set_ylabel("$y$", fontsize=32)
    ax.set_title("Point-wise Error", fontsize=34)
    ax.tick_params(axis="x", labelsize=28)
    ax.tick_params(axis="y", labelsize=28)
    plt.plot(
        interface1_x,
        interface1_y,
        "w-",
        markersize=2,
        label="Interface Pts",
    )
    plt.plot(
        interface2_x,
        interface2_y,
        "w-",
        markersize=2,
        label="Interface Pts",
    )
    fig.set_size_inches(w=12, h=9)
    savefig(os.path.join(save_path, "XPINN_PoissonEq_Err"))
    plt.show()


PGF_WITH_LATEX = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or latex
    # "text.usetex": True,  # use LaTeX to write all text
    # "font.family": "serif",
    # "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    # "font.sans-serif": [],
    # "font.monospace": [],
    "axes.labelsize": 10,  # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,  # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(1.0),  # default fig size of 0.9 textwidth
    "pgf.preamble": "\n".join(
        [
            r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts because your computer can handle it.
            r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
        ]
    ),
}
mpl.rcParams.update(PGF_WITH_LATEX)
