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


# I make my own newfig and savefig functions
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
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    x3: paddle.Tensor,
    y1: paddle.Tensor,
    y2: paddle.Tensor,
    y3: paddle.Tensor,
    xi1: paddle.Tensor,
    yi1: paddle.Tensor,
    xi2: paddle.Tensor,
    yi2: paddle.Tensor,
    xb: paddle.Tensor,
    yb: paddle.Tensor,
    u_pred: paddle.Tensor,
    u_exact: paddle.Tensor,
):
    save_path = "./result"
    os.makedirs(save_path, exist_ok=True)

    xi1 = xi1.numpy()
    yi1 = yi1.numpy()
    xi2 = xi2.numpy()
    yi2 = yi2.numpy()
    x_tot = np.concatenate([x1, x2, x3])
    y_tot = np.concatenate([y1, y2, y3])

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

    fig, ax = newfig(1.0, 1.1)
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
        xi1,
        yi1,
        "w-",
        markersize=2,
        label="Interface Pts",
    )
    plt.plot(
        xi2,
        yi2,
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
        xi1,
        yi1,
        "w-",
        markersize=2,
        label="Interface Pts",
    )
    plt.plot(
        xi2,
        yi2,
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
        paddle.abs(paddle.squeeze(u_exact) - u_pred.flatten()),
        100,
        cmap="jet",
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
        xi1,
        yi1,
        "w-",
        markersize=2,
        label="Interface Pts",
    )
    plt.plot(
        xi2,
        yi2,
        "w-",
        markersize=2,
        label="Interface Pts",
    )
    fig.set_size_inches(w=12, h=9)
    savefig(os.path.join(save_path, "XPINN_PoissonEq_Err"))
    plt.show()


pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or latex
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,  # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,  # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(1.0),  # default fig size of 0.9 textwidth
    "pgf.preamble": "\n".join(
        [
            r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts because your computer can handle it :)
            r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
        ]
    ),
}
mpl.rcParams.update(pgf_with_latex)
