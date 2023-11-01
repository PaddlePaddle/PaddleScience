from os import path as osp

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata


def draw_subplot(subfigname, figdata, fig, gs, cmap, boundary, loc):
    ax = plt.subplot(gs[:, loc])
    h = ax.imshow(
        figdata,
        interpolation="nearest",
        cmap=cmap,
        extent=boundary,  # [cfg.T_LB, cfg.T_UB, cfg.X_LB, cfg.X_UB]
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    ax.set_aspect("auto", "box")
    ax.set_title(subfigname, fontsize=10)


def draw_and_save(
    figname, data_exact, data_learned, boundary, griddata_points, griddata_xi, save_path
):
    fig = plt.figure(figname, figsize=(10, 6))
    gs = gridspec.GridSpec(1, 2)
    gs.update(top=0.8, bottom=0.2, left=0.1, right=0.9, wspace=0.5)

    # Exact p(t,x,y)
    plot_data_label = griddata(
        griddata_points, data_exact.flatten(), griddata_xi, method="cubic"
    )
    draw_subplot("Exact Dynamics", plot_data_label, fig, gs, "jet", boundary, loc=0)
    # Predicted p(t,x,y)
    plot_data_pred = griddata(
        griddata_points, data_learned.flatten(), griddata_xi, method="cubic"
    )
    draw_subplot("Learned Dynamics", plot_data_pred, fig, gs, "jet", boundary, loc=1)

    plt.savefig(osp.join(save_path, figname))
    plt.close()


def draw_and_save_ns(figname, data_exact, data_learned, grid_data, save_path):
    snap = 120
    nn = 200
    lb_x, lb_y = grid_data[:, 0].min(), grid_data[:, 1].min()
    ub_x, ub_y = grid_data[:, 0].max(), grid_data[:, 1].max()
    x_plot = np.linspace(lb_x, ub_x, nn)
    y_plot = np.linspace(lb_y, ub_y, nn)
    X_plot, Y_plot = np.meshgrid(x_plot, y_plot)

    fig = plt.figure(figname, figsize=(10, 6))
    gs = gridspec.GridSpec(1, 2)
    gs.update(top=0.8, bottom=0.2, left=0.1, right=0.9, wspace=0.5)
    # Exact p(t,x,y)
    plot_data_label = griddata(
        grid_data,
        data_exact[:, snap].flatten(),
        (X_plot, Y_plot),
        method="cubic",
    )
    draw_subplot(
        "Exact Dynamics",
        plot_data_label,
        fig,
        gs,
        "seismic",
        [lb_x, lb_y, ub_x, ub_y],
        loc=0,
    )
    # Predicted p(t,x,y)
    plot_data_pred = griddata(
        grid_data,
        data_learned[:, snap].flatten(),
        (X_plot, Y_plot),
        method="cubic",
    )
    draw_subplot(
        "Learned Dynamics",
        plot_data_pred,
        fig,
        gs,
        "seismic",
        [lb_x, lb_y, ub_x, ub_y],
        loc=1,
    )
    plt.savefig(osp.join(save_path, figname))
    plt.close()
