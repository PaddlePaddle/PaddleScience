import time
import numpy as np
import sympy as sp
import scipy
from scipy import interpolate
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# functions to plot a variable
def plot_field(
    var,
    save_name,
    coordinates=["x", "y"],
    bounds_var=None,
    criteria=None,
    plot_title="",
    resolution=128,
    figsize=(8, 8),
):

    plt.figure(figsize=figsize)
    nr_plots = len(list(var.keys())) - 2  # not plotting x or y
    plot_index = 1
    for key, value in var.items():
        if key in coordinates:
            continue

        X, Y = _make_mesh(var, coordinates, bounds_var, resolution)
        pos = np.concatenate([var[coordinates[0]], var[coordinates[1]]], axis=1)
        value_star = interpolate.griddata(pos, value.flatten(), (X, Y), method="linear")

        if criteria is not None:
            np_criteria = _compile_criteria(coordinates, criteria)
            nan_mask = np.where(np_criteria(X, Y), 0.0, np.nan)
            value_star += nan_mask

        plt.subplot(nr_plots, 1, plot_index)
        plt.title(plot_title + ": " + key)
        plt.imshow(
            np.flip(value_star, axis=0),
            cmap="jet",
            extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)],
        )
        plt.xlabel(coordinates[0])
        plt.ylabel(coordinates[1])

        plt.colorbar()
        plot_index += 1

    plt.savefig(save_name + ".png")
    plt.close()


# functions to plot true and pred variables with diff plot
def plot_field_compare(
    true_var,
    pred_var,
    save_name,
    coordinates=["x", "y"],
    bounds_var=None,
    criteria=None,
    resolution=128,
    figsize=(12, 10),
    same_colorbar=False,
):

    plt.figure(figsize=figsize)
    nr_plots = len(list(true_var.keys())) - 2  # not plotting x or y
    plot_index = 1
    for key in true_var.keys():
        if key in coordinates:
            continue
        X, Y = _make_mesh(true_var, coordinates, bounds_var, resolution)
        pos = np.concatenate(
            [true_var[coordinates[0]], true_var[coordinates[1]]], axis=1
        )
        true_field_star = interpolate.griddata(
            pos, true_var[key].flatten(), (X, Y), method="linear"
        )
        pred_field_star = interpolate.griddata(
            pos, pred_var[key].flatten(), (X, Y), method="linear"
        )

        if criteria is not None:
            np_criteria = _compile_criteria(coordinates, criteria)
            nan_mask = np.where(np_criteria(X, Y), 0.0, np.nan)
            true_field_star += nan_mask
            pred_field_star += nan_mask

        if same_colorbar:
            vmax = np.max(true_var[key])
            vmin = np.min(true_var[key])
        else:
            vmax = None
            vmin = None

        # pred plot
        plt.subplot(nr_plots, 3, plot_index)
        plt.title("Predicted: " + key)
        plt.imshow(pred_field_star, cmap="jet", vmax=vmax, vmin=vmin)
        plt.colorbar()
        plot_index += 1

        # true plot
        plt.subplot(nr_plots, 3, plot_index)
        plt.title("True: " + key)
        plt.imshow(true_field_star, cmap="jet", vmax=vmax, vmin=vmin)
        plt.colorbar()
        plot_index += 1

        # diff plot
        plt.subplot(nr_plots, 3, plot_index)
        plt.title("Difference: " + key)
        plt.imshow((true_field_star - pred_field_star), cmap="jet")
        plt.colorbar()
        plot_index += 1

    plt.savefig(save_name + ".png")
    plt.close()


def _make_mesh(var, coordinates, bounds_var, resolution):
    if bounds_var is not None:
        x_min = bounds_var[coordinates[0] + "_min"]
        x_max = bounds_var[coordinates[0] + "_max"]
        y_min = bounds_var[coordinates[1] + "_min"]
        y_max = bounds_var[coordinates[1] + "_max"]
    else:
        x_min = np.min(var[coordinates[0]])
        x_max = np.max(var[coordinates[0]])
        y_min = np.min(var[coordinates[1]])
        y_max = np.max(var[coordinates[1]])
    x_len = x_max - x_min
    y_len = y_max - y_min
    len_min = max(x_len, y_len)

    nn_x = int((x_len / len_min) * resolution)
    nn_y = int((y_len / len_min) * resolution)
    if bounds_var is not None:
        x = np.linspace(
            bounds_var[coordinates[0] + "_min"],
            bounds_var[coordinates[0] + "_max"],
            nn_x,
        )
        y = np.linspace(
            bounds_var[coordinates[1] + "_min"],
            bounds_var[coordinates[1] + "_max"],
            nn_y,
        )
    else:
        x = np.linspace(np.min(var[coordinates[0]]), np.max(var[coordinates[0]]), nn_x)
        y = np.linspace(np.min(var[coordinates[1]]), np.max(var[coordinates[1]]), nn_y)
    return np.meshgrid(x, y)


def _compile_criteria(coordinates, criteria):
    np_criteria = sp.lambdify([sp.Symbol(k) for k in coordinates], criteria, "numpy")
    return np_criteria


# functions to plot a variable
def _var_to_mesh(
    var, key, coordinates=["x", "y"], bounds_var=None, criteria=None, resolution=128
):

    X, Y = _make_mesh(var, coordinates, bounds_var, resolution)
    pos = np.concatenate([var[coordinates[0]], var[coordinates[1]]], axis=1)
    value_star = interpolate.griddata(pos, var[key].flatten(), (X, Y), method="linear")

    if criteria is not None:
        np_criteria = _compile_criteria(coordinates, criteria)
        nan_mask = np.where(np_criteria(X, Y), 0.0, np.nan)
        value_star += nan_mask
        # value_star = value_star * nan_mask
    return value_star
