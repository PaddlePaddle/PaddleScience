import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import paddle
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


def dfdx(
    f: paddle.Tensor,
    dydeta: paddle.Tensor,
    dydxi: paddle.Tensor,
    jinv: paddle.Tensor,
    h: int = 0.01,
):
    """Calculate the derivative of the given function f in the x direction
    using the form of discrete difference.

    Args:
        f (paddle.Tensor): The matrix that needs to calculate differentiation.
        dydeta (paddle.Tensor): The dydeta data.
        dydxi (paddle.Tensor): The dydxi data.
        jinv (paddle.Tensor): Jacobian matrix.
        h (int, optional): Differential interval. Defaults to 0.01.
    """
    dfdxi_internal = (
        (
            -f[:, :, :, 4:]
            + 8 * f[:, :, :, 3:-1]
            - 8 * f[:, :, :, 1:-3]
            + f[:, :, :, 0:-4]
        )
        / 12
        / h
    )
    dfdxi_left = (
        (
            -11 * f[:, :, :, 0:-3]
            + 18 * f[:, :, :, 1:-2]
            - 9 * f[:, :, :, 2:-1]
            + 2 * f[:, :, :, 3:]
        )
        / 6
        / h
    )
    dfdxi_right = (
        (
            11 * f[:, :, :, 3:]
            - 18 * f[:, :, :, 2:-1]
            + 9 * f[:, :, :, 1:-2]
            - 2 * f[:, :, :, 0:-3]
        )
        / 6
        / h
    )
    dfdxi = paddle.concat(
        (dfdxi_left[:, :, :, 0:2], dfdxi_internal, dfdxi_right[:, :, :, -2:]), 3
    )
    dfdeta_internal = (
        (
            -f[:, :, 4:, :]
            + 8 * f[:, :, 3:-1, :]
            - 8 * f[:, :, 1:-3, :]
            + f[:, :, 0:-4, :]
        )
        / 12
        / h
    )
    dfdeta_low = (
        (
            -11 * f[:, :, 0:-3, :]
            + 18 * f[:, :, 1:-2, :]
            - 9 * f[:, :, 2:-1, :]
            + 2 * f[:, :, 3:, :]
        )
        / 6
        / h
    )
    dfdeta_up = (
        (
            11 * f[:, :, 3:, :]
            - 18 * f[:, :, 2:-1, :]
            + 9 * f[:, :, 1:-2, :]
            - 2 * f[:, :, 0:-3, :]
        )
        / 6
        / h
    )
    dfdeta = paddle.concat(
        (dfdeta_low[:, :, 0:2, :], dfdeta_internal, dfdeta_up[:, :, -2:, :]), 2
    )
    dfdx = jinv * (dfdxi * dydeta - dfdeta * dydxi)
    return dfdx


def dfdy(
    f: paddle.Tensor,
    dxdxi: paddle.Tensor,
    dxdeta: paddle.Tensor,
    jinv: paddle.Tensor,
    h: int = 0.01,
):
    """Calculate the derivative of the given function f in the y direction
    using the form of discrete difference.

    Args:
        f (paddle.Tensor): The matrix that needs to calculate differentiation.
        dxdxi (paddle.Tensor): The dxdxi data.
        dxdeta (paddle.Tensor): The dxdeta data.
        jinv (paddle.Tensor): Jacobian matrix.
        h (int, optional): Differential interval. Defaults to 0.01.
    """
    dfdxi_internal = (
        (
            -f[:, :, :, 4:]
            + 8 * f[:, :, :, 3:-1]
            - 8 * f[:, :, :, 1:-3]
            + f[:, :, :, 0:-4]
        )
        / 12
        / h
    )
    dfdxi_left = (
        (
            -11 * f[:, :, :, 0:-3]
            + 18 * f[:, :, :, 1:-2]
            - 9 * f[:, :, :, 2:-1]
            + 2 * f[:, :, :, 3:]
        )
        / 6
        / h
    )
    dfdxi_right = (
        (
            11 * f[:, :, :, 3:]
            - 18 * f[:, :, :, 2:-1]
            + 9 * f[:, :, :, 1:-2]
            - 2 * f[:, :, :, 0:-3]
        )
        / 6
        / h
    )
    dfdxi = paddle.concat(
        (dfdxi_left[:, :, :, 0:2], dfdxi_internal, dfdxi_right[:, :, :, -2:]), 3
    )

    dfdeta_internal = (
        (
            -f[:, :, 4:, :]
            + 8 * f[:, :, 3:-1, :]
            - 8 * f[:, :, 1:-3, :]
            + f[:, :, 0:-4, :]
        )
        / 12
        / h
    )
    dfdeta_low = (
        (
            -11 * f[:, :, 0:-3, :]
            + 18 * f[:, :, 1:-2, :]
            - 9 * f[:, :, 2:-1, :]
            + 2 * f[:, :, 3:, :]
        )
        / 6
        / h
    )
    dfdeta_up = (
        (
            11 * f[:, :, 3:, :]
            - 18 * f[:, :, 2:-1, :]
            + 9 * f[:, :, 1:-2, :]
            - 2 * f[:, :, 0:-3, :]
        )
        / 6
        / h
    )
    dfdeta = paddle.concat(
        (dfdeta_low[:, :, 0:2, :], dfdeta_internal, dfdeta_up[:, :, -2:, :]), 2
    )
    dfdy = jinv * (dfdeta * dxdxi - dfdxi * dxdeta)
    return dfdy


def set_axis_label(ax, type):
    if type == "p":
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
    elif type == "r":
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
    else:
        raise ValueError("The axis type only can be reference or physical")


def gen_e2vcg(x: np.ndarray):
    """Generate adjacent coordinate indices for each point based on the shape of x.

    Args:
        x (np.ndarray): Input coordinate array.
    """
    nelemx = x.shape[1] - 1
    nelemy = x.shape[0] - 1
    nelem = nelemx * nelemy
    nnx = x.shape[1]
    e2vcg = np.zeros([4, nelem])
    for j in range(nelemy):
        for i in range(nelemx):
            e2vcg[:, j * nelemx + i] = np.asarray(
                [j * nnx + i, j * nnx + i + 1, (j + 1) * nnx + i, (j + 1) * nnx + i + 1]
            )
    return e2vcg.astype("int64")


def visualize(ax, x, y, u, colorbarPosition="vertical", colorlimit=None):
    xdg0 = np.vstack([x.flatten(order="C"), y.flatten(order="C")])
    udg0 = u.flatten(order="C")
    idx = np.asarray([0, 1, 3, 2])
    nelemx = x.shape[1] - 1
    nelemy = x.shape[0] - 1
    nelem = nelemx * nelemy
    e2vcg0 = gen_e2vcg(x)
    udg_ref = udg0[e2vcg0]
    cmap = matplotlib.cm.coolwarm
    polygon_list = []
    for i in range(nelem):
        polygon_ = Polygon(xdg0[:, e2vcg0[idx, i]].T)
        polygon_list.append(polygon_)
    polygon_ensemble = PatchCollection(polygon_list, cmap=cmap, alpha=1)
    polygon_ensemble.set_edgecolor("face")
    polygon_ensemble.set_array(np.mean(udg_ref, axis=0))
    if colorlimit is None:
        pass
    else:
        polygon_ensemble.set_clim(colorlimit)
    ax.add_collection(polygon_ensemble)
    ax.set_xlim(np.min(xdg0[0, :]), np.max(xdg0[0, :]))
    ax.set_ylim(np.min(xdg0[1, :]), np.max(xdg0[1, :]))
    cbar = plt.colorbar(polygon_ensemble, orientation=colorbarPosition)
    return ax, cbar
