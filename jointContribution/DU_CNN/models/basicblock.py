from collections import OrderedDict

import paddle.nn as nn

"""
# --------------------------------------------
# Advanced nn.Sequential
# https://github.com/xinntao/BasicSR
# --------------------------------------------
"""


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Layer

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Layer):
            modules.append(module)
    return nn.Sequential(*modules)


# --------------------------------------------
# return nn.Sequential of (Conv + BN + ReLU)
# --------------------------------------------
def conv1(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="CBR",
    negative_slope=0.2,
):
    bias_attr = bias
    if bias_attr:
        bias_attr = None
    L = []
    for t in mode:
        if t == "C":
            L.append(
                nn.Conv1D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias_attr=bias_attr,
                )
            )
        elif t == "T":
            L.append(
                nn.Conv1DTranspose(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias_attr=bias_attr,
                )
            )
        elif t == "B":
            L.append(
                nn.BatchNorm1D(
                    out_channels,
                    momentum=0.9,
                    epsilon=1e-04,
                    weight_attr=None,
                    bias_attr=None,
                )
            )
        elif t == "I":
            L.append(nn.InstanceNorm1D(out_channels, weight_attr=None, bias_attr=None))
        elif t == "R":
            L.append(nn.ReLU())
        elif t == "r":
            L.append(nn.ReLU())
        elif t == "L":
            L.append(nn.LeakyReLU(negative_slope=negative_slope))
        elif t == "l":
            L.append(nn.LeakyReLU(negative_slope=negative_slope))
        elif t == "M":
            L.append(nn.MaxPool1D(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == "A":
            L.append(nn.AvgPool1D(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError(f"Undefined type: {t}")
    return sequential(*L)
