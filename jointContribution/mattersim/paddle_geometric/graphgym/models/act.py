import paddle

from paddle_geometric.graphgym.config import cfg
from paddle_geometric.graphgym.register import register_act


def relu():
    return paddle.nn.ReLU() if not cfg.mem.inplace else paddle.nn.functional.relu


def selu():
    return paddle.nn.SELU() if not cfg.mem.inplace else paddle.nn.functional.selu


def prelu():
    return paddle.nn.PReLU()


def elu():
    return paddle.nn.ELU() if not cfg.mem.inplace else paddle.nn.functional.elu


def lrelu_01():
    return paddle.nn.LeakyReLU(0.1) if not cfg.mem.inplace else lambda x: paddle.nn.functional.leaky_relu(x, negative_slope=0.1)


def lrelu_025():
    return paddle.nn.LeakyReLU(0.25) if not cfg.mem.inplace else lambda x: paddle.nn.functional.leaky_relu(x, negative_slope=0.25)


def lrelu_05():
    return paddle.nn.LeakyReLU(0.5) if not cfg.mem.inplace else lambda x: paddle.nn.functional.leaky_relu(x, negative_slope=0.5)


if cfg is not None:
    register_act('relu', relu)
    register_act('selu', selu)
    register_act('prelu', prelu)
    register_act('elu', elu)
    register_act('lrelu_01', lrelu_01)
    register_act('lrelu_025', lrelu_025)
    register_act('lrelu_05', lrelu_05)
