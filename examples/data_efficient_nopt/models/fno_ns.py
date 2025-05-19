import paddle.nn as nn

from .basics import SpectralConv3d
from .utils import _get_act
from .utils import add_padding
from .utils import remove_padding


class FNO3d_Backbone(nn.Layer):
    def __init__(
        self,
        modes1,
        modes2,
        modes3,
        width=16,
        layers=None,
        in_dim=4,
        act="gelu",
        pad_ratio=[0.0, 0.0],
    ):
        """
        Args:
            modes1: list of int, first dimension maximal modes for each layer
            modes2: list of int, second dimension maximal modes for each layer
            modes3: list of int, third dimension maximal modes for each layer
            layers: list of int, channels for each layer
            in_dim: int, input dimension
            act: {tanh, gelu, relu, leaky_relu}, activation function
            pad_ratio: the ratio of the extended domain
        """
        super(FNO3d_Backbone, self).__init__()

        if isinstance(pad_ratio, float):
            pad_ratio = [pad_ratio, pad_ratio]
        else:
            assert len(pad_ratio) == 2, "Cannot add padding in more than 2 directions."

        self.pad_ratio = pad_ratio
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.pad_ratio = pad_ratio

        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.LayerList(
            [
                SpectralConv3d(in_size, out_size, mode1_num, mode2_num, mode3_num)
                for in_size, out_size, mode1_num, mode2_num, mode3_num in zip(
                    self.layers, self.layers[1:], self.modes1, self.modes2, self.modes3
                )
            ]
        )

        self.ws = nn.LayerList(
            [
                nn.Conv1D(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        self.act = _get_act(act)

    def forward(self, x):
        """
        Args:
            x: (batchsize, x_grid, y_grid, t_grid, 3)

        Returns:
            feature: (batchsize, layers[-1], x_grid, y_grid, t_grid)

        """
        size_z = x.shape[-2]
        if max(self.pad_ratio) > 0:
            num_pad = [round(size_z * i) for i in self.pad_ratio]
        else:
            num_pad = [0.0, 0.0]
        length = len(self.ws)
        batchsize = x.shape[0]

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = add_padding(x, num_pad=num_pad)
        size_x, size_y, size_z = x.shape[-3], x.shape[-2], x.shape[-1]

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(
                batchsize, self.layers[i + 1], size_x, size_y, size_z
            )
            x = x1 + x2
            if i != length - 1:
                x = self.act(x)
        x = remove_padding(x, num_pad=num_pad)
        return x


class FNO3d(nn.Layer):
    def __init__(
        self,
        modes1,
        modes2,
        modes3,
        width=16,
        fc_dim=128,
        layers=None,
        in_dim=4,
        out_dim=1,
        act="gelu",
        pad_ratio=[0.0, 0.0],
        num_demos=0,
    ):
        """
        Args:
            modes1: list of int, first dimension maximal modes for each layer
            modes2: list of int, second dimension maximal modes for each layer
            modes3: list of int, third dimension maximal modes for each layer
            layers: list of int, channels for each layer
            fc_dim: dimension of fully connected layers
            in_dim: int, input dimension
            out_dim: int, output dimension
            act: {tanh, gelu, relu, leaky_relu}, activation function
            pad_ratio: the ratio of the extended domain
        """
        super(FNO3d, self).__init__()

        if isinstance(pad_ratio, float):
            pad_ratio = [pad_ratio, pad_ratio]
        else:
            assert len(pad_ratio) == 2, "Cannot add padding in more than 2 directions."

        self.pad_ratio = pad_ratio
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.pad_ratio = pad_ratio

        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers

        self.backbone = FNO3d_Backbone(
            modes1=modes1,
            modes2=modes2,
            modes3=modes3,
            layers=layers,
            in_dim=in_dim,
            act=act,
            pad_ratio=pad_ratio,
        )

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        self.act = _get_act(act)

        self.num_demos = num_demos
        # if self.num_demos and self.num_demos > 0:
        #     self.lossgen = LossGenerator(dx=2.0*math.pi/512., kernel_size=3) # TODO:

    def forward(self, x):
        """
        Args:
            x: (batchsize, x_grid, y_grid, t_grid, 3)

        Returns:
            u: (batchsize, x_grid, y_grid, t_grid, 1)

        """
        x = self.backbone(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def forward_icl(self, x, demo_xs, demo_ys, use_tqdm=False):
        raise NotImplementedError("not implemented yet")


class FNO3d_MAE(nn.Layer):
    def __init__(
        self,
        modes1,
        modes2,
        modes3,
        width=16,
        fc_dim=128,
        layers=None,
        in_dim=4,
        out_dim=1,
        act="gelu",
        pad_ratio=[0.0, 0.0],
    ):
        """
        Args:
            modes1: list of int, first dimension maximal modes for each layer
            modes2: list of int, second dimension maximal modes for each layer
            modes3: list of int, third dimension maximal modes for each layer
            layers: list of int, channels for each layer
            fc_dim: dimension of fully connected layers
            in_dim: int, input dimension
            out_dim: int, output dimension
            act: {tanh, gelu, relu, leaky_relu}, activation function
            pad_ratio: the ratio of the extended domain
        """
        super(FNO3d_MAE, self).__init__()
        if isinstance(pad_ratio, float):
            pad_ratio = [pad_ratio, pad_ratio]
        else:
            assert len(pad_ratio) == 2, "Cannot add padding in more than 2 directions."

        self.pad_ratio = pad_ratio
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.pad_ratio = pad_ratio

        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers

        self.encoder = FNO3d_Backbone(
            modes1=modes1,
            modes2=modes2,
            modes3=modes3,
            layers=layers,
            in_dim=in_dim,
            act=act,
            pad_ratio=pad_ratio,
        )
        self.encoder_to_decoder = nn.Linear(layers[-1], layers[-1])
        self.decoder = FNO3d_Backbone(
            modes1=modes1,
            modes2=modes2,
            modes3=modes3,
            layers=layers[:-1] + [in_dim],
            in_dim=layers[0],
            act=act,
            pad_ratio=pad_ratio,
        )

    def forward(self, x, mask):
        """
        x: (b, h, w, t, 4)
        """
        # B, C, H, W = x.shape
        x_enc = self.encoder(x * mask)
        x_enc = self.encoder_to_decoder(x_enc.permute(0, 2, 3, 4, 1))
        x_dec = self.decoder(x_enc).permute(0, 2, 3, 4, 1)
        return x_dec
