# import math
# import random
# from pdb import set_trace as bp

# import numpy as np
# import paddle
import paddle.nn as nn

# from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from .basics import SpectralConv3d
from .utils import _get_act
from .utils import add_padding
from .utils import remove_padding

# from .eval import LossGenerator
# from tqdm import tqdm


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
        # """
        # x: B, H, W, T, C (4)
        # demo_xs: J, H, W, T, C
        # demo_ys: J, H, W
        # """
        # C_out = 1
        # B, H, W, T, C = x.shape

        # repeat = 10
        # p = 0.2
        # sigma_range = [0, 0]
        # x_aug = []
        # demo_xs_aug = []
        # for _ in range(repeat):
        #     if sum(sigma_range) > 0:
        #         sigma = random.uniform(*sigma_range)
        #         # https://github.com/scipy/scipy/blob/v1.11.4/scipy/ndimage/_filters.py#L232
        #         _kernel = min(
        #             int((sigma * 4 + 1) / 2) * 2 + 1, (x.shape[1] // 2) * 2 - 1
        #         )
        #     mask = paddle.nn.functional.dropout(paddle.ones([T, C, H, W]), p=p)
        #     ######
        #     _x_aug = []
        #     for _x in x.permute(0, 3, 4, 1, 2):
        #         if sum(sigma_range) > 0:
        #             # [TODO] [https://github.com/PaddlePaddle/Paddle/issues/26568]
        #             raise NotImplementedError(">>>")
        #             # __x = torchvision.transforms.functional.gaussian_blur(_x.clone(), kernel_size=[_kernel, _kernel], sigma=sigma)
        #         else:
        #             __x = _x.clone()
        #         __x = __x * mask
        #         _x_aug.append(__x)
        #     _x_aug = paddle.stack(_x_aug, axis=0).permute([0, 3, 4, 1, 2])
        #     x_aug.append(_x_aug)
        #     ######
        #     _demo_xs_aug = []
        #     for _x in demo_xs.permute(0, 3, 4, 1, 2):
        #         if sum(sigma_range) > 0:
        #             # [TODO] [https://github.com/PaddlePaddle/Paddle/issues/26568]
        #             raise NotImplementedError(">>>")
        #             # __x = torchvision.transforms.functional.gaussian_blur(_x.clone(), kernel_size=[_kernel, _kernel], sigma=sigma)
        #         else:
        #             __x = _x.clone()
        #         __x = __x * mask
        #         _demo_xs_aug.append(__x)
        #     _demo_xs_aug = paddle.stack(_demo_xs_aug, axis=0).permute(0, 3, 4, 1, 2)
        #     demo_xs_aug.append(_demo_xs_aug)
        # x_aug = paddle.stack(x_aug, axis=0)
        # demo_xs_aug = paddle.stack(demo_xs_aug, axis=0)

        # J = demo_xs.shape[0]
        # # pred = self.forward(x).contiguous() # B, H, W, T, 1
        # pred0 = self.forward(x)  # B, H, W, T, 1
        # pred = paddle.concat(
        #     [self.forward(_x) for _x in x_aug], axis=-1
        # )  # B, H, W, T, 1
        # C = pred.shape[-1]
        # # # div = self.lossgen.get_div_loss(pred).contiguous()
        # demo_pred = []
        # idx = 0
        # for _demo_xs_aug in demo_xs_aug:
        #     idx = 0
        #     _demo_pred = []
        #     while idx < _demo_xs_aug.shape[0]:
        #         _x = _demo_xs_aug[idx : idx + B]
        #         _pred = self.forward(_x)
        #         _demo_pred.append(_pred)
        #         idx += _x.shape[0]
        #     demo_pred.append(paddle.concat(_demo_pred, axis=0))
        # demo_pred = paddle.concat(demo_pred, axis=-1)

        # demo_pred_flat = demo_pred.view([1, -1, C])
        # y_nn = paddle.zeros([B, H, W, T, C_out])
        # stds_nn = paddle.zeros([B, H, W, T, C_out])
        # batch_b = 1
        # _b = 0
        # batch_h = 8
        # _h = 0
        # batch_w = 8
        # _w = 0
        # batch_t = 1
        # _t = 0
        # topk = int(20 * (J**0.5))  # TODO:
        # if use_tqdm:
        #     pbar = tqdm(
        #         total=np.ceil(B / batch_b)
        #         * np.ceil(H / batch_h)
        #         * np.ceil(W / batch_w)
        #         * np.ceil(T / batch_t)
        #     )
        # else:
        #     pbar = None
        # while _b < B:
        #     _h = 0
        #     while _h < H:
        #         _w = 0
        #         while _w < W:
        #             _t = 0
        #             while _t < T:
        #                 if pbar is not None:
        #                     pbar.set_description(
        #                         "_b %d, _h %d, _w %d, _t %d" % (_b, _h, _w, _t)
        #                     )
        #                     pbar.update(1)
        #                 pred_flat = pred[
        #                     _b : _b + batch_b,
        #                     _h : _h + batch_h,
        #                     _w : _w + batch_w,
        #                     _t : _t + batch_t,
        #                     :,
        #                 ]
        #                 __b, __h, __w, __t, _ = pred_flat.shape
        #                 pred_flat = pred_flat.contiguous().view(-1, 1, C)

        #                 # gap = (pred_flat - demo_pred_flat).pow(2) / pred_flat.pow(2)
        #                 # gap = paddle.linalg.norm(
        #                 #     (pred_flat - demo_pred_flat).pow(2) / pred_flat.pow(2),
        #                 #     axis=-1,
        #                 # )
        #                 raise NotImplementedError("not implemented yet")
        #                 # cos = 1 - pairwise_cosine_similarity(
        #                 #     pred_flat.squeeze(1), demo_pred_flat.squeeze(0)
        #                 # )
        #                 # gap = gap * cos
        #                 # gap_re = gap.view([__b, __h, __w, __t, -1])
        #                 # # gap_nn[_b:_b+batch_b, _h:_h+batch_h, _w:_w+batch_w, _t:_t+batch_t] = torch.mean(torch.sort(gap_re, -1)[0][:, :, :, :, :topk], -1)
        #                 # index = paddle.argsort(paddle.abs(gap_re), -1)[
        #                 #     :, :, :, :, :topk
        #                 # ]  # TODO: spatial index of ascending sort by pred gap
        #                 # _y_nn = paddle.stack(
        #                 #     [
        #                 #         paddle.take_along_axis(
        #                 #             demo_ys.view([-1, C_out]),
        #                 #             index[:, :, :, :, _k].view([-1, 1]),
        #                 #             axis=0,
        #                 #         ).view([__b, __h, __w, __t, C_out])
        #                 #         for _k in range(topk)
        #                 #     ],
        #                 #     -1,
        #                 # )
        #                 # # spatial_dims = (2, 3)
        #                 # # y_nn[_b:_b+batch_b, _h:_h+batch_h, _w:_w+batch_w, _t:_t+batch_t, :] = _y_nn
        #                 # y_nn[
        #                 #     _b : _b + batch_b,
        #                 #     _h : _h + batch_h,
        #                 #     _w : _w + batch_w,
        #                 #     _t : _t + batch_t,
        #                 #     :,
        #                 # ] = _y_nn.mean(-1)
        #                 # stds_nn[
        #                 #     _b : _b + batch_b,
        #                 #     _h : _h + batch_h,
        #                 #     _w : _w + batch_w,
        #                 #     _t : _t + batch_t,
        #                 #     :,
        #                 # ] = paddle.abs(_y_nn.std(-1) / _y_nn.mean(-1))
        #                 # # np.set_printoptions(precision=5)
        #                 # # print(_h, _w, sum(pred[_b, :2, _h+1, _w+1]-y_nn[_b, :2, _h+1, _w+1]).item(), np.round(pred[_b, :, _h+1, _w+1].detach().cpu().numpy(), 5).tolist(), np.round(y_nn[_b, :, _h+1, _w+1].detach().cpu().numpy(), 5).tolist())

        #                 # _t += batch_t
        #             _w += batch_w
        #         _h += batch_h
        #     _b += batch_b
        # # bp()
        # # print(y_nn.mean(), demo_ys.mean())
        # # return y_nn
        # mask = (stds_nn < stds_nn.mean()).astype(paddle.float32)  # TODO:
        # return mask * y_nn + (1 - mask) * pred0


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
