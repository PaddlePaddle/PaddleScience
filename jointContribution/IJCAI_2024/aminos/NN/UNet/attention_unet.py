from typing import List

import paddle

from .BuildingBlocks import Decoder
from .BuildingBlocks import DoubleConv
from .BuildingBlocks import Encoder


def create_feature_maps(init_channel_number, number_of_fmaps):
    return [(init_channel_number * 2**k) for k in range(number_of_fmaps)]


class UNet3D(paddle.nn.Layer):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(
        self,
        in_channels,
        nlayers=4,
        f_maps=16,
        layer_order="crg",
        num_groups=8,
        **kwargs
    ):
        super(UNet3D, self).__init__()
        if isinstance(f_maps, int):
            f_maps = create_feature_maps(f_maps, number_of_fmaps=nlayers)
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(
                    in_channels,
                    out_feature_num,
                    apply_pooling=False,
                    basic_module=DoubleConv,
                    conv_layer_order=layer_order,
                    num_groups=num_groups,
                )
            else:
                encoder = Encoder(
                    f_maps[i - 1],
                    out_feature_num,
                    basic_module=DoubleConv,
                    conv_layer_order=layer_order,
                    num_groups=num_groups,
                )
            encoders.append(encoder)
        self.encoders = paddle.nn.LayerList(sublayers=encoders)
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(
                in_feature_num,
                out_feature_num,
                basic_module=DoubleConv,
                conv_layer_order=layer_order,
                num_groups=num_groups,
            )
            decoders.append(decoder)
        self.decoders = paddle.nn.LayerList(sublayers=decoders)

    def forward(self, x):
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)
        encoders_features = encoders_features[1:]
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)
        return x


class UNet3DWithSamplePoints(paddle.nn.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_levels: int,
        use_position_input: bool = True,
    ):
        super().__init__()
        self.unet3d = UNet3D(
            in_channels=in_channels, nlayers=num_levels, f_maps=hidden_channels
        )
        self.out_ln = paddle.nn.LayerNorm(normalized_shape=hidden_channels)

    def voxel_expand(self, voxel_left):
        indices = paddle.arange(start=31, end=-1, step=-1, dtype="int32")
        voxel_right = voxel_left[:, :, :, indices, :]
        voxel_merge = paddle.concat((voxel_left, voxel_right), axis=3)
        return voxel_merge

    def forward(self, x, output_points, half=False):
        x = self.unet3d.forward(x)
        if half:
            x = self.voxel_expand(x)
        if isinstance(output_points, List):
            rt_x = []
            for idx, output_point in enumerate(output_points):
                cur_x = (
                    paddle.nn.functional.grid_sample(
                        x=x[idx : idx + 1], grid=output_point, align_corners=False
                    )
                    .squeeze()
                    .T
                )
                rt_x.append(cur_x)
            rt_x = paddle.concat(x=rt_x, axis=0)
        else:
            rt_x = (
                paddle.nn.functional.grid_sample(
                    x=x, grid=output_points, align_corners=False
                )
                .squeeze()
                .T
            )
        return self.out_ln(rt_x)


if __name__ == "__main__":
    net = UNet3DWithSamplePoints(1, 64, 64, 4, True)
    sdf = paddle.randn(shape=[7, 1, 64, 64, 64])
    outputpoints = paddle.randn(shape=[7, 1471, 1, 1, 3])
    output = net.forward(sdf, outputpoints, half=False)
    print(tuple(output.shape))
