"""
U-TAE Implementation (Paddle Version)
Converted to PaddlePaddle
"""
import paddle
import paddle.nn as nn
from src.backbones.convlstm import BConvLSTM
from src.backbones.convlstm import ConvLSTM
from src.backbones.ltae import LTAE2d


class UTAE(nn.Layer):
    """
    U-TAE architecture for spatio-temporal encoding of satellite image time series.
    Args:
        input_dim (int): Number of channels in the input images.
        encoder_widths (List[int]): List giving the number of channels of the successive encoder_widths of the convolutional encoder.
        This argument also defines the number of encoder_widths (i.e. the number of downsampling steps +1)
        in the architecture.
        The number of channels are given from top to bottom, i.e. from the highest to the lowest resolution.
        decoder_widths (List[int], optional): Same as encoder_widths but for the decoder. The order in which the number of
        channels should be given is also from top to bottom. If this argument is not specified the decoder
        will have the same configuration as the encoder.
        out_conv (List[int]): Number of channels of the successive convolutions for the
        str_conv_k (int): Kernel size of the strided up and down convolutions.
        str_conv_s (int): Stride of the strided up and down convolutions.
        str_conv_p (int): Padding of the strided up and down convolutions.
        agg_mode (str): Aggregation mode for the skip connections. Can either be:
            - att_group (default) : Attention weighted temporal average, using the same
            channel grouping strategy as in the LTAE. The attention masks are bilinearly
            resampled to the resolution of the skipped feature maps.
            - att_mean : Attention weighted temporal average,
                using the average attention scores across heads for each date.
            - mean : Temporal average excluding padded dates.
        encoder_norm (str): Type of normalisation layer to use in the encoding branch. Can either be:
            - group : GroupNorm (default)
            - batch : BatchNorm
            - instance : InstanceNorm
        n_head (int): Number of heads in LTAE.
        d_model (int): Parameter of LTAE
        d_k (int): Key-Query space dimension
        encoder (bool): If true, the feature maps instead of the class scores are returned (default False)
        return_maps (bool): If true, the feature maps instead of the class scores are returned (default False)
        pad_value (float): Value used by the dataloader for temporal padding.
        padding_mode (str): Spatial padding strategy for convolutional layers (passed to nn.Conv2D).
    """

    def __init__(
        self,
        input_dim,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        out_conv=[32, 20],
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        agg_mode="att_group",
        encoder_norm="group",
        n_head=16,
        d_model=256,
        d_k=4,
        encoder=False,
        return_maps=False,
        pad_value=0,
        padding_mode="reflect",
    ):

        super(UTAE, self).__init__()
        self.n_stages = len(encoder_widths)
        self.return_maps = return_maps
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.enc_dim = (
            decoder_widths[0] if decoder_widths is not None else encoder_widths[0]
        )
        self.stack_dim = (
            sum(decoder_widths) if decoder_widths is not None else sum(encoder_widths)
        )
        self.pad_value = pad_value
        self.encoder = encoder
        if encoder:
            self.return_maps = True

        if decoder_widths is not None:
            assert len(encoder_widths) == len(decoder_widths)
            assert encoder_widths[-1] == decoder_widths[-1]
        else:
            decoder_widths = encoder_widths

        self.in_conv = ConvBlock(
            nkernels=[input_dim] + [encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        self.down_blocks = nn.LayerList(
            [
                DownConvBlock(
                    d_in=encoder_widths[i],
                    d_out=encoder_widths[i + 1],
                    k=str_conv_k,
                    s=str_conv_s,
                    p=str_conv_p,
                    pad_value=pad_value,
                    norm=encoder_norm,
                    padding_mode=padding_mode,
                )
                for i in range(self.n_stages - 1)
            ]
        )
        self.up_blocks = nn.LayerList(
            [
                UpConvBlock(
                    d_in=decoder_widths[i],
                    d_out=decoder_widths[i - 1],
                    d_skip=encoder_widths[i - 1],
                    k=str_conv_k,
                    s=str_conv_s,
                    p=str_conv_p,
                    norm="batch",
                    padding_mode=padding_mode,
                )
                for i in range(self.n_stages - 1, 0, -1)
            ]
        )
        self.temporal_encoder = LTAE2d(
            in_channels=encoder_widths[-1],
            d_model=d_model,
            n_head=n_head,
            mlp=[d_model, encoder_widths[-1]],
            return_att=True,
            d_k=d_k,
        )
        self.temporal_aggregator = Temporal_Aggregator(mode=agg_mode)
        self.out_conv = ConvBlock(
            nkernels=[decoder_widths[0]] + out_conv, padding_mode=padding_mode
        )

    def forward(self, input, batch_positions=None, return_att=False):
        # Create pad mask by comparing with pad_value
        # Use safe tensor comparison to avoid type issues
        pad_value_tensor = paddle.to_tensor(self.pad_value, dtype=input.dtype)
        comparison = paddle.equal(input, pad_value_tensor)

        # Sequentially reduce dimensions using all()
        mask_step1 = paddle.all(comparison, axis=-1)  # Reduce last dim
        mask_step2 = paddle.all(mask_step1, axis=-1)  # Reduce second-to-last dim
        pad_mask = paddle.all(mask_step2, axis=-1)  # Reduce third-to-last dim (BxT)
        out = self.in_conv.smart_forward(input)
        feature_maps = [out]
        # SPATIAL ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)
        # TEMPORAL ENCODER
        out, att = self.temporal_encoder(
            feature_maps[-1], batch_positions=batch_positions, pad_mask=pad_mask
        )
        # SPATIAL DECODER
        if self.return_maps:
            maps = [out]
        for i in range(self.n_stages - 1):
            skip = self.temporal_aggregator(
                feature_maps[-(i + 2)], pad_mask=pad_mask, attn_mask=att
            )
            out = self.up_blocks[i](out, skip)
            if self.return_maps:
                maps.append(out)

        if self.encoder:
            return out, maps
        else:
            out = self.out_conv(out)
            if return_att:
                return out, att
            if self.return_maps:
                return out, maps
            else:
                return out


class TemporallySharedBlock(nn.Layer):
    """
    Helper module for convolutional encoding blocks that are shared across a sequence.
    This module adds the self.smart_forward() method the the block.
    smart_forward will combine the batch and temporal dimension of an input tensor
    if it is 5-D and apply the shared convolutions to all the (batch x temp) positions.
    """

    def __init__(self, pad_value=None):
        super(TemporallySharedBlock, self).__init__()
        self.out_shape = None
        self.pad_value = pad_value

    def smart_forward(self, input):
        if len(input.shape) == 4:
            return self.forward(input)
        else:
            b, t, c, h, w = input.shape

            if self.pad_value is not None:
                dummy = paddle.zeros(input.shape).astype("float32")
                self.out_shape = self.forward(dummy.reshape([b * t, c, h, w])).shape

            out = input.reshape([b * t, c, h, w])
            if self.pad_value is not None:
                pad_value_tensor = paddle.to_tensor(self.pad_value, dtype=out.dtype)
                comparison = paddle.equal(out, pad_value_tensor)
                mask_step1 = paddle.all(comparison, axis=-1)  # Reduce last dim
                mask_step2 = paddle.all(
                    mask_step1, axis=-1
                )  # Reduce second-to-last dim
                pad_mask = paddle.all(mask_step2, axis=-1)  # Reduce third-to-last dim
                if pad_mask.any():
                    temp = paddle.ones(self.out_shape) * self.pad_value
                    temp[~pad_mask] = self.forward(out[~pad_mask])
                    out = temp
                else:
                    out = self.forward(out)
            else:
                out = self.forward(out)
            _, c, h, w = out.shape
            out = out.reshape([b, t, c, h, w])
            return out


class ConvLayer(nn.Layer):
    def __init__(
        self,
        nkernels,
        norm="batch",
        k=3,
        s=1,
        p=1,
        n_groups=4,
        last_relu=True,
        padding_mode="reflect",
    ):
        super(ConvLayer, self).__init__()
        layers = []
        if norm == "batch":
            nl = nn.BatchNorm2D
        elif norm == "instance":
            nl = nn.InstanceNorm2D
        elif norm == "group":
            nl = lambda num_feats: nn.GroupNorm(
                num_channels=num_feats,
                num_groups=n_groups,
            )
        else:
            nl = None
        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2D(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))

            if last_relu:
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv(input)


class ConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        nkernels,
        pad_value=None,
        norm="batch",
        last_relu=True,
        padding_mode="reflect",
    ):
        super(ConvBlock, self).__init__(pad_value=pad_value)
        self.conv = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        return self.conv(input)


class DownConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        d_in,
        d_out,
        k,
        s,
        p,
        pad_value=None,
        norm="batch",
        padding_mode="reflect",
    ):
        super(DownConvBlock, self).__init__(pad_value=pad_value)
        self.down = ConvLayer(
            nkernels=[d_in, d_in],
            norm=norm,
            k=k,
            s=s,
            p=p,
            padding_mode=padding_mode,
        )
        self.conv1 = ConvLayer(
            nkernels=[d_in, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        out = self.down(input)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class UpConvBlock(nn.Layer):
    def __init__(
        self, d_in, d_out, k, s, p, norm="batch", d_skip=None, padding_mode="reflect"
    ):
        super(UpConvBlock, self).__init__()
        d = d_out if d_skip is None else d_skip
        self.skip_conv = nn.Sequential(
            nn.Conv2D(in_channels=d, out_channels=d, kernel_size=1),
            nn.BatchNorm2D(d),
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.Conv2DTranspose(
                in_channels=d_in, out_channels=d_out, kernel_size=k, stride=s, padding=p
            ),
            nn.BatchNorm2D(d_out),
            nn.ReLU(),
        )
        self.conv1 = ConvLayer(
            nkernels=[d_out + d, d_out], norm=norm, padding_mode=padding_mode
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode
        )

    def forward(self, input, skip):
        out = self.up(input)
        out = paddle.concat([out, self.skip_conv(skip)], axis=1)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class Temporal_Aggregator(nn.Layer):
    def __init__(self, mode="mean"):
        super(Temporal_Aggregator, self).__init__()
        self.mode = mode

    def forward(self, x, pad_mask=None, attn_mask=None):
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.reshape([n_heads * b, t, h, w])

                if x.shape[-2] > w:
                    attn = nn.functional.interpolate(
                        attn, size=x.shape[-2:], mode="bilinear", align_corners=False
                    )
                else:
                    attn = nn.functional.avg_pool2d(attn, kernel_size=w // x.shape[-2])

                attn = attn.reshape([n_heads, b, t, *x.shape[-2:]])
                attn = attn * (~pad_mask).astype("float32")[None, :, :, None, None]

                # Split x into n_heads chunks along axis 2
                chunk_size = x.shape[2] // n_heads
                x_chunks = []
                for i in range(n_heads):
                    start_idx = i * chunk_size
                    end_idx = (i + 1) * chunk_size
                    x_chunks.append(x[:, :, start_idx:end_idx, :, :])
                out = paddle.stack(x_chunks)  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(axis=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = paddle.concat([group for group in out], axis=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(axis=0)  # average over heads -> BxTxHxW
                attn = nn.functional.interpolate(
                    attn, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
                attn = attn * (~pad_mask).astype("float32")[:, :, None, None]
                out = (x * attn[:, :, None, :, :]).sum(axis=1)
                return out
            elif self.mode == "mean":
                out = x * (~pad_mask).astype("float32")[:, :, None, None, None]
                out = out.sum(axis=1) / (~pad_mask).sum(axis=1)[:, None, None, None]
                return out
        else:
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.reshape([n_heads * b, t, h, w])
                if x.shape[-2] > w:
                    attn = nn.functional.interpolate(
                        attn, size=x.shape[-2:], mode="bilinear", align_corners=False
                    )
                else:
                    attn = nn.functional.avg_pool2d(attn, kernel_size=w // x.shape[-2])
                attn = attn.reshape([n_heads, b, t, *x.shape[-2:]])
                # Split x into n_heads chunks along axis 2
                chunk_size = x.shape[2] // n_heads
                x_chunks = []
                for i in range(n_heads):
                    start_idx = i * chunk_size
                    end_idx = (i + 1) * chunk_size
                    x_chunks.append(x[:, :, start_idx:end_idx, :, :])
                out = paddle.stack(x_chunks)  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(axis=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = paddle.concat([group for group in out], axis=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(axis=0)  # average over heads -> BxTxHxW
                attn = nn.functional.interpolate(
                    attn, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
                out = (x * attn[:, :, None, :, :]).sum(axis=1)
                return out
            elif self.mode == "mean":
                return x.mean(axis=1)


class RecUNet(nn.Layer):
    """Recurrent U-Net architecture. Similar to the U-TAE architecture but
    the L-TAE is replaced by a recurrent network
    and temporal averages are computed for the skip connections."""

    def __init__(
        self,
        input_dim,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        out_conv=[32, 20],
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        temporal="lstm",
        input_size=128,
        encoder_norm="group",
        hidden_dim=128,
        encoder=False,
        padding_mode="reflect",
        pad_value=0,
    ):
        super(RecUNet, self).__init__()
        self.n_stages = len(encoder_widths)
        self.temporal = temporal
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.enc_dim = (
            decoder_widths[0] if decoder_widths is not None else encoder_widths[0]
        )
        self.stack_dim = (
            sum(decoder_widths) if decoder_widths is not None else sum(encoder_widths)
        )
        self.pad_value = pad_value

        self.encoder = encoder
        if encoder:
            self.return_maps = True
        else:
            self.return_maps = False

        if decoder_widths is not None:
            assert len(encoder_widths) == len(decoder_widths)
            assert encoder_widths[-1] == decoder_widths[-1]
        else:
            decoder_widths = encoder_widths

        self.in_conv = ConvBlock(
            nkernels=[input_dim] + [encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value,
            norm=encoder_norm,
        )

        self.down_blocks = nn.LayerList(
            [
                DownConvBlock(
                    d_in=encoder_widths[i],
                    d_out=encoder_widths[i + 1],
                    k=str_conv_k,
                    s=str_conv_s,
                    p=str_conv_p,
                    pad_value=pad_value,
                    norm=encoder_norm,
                    padding_mode=padding_mode,
                )
                for i in range(self.n_stages - 1)
            ]
        )
        self.up_blocks = nn.LayerList(
            [
                UpConvBlock(
                    d_in=decoder_widths[i],
                    d_out=decoder_widths[i - 1],
                    d_skip=encoder_widths[i - 1],
                    k=str_conv_k,
                    s=str_conv_s,
                    p=str_conv_p,
                    norm=encoder_norm,
                    padding_mode=padding_mode,
                )
                for i in range(self.n_stages - 1, 0, -1)
            ]
        )
        self.temporal_aggregator = Temporal_Aggregator(mode="mean")

        if temporal == "mean":
            self.temporal_encoder = Temporal_Aggregator(mode="mean")
        elif temporal == "lstm":
            size = int(input_size / str_conv_s ** (self.n_stages - 1))
            self.temporal_encoder = ConvLSTM(
                input_dim=encoder_widths[-1],
                input_size=(size, size),
                hidden_dim=hidden_dim,
                kernel_size=(3, 3),
            )
            self.out_convlstm = nn.Conv2D(
                in_channels=hidden_dim,
                out_channels=encoder_widths[-1],
                kernel_size=3,
                padding=1,
            )
        elif temporal == "blstm":
            size = int(input_size / str_conv_s ** (self.n_stages - 1))
            self.temporal_encoder = BConvLSTM(
                input_dim=encoder_widths[-1],
                input_size=(size, size),
                hidden_dim=hidden_dim,
                kernel_size=(3, 3),
            )
            self.out_convlstm = nn.Conv2D(
                in_channels=2 * hidden_dim,
                out_channels=encoder_widths[-1],
                kernel_size=3,
                padding=1,
            )
        elif temporal == "mono":
            self.temporal_encoder = None
        self.out_conv = ConvBlock(
            nkernels=[decoder_widths[0]] + out_conv, padding_mode=padding_mode
        )

    def forward(self, input, batch_positions=None):
        pad_mask = (
            (input == self.pad_value).all(axis=-1).all(axis=-1).all(axis=-1)
        )  # BxT pad mask

        out = self.in_conv.smart_forward(input)

        feature_maps = [out]
        # ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)

        # Temporal encoder
        if self.temporal == "mean":
            out = self.temporal_encoder(feature_maps[-1], pad_mask=pad_mask)
        elif self.temporal == "lstm":
            _, out = self.temporal_encoder(feature_maps[-1], pad_mask=pad_mask)
            out = out[0][1]  # take last cell state as embedding
            out = self.out_convlstm(out)
        elif self.temporal == "blstm":
            out = self.temporal_encoder(feature_maps[-1], pad_mask=pad_mask)
            out = self.out_convlstm(out)
        elif self.temporal == "mono":
            out = feature_maps[-1]

        if self.return_maps:
            maps = [out]
        for i in range(self.n_stages - 1):
            if self.temporal != "mono":
                skip = self.temporal_aggregator(
                    feature_maps[-(i + 2)], pad_mask=pad_mask
                )
            else:
                skip = feature_maps[-(i + 2)]
            out = self.up_blocks[i](out, skip)
            if self.return_maps:
                maps.append(out)

        if self.encoder:
            return out, maps
        else:
            out = self.out_conv(out)
            if self.return_maps:
                return out, maps
            else:
                return out
