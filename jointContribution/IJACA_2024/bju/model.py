import math

import paddle
from einops import rearrange
from paddle.nn import Linear
from utils.utils import default


class BaseModel(paddle.nn.Layer):
    def __init__(self, hparams, encoder, decoder):
        super(BaseModel, self).__init__()
        self.nb_hidden_layers = hparams["nb_hidden_layers"]
        self.size_hidden_layers = hparams["size_hidden_layers"]
        self.enc_dim = hparams["encoder"][-1]
        self.dec_dim = hparams["decoder"][0]
        self.bn_bool = hparams["bn_bool"]
        self.res_bool = hparams["res_bool"]
        self.activation = paddle.nn.functional.gelu
        self.encoder = encoder
        self.decoder = decoder
        self.in_layer = self._in_layer(hparams)
        self.hidden_layers = self._hidden_layers(hparams)
        self.out_layer = self._out_layer(hparams)
        self.bn = self._bn(hparams)

    def _in_layer(self, hparams):
        raise NotImplementedError

    def _hidden_layers(self, hparams):
        raise NotImplementedError

    def _out_layer(self, hparams):
        raise NotImplementedError

    def _bn(self, hparams):
        bn = None
        if self.bn_bool:
            bn = paddle.nn.LayerList()
            for n in range(self.nb_hidden_layers):
                bn.append(
                    paddle.nn.BatchNorm1D(
                        num_features=self.size_hidden_layers, use_global_stats=False
                    )
                )
        return bn

    def forward(self, data):
        z, edge_index = data.x, data.edge_index
        if hasattr(self, "get_edge_attr"):
            edge_attr = self.get_edge_attr(z, edge_index)
        z = self.encoder(z)
        if self.enc_dim == self.dec_dim:
            z_in = z
        if hasattr(self, "get_edge_attr"):
            z = self.in_layer(z, edge_index, edge_attr)
        else:
            z = self.in_layer(z, edge_index)
        if self.bn_bool:
            z = self.bn[0](z)
        z = self.activation(z)
        for n in range(self.nb_hidden_layers - 1):
            if hasattr(self, "res_bool") and self.res_bool:
                z_res = z
            if hasattr(self, "get_edge_attr"):
                z = self.hidden_layers[n](z, edge_index, edge_attr)
            else:
                z = self.hidden_layers[n](z, edge_index)
            if self.bn_bool:
                z = self.bn[n + 1](z)
            z = self.activation(z)
            if hasattr(self, "res_bool") and self.res_bool:
                z = z + z_res
        if hasattr(self, "get_edge_attr"):
            z = self.out_layer(z, edge_index, edge_attr)
        else:
            z = self.out_layer(z, edge_index)
        if self.enc_dim == self.dec_dim:
            z = z + z_in
        z = self.decoder(z)
        return z


class NN(BaseModel):
    def __init__(self, hparams, encoder, decoder):
        self.enc_dim = hparams["encoder"][-1]
        self.dec_dim = hparams["decoder"][0]
        super(NN, self).__init__(hparams, encoder, decoder)

    def _in_layer(self, hparams):
        return MLP([self.enc_dim, self.size_hidden_layers])

    def _hidden_layers(self, hparams):
        hidden_layers = paddle.nn.LayerList()
        for n in range(self.nb_hidden_layers - 1):
            hidden_layers.append(
                MLP([self.size_hidden_layers, self.size_hidden_layers])
            )
        return hidden_layers

    def _out_layer(self, hparams):
        return MLP([self.size_hidden_layers, self.dec_dim])


class MLP(paddle.nn.Layer):
    def __init__(
        self, channel_list, dropout=0.0, batch_norm=True, activation_first=False
    ):
        super().__init__()
        assert len(channel_list) >= 2
        self.channel_list = channel_list
        self.dropout = dropout
        self.activation_first = activation_first
        self.lins = paddle.nn.LayerList()
        for dims in zip(self.channel_list[:-1], self.channel_list[1:]):
            self.lins.append(Linear(*dims))
        self.norms = paddle.nn.LayerList()
        for dim in zip(self.channel_list[1:-1]):
            self.norms.append(
                paddle.nn.BatchNorm1D(num_features=dim, use_global_stats=False)
                if batch_norm
                else paddle.nn.Identity()
            )
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            self._reset_parameters(lin)
        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()

    def calculate_fan_in_and_fan_out(self, tensor):
        dimensions = tensor.ndim
        if dimensions < 2:
            raise ValueError(
                "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
            )

        if dimensions == 2:  # Linear
            fan_in = tensor.shape[1]
            fan_out = tensor.shape[0]
        else:
            num_input_fmaps = tensor.shape[1]
            num_output_fmaps = tensor.shape[0]
            receptive_field_size = 1
            if tensor.ndim > 2:
                receptive_field_size = tensor[0][
                    0
                ].numel()  # numel returns the number of elements in the tensor
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size
        return fan_in, fan_out

    def _reset_parameters(self, lin) -> None:
        kaiming_init = paddle.nn.initializer.KaimingUniform()
        kaiming_init(lin.weight)
        if lin.bias is not None:
            fan_in, _ = self.calculate_fan_in_and_fan_out(lin.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            uniform_init = paddle.nn.initializer.Uniform(-bound, bound)
            uniform_init(lin.bias)

    def forward(self, x, edge_index=None):
        x = self.lins[0](x)
        for lin, norm in zip(self.lins[1:], self.norms):
            if self.activation_first:
                x = paddle.nn.functional.gelu(x=x)
            x = norm(x)
            if not self.activation_first:
                x = paddle.nn.functional.gelu(x=x)
            x = paddle.nn.functional.dropout(
                x=x, p=self.dropout, training=self.training
            )
            x = lin.forward(x)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self.channel_list)[1:-1]})"


class GEGLU(paddle.nn.Layer):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = paddle.nn.Linear(in_features=dim_in, out_features=dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(chunks=2, axis=-1)
        return x * paddle.nn.functional.gelu(x=gate)


class FeedForward(paddle.nn.Layer):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            paddle.nn.Sequential(
                paddle.nn.Linear(in_features=dim, out_features=inner_dim),
                paddle.nn.GELU(),
            )
            if not glu
            else GEGLU(dim, inner_dim)
        )
        self.net = paddle.nn.Sequential(
            project_in,
            paddle.nn.Dropout(p=dropout),
            paddle.nn.Linear(in_features=inner_dim, out_features=dim_out),
        )

    def forward(self, x):
        return self.net(x)


class FCLayer(paddle.nn.Layer):
    def __init__(self, query_dim, context_dim=None, dropout=0.0):
        super().__init__()
        context_dim = default(context_dim, query_dim)
        self.to_out = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=context_dim, out_features=query_dim),
            paddle.nn.Dropout(p=dropout),
        )

    def forward(self, x, context=None):
        context = default(context, x)
        return self.to_out(context)


class GeoCA3DBlock(paddle.nn.Layer):
    def __init__(self, dim, dropout=0.0, context_dim=None, gated_ff=True):
        super().__init__()
        self.fc1 = FCLayer(query_dim=dim, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.fc2 = FCLayer(query_dim=dim, context_dim=context_dim, dropout=dropout)
        self.norm1 = paddle.nn.LayerNorm(normalized_shape=dim)
        self.norm2 = paddle.nn.LayerNorm(normalized_shape=dim)

    def forward(self, x, context=None):
        x = self.fc1(self.norm1(x)) + x
        x = self.fc2(x, context=context) + x
        x = self.ff(self.norm2(x)) + x
        return x


class GeoCA3D(paddle.nn.Layer):
    def __init__(
        self,
        cfd_model,
        geom_encoder=None,
        geom_proj=None,
        in_out_dim=64,
        dropout=0.0,
        context_dim=512,
        gated_ff=True,
    ) -> None:
        super().__init__()
        self.geom_encoder = geom_encoder
        self.geom_proj = geom_proj
        self.cfd_model = cfd_model
        if self.geom_encoder is not None:
            self.n_blocks = self.cfd_model.nb_hidden_layers + 2
            dims = (
                [in_out_dim]
                + [self.cfd_model.size_hidden_layers] * self.cfd_model.nb_hidden_layers
                + [in_out_dim]
            )
            self.blocks = paddle.nn.LayerList(
                sublayers=[
                    GeoCA3DBlock(
                        dim=dim, dropout=dropout, context_dim=context_dim, gated_ff=True
                    )
                    for dim in dims
                ]
            )

    def forward(self, data):
        cfd_data, geom_data = data
        if self.geom_encoder is None:
            x = self.cfd_model(cfd_data)
            return x
        x, edge_index = cfd_data.x, cfd_data.edge_index
        if hasattr(self.cfd_model, "get_edge_attr"):
            edge_attr = self.cfd_model.get_edge_attr(x, edge_index)
        x = self.cfd_model.encoder(x)
        z = self.geom_encoder(geom_data) @ self.geom_proj
        z = z / z.norm(axis=-1, keepdim=True)
        z = z.repeat_interleave(repeats=tuple(x.shape)[0] // tuple(z.shape)[0], axis=0)
        z = rearrange(z, "(b n) d -> b n d", n=1)
        if self.cfd_model.enc_dim == self.cfd_model.dec_dim:
            x_in = x
        x = rearrange(x, "(b n) d -> b n d", n=1)
        x = self.blocks[0](x, context=z)
        x = rearrange(x, "b n d -> (b n) d")
        if hasattr(self.cfd_model, "get_edge_attr"):
            x = self.cfd_model.in_layer(x, edge_index, edge_attr)
        else:
            x = self.cfd_model.in_layer(x, edge_index)
        if self.cfd_model.bn_bool:
            x = self.cfd_model.bn[0](x)
        x = self.cfd_model.activation(x)
        for i in range(1, self.n_blocks - 2):
            if hasattr(self.cfd_model, "res_bool") and self.cfd_model.res_bool:
                x_res = x
            x = rearrange(x, "(b n) d -> b n d", n=1)
            x = self.blocks[i](x, context=z)
            x = rearrange(x, "b n d -> (b n) d")
            if hasattr(self.cfd_model, "get_edge_attr"):
                x = self.cfd_model.hidden_layers[i - 1](x, edge_index, edge_attr)
            else:
                x = self.cfd_model.hidden_layers[i - 1](x, edge_index)
            if self.cfd_model.bn_bool:
                x = self.cfd_model.bn[i](x)
            x = self.cfd_model.activation(x)
            if hasattr(self.cfd_model, "res_bool") and self.cfd_model.res_bool:
                x = x + x_res
        x = rearrange(x, "(b n) d -> b n d", n=1)
        x = self.blocks[-2](x, context=z)
        x = rearrange(x, "b n d -> (b n) d")
        if hasattr(self.cfd_model, "get_edge_attr"):
            x = self.cfd_model.out_layer(x, edge_index, edge_attr)
        else:
            x = self.cfd_model.out_layer(x, edge_index)
        x = rearrange(x, "(b n) d -> b n d", n=1)
        x = self.blocks[-1](x, context=z)
        x = rearrange(x, "b n d -> (b n) d")
        if self.cfd_model.enc_dim == self.cfd_model.dec_dim:
            x = x + x_in
        x = self.cfd_model.decoder(x)
        return x
