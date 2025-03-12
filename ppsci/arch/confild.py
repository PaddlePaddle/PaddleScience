import math
from collections import OrderedDict
from einops import rearrange

import numpy as np
import paddle

DEFAULT_W0 = 30.


class Swish(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.Sigmoid = paddle.nn.Sigmoid()

    def forward(self, x):
        return x * self.Sigmoid(x)


class Sine(paddle.nn.Layer):
    def __init__(self, w0=DEFAULT_W0):
        self.w0 = w0
        super().__init__()

    def forward(self, input):
        return paddle.sin(x=self.w0 * input)


def sine_init(m, w0=DEFAULT_W0):
    with paddle.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.shape[-1]
            m.weight.uniform_(
                min=-math.sqrt(6 / num_input) / w0, max=math.sqrt(6 / num_input) / w0
            )


def first_layer_sine_init(m):
    with paddle.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.shape[-1]
            m.weight.uniform_(min=-1 / num_input, max=1 / num_input)


def __check_Linear_weight(m):
    if isinstance(m, paddle.nn.Linear):
        if hasattr(m, "weight"):
            return True
    return False


def init_weights_normal(m):
    if __check_Linear_weight(m):
        init_KaimingNormal = paddle.nn.initializer.KaimingNormal(
            nonlinearity="relu", negative_slope=0.0
        )
        init_KaimingNormal(m.weight)


def init_weights_selu(m):
    if __check_Linear_weight(m):
        num_input = m.weight.shape[-1]
        init_Normal = paddle.nn.initializer.Normal(std=1 / math.sqrt(num_input))
        init_Normal(m.weight)


def init_weights_elu(m):
    if __check_Linear_weight(m):
        num_input = m.weight.shape[-1]
        init_Normal = paddle.nn.initializer.Normal(
            std=math.sqrt(1.5505188080679277) / math.sqrt(num_input)
        )
        init_Normal(m.weight)


def init_weights_xavier(m):
    if __check_Linear_weight(m):
        init_XavierNormal = paddle.nn.initializer.XavierNormal()
        init_XavierNormal(m.weight)


NLS_AND_INITS = {
    "sine": (Sine(), sine_init, first_layer_sine_init),
    "relu": (paddle.nn.ReLU(), init_weights_normal, None),
    "sigmoid": (paddle.nn.Sigmoid(), init_weights_xavier, None),
    "tanh": (paddle.nn.Tanh(), init_weights_xavier, None),
    "selu": (paddle.nn.SELU(), init_weights_selu, None),
    "softplus": (paddle.nn.Softplus(), init_weights_normal, None),
    "elu": (paddle.nn.ELU(), init_weights_elu, None),
    "swish": (Swish(), init_weights_xavier, None),
}


class BatchLinear(paddle.nn.Linear):
    """
    This is a linear transformation implemented manually. It also allows maually input parameters.
    for initialization, (in_features, out_features) needs to be provided.
    weight is of shape (out_features*in_features)
    bias is of shape (out_features)

    """

    __doc__ = paddle.nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get("bias", None)
        weight = params["weight"]
        output = paddle.matmul(
            x=input,
            y=weight.transpose(
                perm=[*[i for i in range(len(tuple(weight.shape)) - 2)], -1, -2]
            ),
        )
        if not bias == None:
            output += bias.unsqueeze(axis=-2)
        return output


class FeatureMapping:
    """
    This is feature mapping class for  fourier feature networks
    """

    def __init__(
        self,
        in_features,
        mode="basic",
        gaussian_mapping_size=256,
        gaussian_rand_key=0,
        gaussian_tau=1.0,
        pe_num_freqs=4,
        pe_scale=2,
        pe_init_scale=1,
        pe_use_nyquist=True,
        pe_lowest_dim=None,
        rbf_out_features=None,
        rbf_range=1.0,
        rbf_std=0.5,
    ):
        """
        inputs:
            in_freatures: number of input features
            mapping_size: output features for Gaussian mapping
            rand_key: random key for Gaussian mapping
            tau: standard deviation for Gaussian mapping
            num_freqs: number of frequencies for P.E.
            scale = 2: base scale of frequencies for P.E.
            init_scale: initial scale for P.E.
            use_nyquist: use nyquist to calculate num_freqs or not.

        """
        self.mode = mode
        if mode == "basic":
            self.B = np.eye(in_features)
        elif mode == "gaussian":
            rng = np.random.default_rng(gaussian_rand_key)
            self.B = rng.normal(
                loc=0.0, scale=gaussian_tau, size=(gaussian_mapping_size, in_features)
            )
        elif mode == "positional":
            if pe_use_nyquist == "True" and pe_lowest_dim:
                pe_num_freqs = self.get_num_frequencies_nyquist(pe_lowest_dim)
            self.B = pe_init_scale * np.vstack(
                [(pe_scale**i * np.eye(in_features)) for i in range(pe_num_freqs)]
            )
            self.dim = tuple(self.B.shape)[0] * 2
        elif mode == "rbf":
            self.centers = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.empty(
                    shape=(rbf_out_features, in_features), dtype="float32"
                )
            )
            self.sigmas = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.empty(shape=rbf_out_features, dtype="float32")
            )
            init_Uniform = paddle.nn.initializer.Uniform(
                low=-1 * rbf_range, high=rbf_range
            )
            init_Uniform(self.centers)
            init_Constant = paddle.nn.initializer.Constant(value=rbf_std)
            init_Constant(self.sigmas)

    def __call__(self, input):
        if self.mode in ["basic", "gaussian", "positional"]:
            return self.fourier_mapping(input, self.B)
        elif self.mode == "rbf":
            return self.rbf_mapping(input)

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    @staticmethod
    def fourier_mapping(x, B):
        """
        x is the input, B is the reference information
        """
        if B is None:
            return x
        else:
            B = paddle.to_tensor(data=B, dtype="float32", place=x.place)
            x_proj = 2.0 * np.pi * x @ B.T
            return paddle.concat(
                x=[paddle.sin(x=x_proj), paddle.cos(x=x_proj)], axis=-1
            )

    def rbf_mapping(self, x):
        size = tuple(x.shape)[:-1] + tuple(self.centers.shape)
        x = x.unsqueeze(axis=-2).expand(shape=size)
        distances = (x - self.centers).pow(y=2).sum(axis=-1) * self.sigmas
        return self.gaussian(distances)

    @staticmethod
    def gaussian(alpha):
        phi = paddle.exp(x=-1 * alpha.pow(y=2))
        return phi


class SIRENAutodecoder_film(paddle.nn.Layer):
    """
    siren network with author decoding
    """

    def __init__(
        self,
        input_keys,
        output_keys,
        in_coord_features,
        in_latent_features,
        out_features,
        num_hidden_layers,
        hidden_features,
        outermost_linear=False,
        nonlinearity="sine",
        weight_init=None,
        bias_init=None,
        premap_mode=None,
        **kwargs
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        
        self.premap_mode = premap_mode
        if not self.premap_mode == None:
            self.premap_layer = FeatureMapping(
                in_coord_features, mode=premap_mode, **kwargs
            )
            in_coord_features = self.premap_layer.dim
        self.first_layer_init = None
        self.nl, nl_weight_init, first_layer_init = NLS_AND_INITS[nonlinearity]
        if weight_init is not None:
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init
        self.net1 = paddle.nn.LayerList(
            sublayers=[BatchLinear(in_coord_features, hidden_features)]
            + [
                BatchLinear(hidden_features, hidden_features)
                for i in range(num_hidden_layers)
            ]
            + [BatchLinear(hidden_features, out_features)]
        )
        self.net2 = paddle.nn.LayerList(
            sublayers=[
                BatchLinear(in_latent_features, hidden_features, bias=False)
                for i in range(num_hidden_layers + 1)
            ]
        )
        if self.weight_init is not None:
            self.net1.apply(self.weight_init)
            self.net2.apply(self.weight_init)
        if first_layer_init is not None:
            self.net1[0].apply(first_layer_init)
            self.net2[0].apply(first_layer_init)
        if bias_init is not None:
            self.net2.apply(bias_init)

    def forward(self, input_data):
        coords = input_data[self.input_keys[0]]
        latents = input_data[self.input_keys[1]]
        if not self.premap_mode == None:
            x = self.premap_layer(coords)
        else:
            x = coords
        for i in range(len(self.net1) - 1):
            x = self.net1[i](x) + self.net2[i](latents)
            x = self.nl(x)
        x = self.net1[-1](x)
        return {self.output_keys[0]: x}

    def disable_gradient(self):
        for param in self.parameters():
            param.stop_gradient = not False


class LatentContainer(paddle.nn.Layer):
    """
    a model container that stores latents for multi GPU
    """

    def __init__(
            self, 
            input_key=("input",),
            output_key=("output",),
            N_samples=None,
            N_features=None,
            dims=None,
            lumped=False
    ):
        super().__init__()
        self.input_keys = input_key
        self.output_keys = output_key
        self.expand_dims = " ".join(["1" for _ in range(dims)]) if not lumped else "1"
        self.expand_dims = f"N f -> N {self.expand_dims} f"
        self.latents = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.zeros(shape=(N_samples, N_features), dtype="float32")
        )

    def forward(self, batch_ids):
        x = batch_ids[self.input_keys[0]]
        return {self.output_keys[0]: rearrange(self.latents[x], self.expand_dims)}