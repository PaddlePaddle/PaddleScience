import enum
import math
from abc import abstractmethod
from collections import OrderedDict

import numpy as np
import paddle

DEFAULT_W0 = 30.0


###################### ConFILD Model #######################
class Swish(paddle.nn.Layer):
    """
    Swish activation function: f(x) = x * sigmoid(x).

    A smooth, non-monotonic activation function that has been shown to work
    better than ReLU on deeper models across a number of challenging datasets.
    """

    def __init__(self):
        super().__init__()
        self.Sigmoid = paddle.nn.Sigmoid()

    def forward(self, x):
        """
        Apply Swish activation.

        Args:
            x (paddle.Tensor): Input tensor.

        Returns:
            paddle.Tensor: Output tensor with same shape as input.
        """
        return x * self.Sigmoid(x)


class Sine(paddle.nn.Layer):
    """
    Sine activation function for SIREN (Sinusoidal Representation Networks).

    Args:
        w0 (float, optional): Frequency parameter for sine activation. Defaults to DEFAULT_W0 (30.0).
    """

    def __init__(self, w0=DEFAULT_W0):
        self.w0 = w0
        super().__init__()

    def forward(self, input):
        """
        Apply sine activation with frequency modulation.

        Args:
            input (paddle.Tensor): Input tensor.

        Returns:
            paddle.Tensor: sin(w0 * input).
        """
        return paddle.sin(x=self.w0 * input)


def sine_init(m, w0=DEFAULT_W0):
    """
    Weight initialization for SIREN hidden layers.

    Initializes weights uniformly in [-√(6/n)/w0, √(6/n)/w0] where n is input dimension.
    This initialization is critical for maintaining stable signal propagation in SIREN networks.

    Args:
        m (paddle.nn.Layer): Layer to initialize (must have 'weight' attribute).
        w0 (float, optional): Frequency parameter. Defaults to DEFAULT_W0.
    """
    with paddle.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.shape[-1]
            m.weight.uniform_(
                min=-math.sqrt(6 / num_input) / w0, max=math.sqrt(6 / num_input) / w0
            )


def first_layer_sine_init(m):
    """
    Weight initialization for SIREN first layer.

    Initializes weights uniformly in [-1/n, 1/n] where n is input dimension.
    Different from hidden layers to handle raw coordinate inputs properly.

    Args:
        m (paddle.nn.Layer): Layer to initialize (must have 'weight' attribute).
    """
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
    Batch-wise linear transformation layer that supports manual parameter injection.

    This layer extends paddle.nn.Linear to allow passing parameters explicitly,
    which is useful for meta-learning and hypernetwork applications.

    Args:
        in_features (int): Size of input features.
        out_features (int): Size of output features.

    Note:
        - Weight shape: (out_features, in_features)
        - Bias shape: (out_features,)
    """

    __doc__ = paddle.nn.Linear.__doc__

    def forward(self, input, params=None):
        """
        Forward pass with optional external parameters.

        Args:
            input (paddle.Tensor): Input tensor of shape (..., in_features).
            params (OrderedDict, optional): External parameters dict containing 'weight' and optionally 'bias'.
                                           If None, uses internal parameters. Defaults to None.

        Returns:
            paddle.Tensor: Output tensor of shape (..., out_features).
        """
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get("bias", None)
        weight = params["weight"]

        output = paddle.matmul(x=input, y=weight)
        if bias is not None:
            output += bias.unsqueeze(axis=-2)
        return output


class FeatureMapping:
    """
    Feature mapping class for Fourier Feature Networks.

    Supports multiple mapping strategies including Gaussian random Fourier features,
    positional encoding, and radial basis functions (RBF) for improving coordinate-based
    neural network representations.

    Reference:
        Tancik et al. "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
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
        Initialize feature mapping.

        Args:
            in_features (int): Number of input features.
            mode (str, optional): Mapping mode. Options: "basic", "gaussian", "positional", "rbf". Defaults to "basic".
            gaussian_mapping_size (int, optional): Output dimension for Gaussian mapping. Defaults to 256.
            gaussian_rand_key (int, optional): Random seed for Gaussian mapping. Defaults to 0.
            gaussian_tau (float, optional): Standard deviation for Gaussian mapping. Defaults to 1.0.
            pe_num_freqs (int, optional): Number of frequency bands for positional encoding. Defaults to 4.
            pe_scale (int, optional): Base scale for frequencies in positional encoding. Defaults to 2.
            pe_init_scale (int, optional): Initial scale multiplier for positional encoding. Defaults to 1.
            pe_use_nyquist (bool, optional): Use Nyquist frequency to determine num_freqs. Defaults to True.
            pe_lowest_dim (int, optional): Lowest dimension for Nyquist calculation. Defaults to None.
            rbf_out_features (int, optional): Number of RBF centers. Defaults to None.
            rbf_range (float, optional): Range for RBF center initialization. Defaults to 1.0.
            rbf_std (float, optional): Standard deviation for RBF kernels. Defaults to 0.5.
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
            if pe_use_nyquist and pe_lowest_dim:
                pe_num_freqs = self.get_num_frequencies_nyquist(pe_lowest_dim)
            self.B = pe_init_scale * np.vstack(
                [(pe_scale**i * np.eye(in_features)) for i in range(pe_num_freqs)]
            )
            self.dim = tuple(self.B.shape)[0] * 2
        elif mode == "rbf":
            self.centers = paddle.nn.Parameter(
                paddle.empty(shape=(rbf_out_features, in_features), dtype="float32")
            )
            self.sigmas = paddle.nn.Parameter(
                paddle.empty(shape=rbf_out_features, dtype="float32")
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
        Apply Fourier feature mapping: [sin(2πxB^T), cos(2πxB^T)].

        Args:
            x (paddle.Tensor): Input coordinates of shape (..., in_features).
            B (np.ndarray): Frequency matrix of shape (mapping_size, in_features).

        Returns:
            paddle.Tensor: Fourier features of shape (..., 2 * mapping_size).
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
        distances = paddle.pow(x - self.centers, 2).sum(axis=-1) * self.sigmas
        return self.gaussian(distances)

    @staticmethod
    def gaussian(alpha):
        phi = paddle.exp(x=-1 * paddle.pow(alpha, 2))
        return phi


class SIRENAutodecoder_film(paddle.nn.Layer):
    """
    SIREN (Sinusoidal Representation Networks) with FiLM conditioning for autodecoding.

    This architecture uses sine activations and latent code modulation (FiLM) for
    implicit neural representations. It takes both coordinate inputs and latent codes,
    making it suitable for learning multiple shapes/scenes with a single network.

    Reference:
        Sitzmann et al. "Implicit Neural Representations with Periodic Activation Functions" (NeurIPS 2020)

    Args:
        input_keys (Tuple[str, ...], optional): Keys to get input tensors from dict. First key for coordinates, second for latents.
        output_keys (Tuple[str, ...], optional): Keys to save output tensors into dict.
        in_coord_features (int, optional): Number of input coordinate features (e.g., 2 for 2D, 3 for 3D).
        in_latent_features (int, optional): Number of latent features for conditioning.
        out_features (int, optional): Number of output features (e.g., 3 for RGB).
        num_hidden_layers (int, optional): Number of hidden layers.
        hidden_features (int, optional): Number of hidden layer features.
        outermost_linear (bool, optional): Whether to use linear layer at output. Defaults to False.
        nonlinearity (str, optional): Activation function. Options: "sine", "relu", "tanh", etc. Defaults to "sine".
        weight_init (Callable, optional): Custom weight initialization function. Defaults to None.
        bias_init (Callable, optional): Custom bias initialization function. Defaults to None.
        premap_mode (str, optional): Feature mapping mode before network. Options: "gaussian", "positional", "rbf". Defaults to None.

    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.SIRENAutodecoder_film(
        ...     input_keys=["coords", "latents"],
        ...     output_keys=("output",),
        ...     in_coord_features=2,
        ...     in_latent_features=128,
        ...     out_features=3,
        ...     num_hidden_layers=10,
        ...     hidden_features=128,
        ... )
        >>> input_data = {
        ...     "coords": paddle.randn([1000, 2]),
        ...     "latents": paddle.randn([1000, 128])
        ... }
        >>> out_dict = model(input_data)
        >>> print(out_dict["output"].shape)
        [1000, 3]
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
        **kwargs,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys

        self.premap_mode = premap_mode
        if self.premap_mode is not None:
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
                BatchLinear(in_latent_features, hidden_features, bias_attr=False)
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
        if self.premap_mode is not None:
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
            param.stop_gradient = True


class LatentContainer(paddle.nn.Layer):
    """
    Learnable latent code container for autodecoding applications.

    This module stores and retrieves per-sample latent codes, which can be used
    for representing multiple instances (shapes, scenes) with a single decoder network.
    Supports multi-GPU training and different dimensional arrangements.

    Reference:
        Park et al. "DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation" (CVPR 2019)

    Args:
        input_keys (Tuple[str, ...], optional): Key to get batch indices from dict. Defaults to ("input",).
        output_keys (Tuple[str, ...], optional): Key to save latent codes into dict. Defaults to ("output",).
        N_samples (int, optional): Total number of samples/instances in dataset. Defaults to None.
        N_features (int, optional): Dimension of latent codes. Defaults to None.
        dims (int, optional): Number of spatial dimensions (for proper broadcasting). Defaults to None.
        lumped (bool, optional): If True, adds single dimension; if False, adds dims dimensions. Defaults to False.

    Examples:
        >>> import ppsci
        >>> import paddle
        >>> model = ppsci.arch.LatentContainer(
        ...     N_samples=1600,
        ...     N_features=128,
        ...     dims=2,
        ...     lumped=True
        ... )
        >>> batch_indices = paddle.randint(0, 1600, [32], dtype='int64')
        >>> input_dict = {"input": batch_indices}
        >>> out_dict = model(input_dict)
        >>> print(out_dict["output"].shape)
        [32, 1, 128]
    """

    def __init__(
        self,
        input_keys=("input",),
        output_keys=("output",),
        N_samples=None,
        N_features=None,
        dims=None,
        lumped=False,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.dims = [1] * dims if not lumped else [1]
        self.expand_dims = " ".join(["1" for _ in range(dims)]) if not lumped else "1"
        self.expand_dims = f"N f -> N {self.expand_dims} f"
        self.latents = self.create_parameter(
            shape=(N_samples, N_features),
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0.0),
        )

    def forward(self, batch_ids):
        x = batch_ids[self.input_keys[0]]
        selected_latents = paddle.gather(self.latents, x)
        if len(selected_latents.shape) > 1:
            getShape = (
                [tuple(selected_latents.shape)[0]]
                + self.dims
                + [tuple(selected_latents.shape)[1]]
            )
        else:
            getShape = [-1] + self.dims
        expanded_latents = selected_latents.reshape(getShape)
        return {self.output_keys[0]: expanded_latents}


###################### GaussianDiffusion Model #######################
class ModelVarType(enum.Enum):

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = paddle.to_tensor(data=arr, dtype=paddle.float32)[timesteps]
    while len(tuple(res.shape)) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(shape=broadcast_shape)


def split(x, num_or_sections, axis=0):
    if isinstance(num_or_sections, int):
        return paddle.split(x, x.shape[axis] // num_or_sections, axis)
    else:
        return paddle.split(x, num_or_sections, axis)


class ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()


def mean_flat(tensor):
    return paddle.mean(tensor, axis=list(range(1, len(tensor.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, paddle.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        (
            x
            if isinstance(x, paddle.Tensor)
            else paddle.to_tensor(x, dtype=tensor.dtype, place=tensor.place)
        )
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + paddle.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * paddle.exp(-logvar2)
    )


class GaussianDiffusion:
    """
    Gaussian diffusion process for denoising diffusion probabilistic models (DDPM).

    Implements the forward diffusion process q(x_t|x_0) and reverse denoising process p(x_{t-1}|x_t).
    Supports various parameterizations (epsilon, x_0, x_{t-1}) and variance schedules.

    Reference:
        Ho et al. "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
        Nichol & Dhariwal "Improved Denoising Diffusion Probabilistic Models" (ICML 2021)

    Args:
        betas (np.ndarray): Noise schedule β_t for t=0,...,T-1.
        model_mean_type (ModelMeanType): Parameterization of model output.
        model_var_type (ModelVarType): Variance parameterization (fixed or learned).
        loss_type (LossType): Loss function type (MSE, KL, etc.).
        rescale_timesteps (bool, optional): Rescale timesteps to [0, 1000]. Defaults to False.
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(tuple(betas.shape)) == 1, "betas must be 1-D"
        assert (betas > 0).astype("bool").all() and (betas <= 1).astype("bool").all()

        self.num_timesteps = int(tuple(betas.shape)[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert tuple(self.alphas_cumprod_prev.shape) == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = paddle.randn(x_start.shape)

        sqrt_alpha_cumprod_t = _extract_into_tensor(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alpha_cumprod_t = _extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert tuple(x_t.shape) == tuple(xprev.shape)
        return (
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, tuple(x_t.shape))
            * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1,
                t,
                tuple(x_t.shape),
            )
            * x_t
        )

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert tuple(x_t.shape) == tuple(eps.shape)
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, tuple(x_t.shape))
            * x_t
            - _extract_into_tensor(
                self.sqrt_recipm1_alphas_cumprod, t, tuple(x_t.shape)
            )
            * eps
        )

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        if model_kwargs is None:
            model_kwargs = {}
        B, C = tuple(x.shape)[:2]
        assert tuple(t.shape) == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert tuple(model_output.shape) == (B, C * 2, *tuple(x.shape)[2:])
            model_output, model_var_values = split(
                x=model_output, num_or_sections=C, axis=1
            )
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = paddle.exp(x=model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, tuple(x.shape)
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, tuple(x.shape))
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = paddle.exp(x=model_log_variance)
        else:
            model_variance, model_log_variance = {
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, tuple(x.shape))
            model_log_variance = _extract_into_tensor(
                model_log_variance, t, tuple(x.shape)
            )

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clip(min=-1, max=1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)
        assert (
            tuple(model_mean.shape)
            == tuple(model_log_variance.shape)
            == tuple(pred_xstart.shape)
            == tuple(x.shape)
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert tuple(x_start.shape) == tuple(x_t.shape)
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, tuple(x_t.shape))
            * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, tuple(x_t.shape)) * x_t
        )
        posterior_variance = _extract_into_tensor(
            self.posterior_variance, t, tuple(x_t.shape)
        )
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, tuple(x_t.shape)
        )
        assert (
            tuple(posterior_mean.shape)[0]
            == tuple(posterior_variance.shape)[0]
            == tuple(posterior_log_variance_clipped.shape)[0]
            == tuple(x_start.shape)[0]
        )
        return (posterior_mean, posterior_variance, posterior_log_variance_clipped)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.astype(dtype="float32") * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = p_mean_var["mean"].astype(dtype="float32") + p_mean_var[
            "variance"
        ] * gradient.astype(dtype="float32")
        return new_mean

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - paddle.sqrt(1 - alpha_bar) * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = paddle.randn(shape=x.shape, dtype=x.dtype)
        nonzero_mask = (
            (t != 0)
            .astype(dtype="float32")
            .reshape([-1, *([1] * (len(tuple(x.shape)) - 1))])
        )
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = (
            out["mean"] + nonzero_mask * paddle.exp(x=0.5 * out["log_variance"]) * noise
        )
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = paddle.randn(shape=shape)
        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        for i in indices:
            t = paddle.to_tensor(data=[i] * shape[0])
            with paddle.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def training_losses(
        self, model, x_start, t, model_kwargs=None, noise=None, valid=False
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = paddle.randn(x_start.shape)

        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        # terms = {}
        # model_output = model(x_t, t)

        # # Handle different model outputs
        # if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
        #     assert model_output.shape[1] == 2 * x_start.shape[1], "Output channels must be 2x input channels"
        #     model_output, model_var_values = split(model_output, 2, axis=1)

        # Calculate the MSE loss for epsilon prediction
        terms = {}
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=True,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = split(model_output, C, axis=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = paddle.concat(
                    [model_output.detach(), model_var_values], axis=1
                )
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=True,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape

            if valid is False:
                terms["mse"] = mean_flat((target - model_output) ** 2)
                if "vb" in terms:
                    terms["loss"] = terms["mse"] + terms["vb"]
                else:
                    terms["loss"] = terms["mse"]
            else:
                terms["valid_mse"] = mean_flat((target - model_output) ** 2)
                if "vb" in terms:
                    terms["loss"] = terms["valid_mse"] + terms["vb"]
                else:
                    terms["loss"] = terms["valid_mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = paddle.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = paddle.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = paddle.log(cdf_plus.clip(min=1e-12))
    log_one_minus_cdf_min = paddle.log((1.0 - cdf_min).clip(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = paddle.where(
        x < -0.999,
        log_cdf_plus,
        paddle.where(
            x > 0.999, log_one_minus_cdf_min, paddle.log(cdf_delta.clip(min=1e-12))
        ),
    )
    assert log_probs.shape == x.shape
    return log_probs


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (
        1.0 + paddle.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * paddle.pow(x, 3)))
    )


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class SpacedDiffusion(GaussianDiffusion):
    """
    Accelerated diffusion process that skips timesteps for faster sampling.

    Implements DDIM-style sampling by using a subset of timesteps from the original
    diffusion process, enabling faster inference without retraining the model.

    Reference:
        Song et al. "Denoising Diffusion Implicit Models" (ICLR 2021)

    Args:
        use_timesteps (Sequence[int]): Collection of timesteps to retain from original process
                                       (e.g., [0, 10, 20, ..., 1000] for 100-step sampling).
        **kwargs: Additional arguments for base GaussianDiffusion (betas, model_mean_type, etc.).
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])
        base_diffusion = GaussianDiffusion(**kwargs)
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(self, model, *args, **kwargs):
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = paddle.to_tensor(
            data=self.timestep_map, dtype=ts.dtype  # , place=ts.place
        )
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.astype(dtype="float32") * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


###################### UNET Model #######################
def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return paddle.nn.Conv1D(*args, **kwargs)
    elif dims == 2:
        return paddle.nn.Conv2D(*args, **kwargs)
    elif dims == 3:
        return paddle.nn.Conv3D(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    return paddle.nn.Linear(*args, **kwargs)


class TimestepBlock(paddle.nn.Layer):
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
        pass


class ResBlock(TimestepBlock):
    """
    Residual block with timestep embedding for diffusion models.

    Implements a residual connection with two convolutional layers, timestep conditioning,
    and optional up/downsampling. Supports FiLM-style adaptive normalization.

    Args:
        channels (int): Number of input channels.
        emb_channels (int): Number of timestep embedding channels.
        dropout (float): Dropout probability.
        out_channels (int, optional): Number of output channels. Defaults to channels.
        use_conv (bool, optional): Use conv for skip connection if channels differ. Defaults to False.
        use_scale_shift_norm (bool, optional): Use FiLM-style conditioning. Defaults to False.
        dims (int, optional): Spatial dimensions (1D/2D/3D). Defaults to 2.
        use_checkpoint (bool, optional): Use gradient checkpointing. Defaults to False.
        up (bool, optional): Apply upsampling. Defaults to False.
        down (bool, optional): Apply downsampling. Defaults to False.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.in_layers = paddle.nn.Sequential(
            normalization(channels),
            paddle.nn.Silu(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = paddle.nn.Identity()
        self.emb_layers = paddle.nn.Sequential(
            paddle.nn.Silu(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = paddle.nn.Sequential(
            normalization(self.out_channels),
            paddle.nn.Silu(),
            paddle.nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )
        if self.out_channels == channels:
            self.skip_connection = paddle.nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).astype(h.dtype)
        while len(tuple(emb_out.shape)) < len(tuple(h.shape)):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            (scale, shift) = paddle.chunk(x=emb_out, chunks=2, axis=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class TimestepEmbedSequential(paddle.nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


NUM_CLASSES = 1000


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return paddle.nn.AvgPool1D(*args, **kwargs, exclusive=False)
    elif dims == 2:
        return paddle.nn.AvgPool2D(*args, **kwargs, exclusive=False)
    elif dims == 3:
        return paddle.nn.AvgPool3D(*args, **kwargs, exclusive=False)
    raise ValueError(f"unsupported dimensions: {dims}")


class Downsample(paddle.nn.Layer):
    """
    Spatial downsampling layer (2x reduction).

    Can use either strided convolution or average pooling for downsampling.

    Args:
        channels (int): Number of input channels.
        use_conv (bool): Use strided conv (True) or avg pooling (False).
        dims (int, optional): Spatial dimensions. Defaults to 2.
        out_channels (int, optional): Number of output channels. Defaults to channels.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        """Apply downsampling."""
        assert tuple(x.shape)[1] == self.channels
        return self.op(x)


class Upsample(paddle.nn.Layer):
    """
    Spatial upsampling layer (2x expansion).

    Uses nearest-neighbor interpolation followed by optional convolution.

    Args:
        channels (int): Number of input channels.
        use_conv (bool): Apply convolution after upsampling.
        dims (int, optional): Spatial dimensions. Defaults to 2.
        out_channels (int, optional): Number of output channels. Defaults to channels.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        """Apply upsampling."""
        assert tuple(x.shape)[1] == self.channels
        if self.dims == 3:
            x = paddle.nn.functional.interpolate(
                x=x,
                size=(tuple(x.shape)[2], tuple(x.shape)[3] * 2, tuple(x.shape)[4] * 2),
                mode="nearest",
            )
        else:
            x = paddle.nn.functional.interpolate(x=x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


def count_flops_attn(model, _x, y):
    b, c, *spatial = tuple(y[0].shape)
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += paddle.to_tensor(data=[matmul_ops], dtype="float64")


class QKVAttentionLegacy(paddle.nn.Layer):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = tuple(qkv.shape)
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        # split_size: 为 int 时 torch 表示块的大小，paddle 表示块的个数
        (q, k, v) = split(qkv.reshape((bs * self.n_heads, ch * 3, length)), ch, 1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = paddle.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = paddle.nn.functional.softmax(
            x=weight.astype(dtype="float32"), axis=-1
        ).astype(weight.dtype)
        a = paddle.einsum("bts,bcs->bct", weight, v)
        return a.reshape((bs, -1, length))

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(paddle.nn.Layer):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = tuple(qkv.shape)
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        (q, k, v) = qkv.chunk(chunks=3, axis=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = paddle.einsum(  # 非复数
            "bct,bcs->bts",
            (q * scale).reshape([bs * self.n_heads, ch, length]),
            (k * scale).reshape([bs * self.n_heads, ch, length]),
        )
        weight = paddle.nn.functional.softmax(
            x=weight.astype(dtype="float32"), axis=-1
        ).astype(weight.dtype)
        a = paddle.einsum(
            "bts,bcs->bct", weight, v.reshape((bs * self.n_heads, ch, length))
        )
        return a.reshape((bs, -1, length))

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class GroupNorm32(paddle.nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.astype(dtype="float32")).astype(x.dtype)


def normalization(channels):
    return GroupNorm32(32, channels)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def checkpoint(func, inputs, params, flag):
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with paddle.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [stop_gradient(x, stop=False) for x in ctx.input_tensors]
        with paddle.enable_grad():
            shallow_copies = [x.reshape(x.shape) for x in ctx.input_tensors]
            # print(shallow_copies)
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = paddle.grad(
            outputs=output_tensors,
            inputs=ctx.input_tensors + ctx.input_params,
            grad_outputs=output_grads,
            allow_unused=True,
            # retain_graph=True, create_graph=False
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors

        # 确保将input_grads转换为元组，然后与(None, None)连接
        # PyLayer要求backward方法返回元组类型
        # if input_grads:
        return tuple(input_grads)
        # else:
        # return (None, None)


def stop_gradient(input, stop):
    input.stop_gradient = stop
    return input


class AttentionBlock(paddle.nn.Layer):
    """
    Self-attention block for spatial feature maps.

    Applies multi-head self-attention over spatial locations in feature maps,
    allowing the model to capture long-range dependencies.

    Args:
        channels (int): Number of input/output channels.
        num_heads (int, optional): Number of attention heads. Defaults to 1.
        num_head_channels (int, optional): Channels per head (overrides num_heads). Defaults to -1.
        use_checkpoint (bool, optional): Use gradient checkpointing. Defaults to False.
        use_new_attention_order (bool, optional): Use optimized attention implementation. Defaults to False.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = tuple(x.shape)
        x = x.reshape((b, c, -1))
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape((b, c, *spatial))


def convert_module_to_f16(l):
    if isinstance(l, (paddle.nn.Conv1D, paddle.nn.Conv2D, paddle.nn.Conv3D)):
        l.weight.data = l.weight.data.astype(dtype="float16")
        if l.bias is not None:
            l.bias.data = l.bias.data.astype(dtype="float16")


def convert_module_to_f32(l):
    if isinstance(l, (paddle.nn.Conv1D, paddle.nn.Conv2D, paddle.nn.Conv3D)):
        l.weight.data = l.weight.data.astype(dtype="float32")
        if l.bias is not None:
            l.bias.data = l.bias.data.astype(dtype="float32")


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings for diffusion models.

    Similar to positional encodings in transformers, but for continuous timesteps.
    Uses sinusoids of exponentially increasing frequencies.

    Args:
        timesteps (paddle.Tensor): Timestep values of shape (batch_size,).
        dim (int): Embedding dimension.
        max_period (int, optional): Maximum period for sinusoids. Defaults to 10000.

    Returns:
        paddle.Tensor: Timestep embeddings of shape (batch_size, dim).
    """
    half = dim // 2
    freqs = paddle.exp(
        x=-math.log(max_period)
        * paddle.arange(start=0, end=half, dtype="float32")
        / half
    )  # .to(paddle.CUDAPlace(0))
    args = timesteps[:, None].astype(dtype="float32") * freqs[None]
    embedding = paddle.concat(x=[paddle.cos(x=args), paddle.sin(x=args)], axis=-1)
    if dim % 2:
        embedding = paddle.concat(
            x=[embedding, paddle.zeros_like(x=embedding[:, :1])], axis=-1
        )
    return embedding


class UNetModel(paddle.nn.Layer):
    """
    Full UNet model with attention and timestep embedding for diffusion models.

    Implements a U-Net architecture with residual blocks, self-attention at multiple resolutions,
    and timestep conditioning via adaptive normalization (FiLM). Designed for denoising diffusion
    probabilistic models (DDPM) and can be conditioned on class labels.

    Reference:
        Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI 2015)
        Dhariwal & Nichol "Diffusion Models Beat GANs on Image Synthesis" (NeurIPS 2021)

    Args:
        image_size (int): Input image size (maintained for interface compatibility).
        in_channels (int): Number of channels in input tensor.
        model_channels (int): Base channel count for model (multiplied by channel_mult).
        out_channels (int): Number of channels in output tensor.
        num_res_blocks (int): Number of residual blocks per downsampling level.
        attention_resolutions (list/tuple): Downsample factors where to apply attention (e.g., [4, 8, 16]).
        dropout (float, optional): Dropout probability in residual blocks. Defaults to 0.0.
        channel_mult (tuple, optional): Channel multipliers per level (e.g., (1, 2, 4, 8)). Defaults to (1, 2, 4, 8).
        conv_resample (bool, optional): Use learned convolutional up/downsampling. Defaults to True.
        dims (int, optional): Data dimensionality (1=1D, 2=2D, 3=3D). Defaults to 2.
        num_classes (int, optional): Number of classes for class-conditional generation. Defaults to None.
        use_checkpoint (bool, optional): Enable gradient checkpointing to save memory. Defaults to False.
        use_fp16 (bool, optional): Use float16 precision for forward pass. Defaults to False.
        num_heads (int, optional): Number of attention heads in each attention block. Defaults to 1.
        num_head_channels (int, optional): Fixed channels per head (overrides num_heads if set). Defaults to -1.
        num_heads_upsample (int, optional): Attention heads for upsampling blocks. Defaults to -1 (use num_heads).
        use_scale_shift_norm (bool, optional): Use FiLM-style conditioning in ResBlocks. Defaults to False.
        resblock_updown (bool, optional): Use ResBlocks for up/downsampling instead of conv layers. Defaults to False.
        use_new_attention_order (bool, optional): Use optimized QKV attention implementation. Defaults to False.

    Examples:
        >>> import ppsci
        >>> import paddle
        >>> model = ppsci.arch.UNetModel(
        ...     image_size=64,
        ...     in_channels=3,
        ...     model_channels=128,
        ...     out_channels=3,
        ...     num_res_blocks=2,
        ...     attention_resolutions=[8, 16],
        ...     channel_mult=(1, 2, 4, 8),
        ...     num_heads=4,
        ... )
        >>> x = paddle.randn([4, 3, 64, 64])
        >>> t = paddle.randint(0, 1000, [4])
        >>> out = model(x, t)
        >>> print(out.shape)
        [4, 3, 64, 64]
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = "float16" if use_fp16 else "float32"
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        time_embed_dim = model_channels * 4
        self.time_embed = paddle.nn.Sequential(
            linear(model_channels, time_embed_dim),
            paddle.nn.Silu(),
            linear(time_embed_dim, time_embed_dim),
        )
        if self.num_classes is not None:
            self.label_emb = paddle.nn.Embedding(
                num_embeddings=self.num_classes, embedding_dim=time_embed_dim
            )
        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = paddle.nn.LayerList(
            sublayers=[
                TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))
            ]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = []
                layers.append(
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.output_blocks = paddle.nn.LayerList(sublayers=[])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = []
                layers.append(
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
        self.out = paddle.nn.Sequential(
            normalization(ch),
            paddle.nn.Silu(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert tuple(y.shape) == (tuple(x.shape)[0],)
            emb = emb + self.label_emb(y)
        h = x.astype(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = paddle.concat(x=[h, hs.pop()], axis=1)
            h = module(h, emb)
        h = h.astype(x.dtype)
        return self.out(h)
