import numpy as np
import paddle


def transform_in(x, config, p=0.1):
    """
    对输入数据进行变换，并返回变换后的数据、时间步长、残差和噪声。

    Args:
        x (dict): 包含输入数据的字典，键为 "x0"，值为 Paddle Tensor。
        config (dict): 配置参数字典，包含扩散参数、数据路径等。
        p (float, optional): 概率值，用于决定是否计算残差。默认为 0.1。

    Returns:
        tuple: 包含四个元素的元组，分别为：
            - x (Paddle Tensor): 变换后的数据。
            - t (Paddle Tensor): 时间步长。
            - dx (Paddle Tensor or None): 残差，如果 flag < p 则为 None。
            - e (Paddle Tensor): 噪声。

    """
    x = x["x0"]
    # print(x)
    betas = get_beta_schedule_train(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = paddle.to_tensor(data=betas).astype(dtype="float32")
    num_timesteps = betas.shape[0]
    b = betas
    n = x.shape[0]
    e = paddle.randn(shape=x.shape, dtype=x.dtype)
    t = paddle.randint(low=0, high=num_timesteps, shape=(n // 2 + 1,))
    t = paddle.concat(x=[t, num_timesteps - t - 1], axis=0)[:n]
    data = np.load(config.data.stat_path)
    x_offset = data["mean"]
    x_scale = data["scale"]
    beta_sub = 1 - b
    cumprod_beta = paddle.cumprod(beta_sub, dim=0)
    if not isinstance(t, paddle.Tensor):
        t = paddle.to_tensor(t, dtype="int64")
    selected = paddle.gather(cumprod_beta, index=t, axis=0)
    a = paddle.reshape(selected, shape=[-1, 1, 1, 1])
    x = x * a.sqrt() + e * (1.0 - a).sqrt()
    flag = np.random.uniform(0, 1)
    if flag < p:
        dx = None
    else:
        dx = voriticity_residual_train(x * x_scale + x_offset) / x_scale
    return x, t, dx, e


def transform_out(outputs):
    loss = (outputs[1] - outputs[0]).square().sum(axis=(1, 2, 3)).mean(axis=0)

    return {"loss": loss}


def train_loss_func(result_dict, *args) -> paddle.Tensor:
    """For model calculation of loss.

    Args:
        result_dict (Dict[str, paddle.Tensor]): The result dict.

    Returns:
        paddle.Tensor: Loss value.
    """
    return result_dict["loss"]


def float_to_complex(x):
    x = paddle.cast(x, dtype="float32")
    img_x = paddle.zeros_like(x)
    x = paddle.complex(x, img_x)
    return x


def voriticity_residual_train(w, re=1000.0, dt=1 / 32):
    w = w.clone()
    out_5 = w
    out_5.stop_gradient = not True
    out_5
    nx = w.shape[2]
    w_h = paddle.fft.fft2(x=w[:, 1:-1], axes=[2, 3])
    k_max = nx // 2
    N = nx
    k_x = (
        paddle.concat(
            x=(
                paddle.arange(start=0, end=k_max, step=1),
                paddle.arange(start=-k_max, end=0, step=1),
            ),
            axis=0,
        )
        .reshape([N, 1])
        .tile(repeat_times=[1, N])
        .reshape([1, 1, N, N])
    )
    k_y = (
        paddle.concat(
            x=(
                paddle.arange(start=0, end=k_max, step=1),
                paddle.arange(start=-k_max, end=0, step=1),
            ),
            axis=0,
        )
        .reshape([1, N])
        .tile(repeat_times=[N, 1])
        .reshape([1, 1, N, N])
    )

    lap = k_x**2 + k_y**2
    lap = float_to_complex(lap)
    k_x = float_to_complex(k_x)
    k_y = float_to_complex(k_y)
    lap[..., 0, 0] = 1.0
    psi_h = w_h / lap
    u_h = 1.0j * k_y * psi_h
    v_h = -1.0j * k_x * psi_h
    wx_h = 1.0j * k_x * w_h
    wy_h = 1.0j * k_y * w_h
    wlap_h = -lap * w_h
    u = paddle.fft.irfft2(x=u_h[(...), :, : k_max + 1], axes=[2, 3])
    v = paddle.fft.irfft2(x=v_h[(...), :, : k_max + 1], axes=[2, 3])
    wx = paddle.fft.irfft2(x=wx_h[(...), :, : k_max + 1], axes=[2, 3])
    wy = paddle.fft.irfft2(x=wy_h[(...), :, : k_max + 1], axes=[2, 3])
    wlap = paddle.fft.irfft2(x=wlap_h[(...), :, : k_max + 1], axes=[2, 3])
    advection = u * wx + v * wy
    wt = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * dt)
    x = paddle.linspace(start=0, stop=2 * np.pi, num=nx + 1)
    x = x[0:-1]
    X, Y = paddle.meshgrid(x, x)
    f = -4 * paddle.cos(x=4 * Y)
    residual = wt + (advection - 1.0 / re * wlap + 0.1 * w[:, 1:-1]) - f
    residual_loss = (residual**2).mean()
    dw = paddle.grad(outputs=residual_loss, inputs=w)[0]
    return dw


def get_beta_schedule_train(
    beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps
):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, paddle.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if not param.stop_gradient:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, paddle.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if not param.stop_gradient:
                self.shadow[name].data = (
                    1.0 - self.mu
                ) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, paddle.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if not param.stop_gradient:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, paddle.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(
                inner_module.config.device
            )
            module_copy.set_state_dict(state_dict=inner_module.state_dict())
            module_copy = paddle.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.set_state_dict(state_dict=module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
