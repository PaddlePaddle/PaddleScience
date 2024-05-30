import paddle


def compute_alpha(beta, t):
    beta = paddle.concat(x=[paddle.zeros(shape=[1]).to("gpu"), beta], axis=0)
    beta_sub = 1 - beta

    # 2. 计算累积乘积
    cumprod_beta = paddle.cumprod(beta_sub, dim=0)

    # 3. 根据t + 1的值选择特定的元素
    # 注意：Paddle中使用gather函数来实现类似的索引选择功能
    # 假设t是一个标量，我们需要将其转换为Tensor，如果t已经是Tensor则不需要转换
    if not isinstance(t, paddle.Tensor):
        t = paddle.to_tensor(t, dtype="int64")
    selected = paddle.gather(cumprod_beta, index=t + 1, axis=0)

    # 4. 改变张量的形状
    # PaddlePaddle使用reshape函数来改变形状
    a = paddle.reshape(selected, shape=[-1, 1, 1, 1])
    #     """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
    # >>>>>>    a = (1 - beta).cumprod(dim=0).index_select(axis=0, index=t + 1).view(-1,
    #         1, 1, 1)
    return a


def ddim_steps(x, seq, model, b, **kwargs):
    n = x.shape[0]
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    dx_func = kwargs.get("dx_func", None)
    clamp_func = kwargs.get("clamp_func", None)
    cache = kwargs.get("cache", False)
    logger = kwargs.get("logger", None)
    if logger is not None:
        logger.update(x=xs[-1])
    for i, j in zip(reversed(seq), reversed(seq_next)):
        with paddle.no_grad():
            t = (paddle.ones(shape=n) * i).to(x.place)
            next_t = (paddle.ones(shape=n) * j).to(x.place)
            at = compute_alpha(b, t.astype(dtype="int64"))
            at_next = compute_alpha(b, next_t.astype(dtype="int64"))
            xt = xs[-1].to("cuda")
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to("cpu"))
            c2 = (1 - at_next).sqrt()
        if dx_func is not None:
            dx = dx_func(xt)
        else:
            dx = 0
        with paddle.no_grad():
            xt_next = at_next.sqrt() * x0_t + c2 * et - dx
            if clamp_func is not None:
                xt_next = clamp_func(xt_next)
            xs.append(xt_next.to("cpu"))
        if logger is not None:
            logger.update(x=xs[-1])
        if not cache:
            xs = xs[-1:]
            x0_preds = x0_preds[-1:]
    return xs, x0_preds


def ddpm_steps(x, seq, model, b, **kwargs):
    n = x.shape[0]
    seq_next = [-1] + list(seq[:-1])
    xs = [x]
    x0_preds = []
    betas = b
    dx_func = kwargs.get("dx_func", None)
    cache = kwargs.get("cache", False)
    clamp_func = kwargs.get("clamp_func", None)
    for i, j in zip(reversed(seq), reversed(seq_next)):
        with paddle.no_grad():
            t = (paddle.ones(shape=n) * i).to(x.place)
            next_t = (paddle.ones(shape=n) * j).to(x.place)
            at = compute_alpha(betas, t.astype(dtype="int64"))
            atm1 = compute_alpha(betas, next_t.astype(dtype="int64"))
            beta_t = 1 - at / atm1
            x = xs[-1].to("cuda")
            output = model(x, t.astype(dtype="float32"))
            e = output
            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = paddle.clip(x=x0_from_e, min=-1, max=1)
            x0_preds.append(x0_from_e.to("cpu"))
            mean_eps = (
                atm1.sqrt() * beta_t * x0_from_e + (1 - beta_t).sqrt() * (1 - atm1) * x
            ) / (1.0 - at)
            mean = mean_eps
            noise = paddle.randn(shape=x.shape, dtype=x.dtype)
            mask = 1 - (t == 0).astype(dtype="float32")
            """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            # >>>>>>            mask = mask.view(-1, 1, 1, 1)
            mask = paddle.reshape(mask, shape=[-1, 1, 1, 1])
            logvar = beta_t.log()
        if dx_func is not None:
            dx = dx_func(x)
        else:
            dx = 0
        with paddle.no_grad():
            sample = mean + mask * paddle.exp(x=0.5 * logvar) * noise - dx
            if clamp_func is not None:
                sample = clamp_func(sample)
            xs.append(sample.to("cpu"))
        if not cache:
            xs = xs[-1:]
            x0_preds = x0_preds[-1:]
    return xs, x0_preds


def guided_ddpm_steps(x, seq, model, b, **kwargs):
    n = x.shape[0]
    seq_next = [-1] + list(seq[:-1])
    xs = [x]
    x0_preds = []
    betas = b
    dx_func = kwargs.get("dx_func", None)
    if dx_func is None:
        raise ValueError("dx_func is required for guided denoising")
    clamp_func = kwargs.get("clamp_func", None)
    cache = kwargs.get("cache", False)
    w = kwargs.get("w", 3.0)
    for i, j in zip(reversed(seq), reversed(seq_next)):
        with paddle.no_grad():
            t = (paddle.ones(shape=n) * i).to(x.place)
            next_t = (paddle.ones(shape=n) * j).to(x.place)
            at = compute_alpha(betas, t.astype(dtype="int64"))
            atm1 = compute_alpha(betas, next_t.astype(dtype="int64"))
            beta_t = 1 - at / atm1
            x = xs[-1].to("cuda")
        dx = dx_func(x)
        with paddle.no_grad():
            output = (w + 1) * model(x, t.astype(dtype="float32"), dx) - w * model(
                x, t.astype(dtype="float32")
            )
            e = output
            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = paddle.clip(x=x0_from_e, min=-1, max=1)
            x0_preds.append(x0_from_e.to("cpu"))
            mean_eps = (
                atm1.sqrt() * beta_t * x0_from_e + (1 - beta_t).sqrt() * (1 - atm1) * x
            ) / (1.0 - at)
            mean = mean_eps
            noise = paddle.randn(shape=x.shape, dtype=x.dtype)
            mask = 1 - (t == 0).astype(dtype="float32")
            """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            # >>>>>>            mask = mask.view(-1, 1, 1, 1)
            mask = paddle.reshape(mask, shape=[-1, 1, 1, 1])
            logvar = beta_t.log()
        with paddle.no_grad():
            sample = mean + mask * paddle.exp(x=0.5 * logvar) * noise - dx
            if clamp_func is not None:
                sample = clamp_func(sample)
            xs.append(sample.to("cpu"))
        if not cache:
            xs = xs[-1:]
            x0_preds = x0_preds[-1:]
    return xs, x0_preds


def guided_ddim_steps(x, seq, model, b, **kwargs):
    n = x.shape[0]
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    dx_func = kwargs.get("dx_func", None)
    if dx_func is None:
        raise ValueError("dx_func is required for guided denoising")
    clamp_func = kwargs.get("clamp_func", None)
    cache = kwargs.get("cache", False)
    w = kwargs.get("w", 3.0)
    logger = kwargs.get("logger", None)
    if logger is not None:
        xs[-1] = paddle.to_tensor(xs[-1], place=paddle.CUDAPlace(0))  # 将张量转移到 CUDA 设备上
        logger.update(x=xs[-1])
    for i, j in zip(reversed(seq), reversed(seq_next)):
        with paddle.no_grad():
            t = (paddle.ones(shape=n) * i).to("gpu")
            next_t = (paddle.ones(shape=n) * j).to("gpu")
            at = compute_alpha(b, t.astype(dtype="int64"))
            at_next = compute_alpha(b, next_t.astype(dtype="int64"))
            xt = xs[-1].to("gpu")
        dx = dx_func(xt)
        with paddle.no_grad():
            et = (w + 1) * model(xt, t, dx) - w * model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to("cpu"))
            c2 = (1 - at_next).sqrt()
        with paddle.no_grad():
            xt_next = at_next.sqrt() * x0_t + c2 * et - dx
            if clamp_func is not None:
                xt_next = clamp_func(xt_next)
            xs.append(xt_next.to("cpu"))
        if logger is not None:
            logger.update(x=xs[-1])
        if not cache:
            xs = xs[-1:]
            x0_preds = x0_preds[-1:]
    return xs, x0_preds
