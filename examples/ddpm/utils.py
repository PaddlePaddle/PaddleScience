import numpy as np
import paddle

class KMFlowTensorDataset(paddle.io.Dataset):

    def __init__(self, data_path, train_ratio=0.9, test=False, stat_path=
        None, max_cache_len=4000):
        np.random.seed(1)
        self.all_data = np.load(data_path)
        print('Data set shape: ', self.all_data.shape)
        idxs = np.arange(self.all_data.shape[0])
        num_of_training_seeds = int(train_ratio * len(idxs))
        self.train_idx_lst = idxs[:num_of_training_seeds]
        self.test_idx_lst = idxs[num_of_training_seeds:]
        self.time_step_lst = np.arange(self.all_data.shape[1] - 2)
        if not test:
            self.idx_lst = self.train_idx_lst[:]
        else:
            self.idx_lst = self.test_idx_lst[:]
        self.cache = {}
        self.max_cache_len = max_cache_len
        if stat_path is not None:
            self.stat_path = stat_path
            self.stat = np.load(stat_path)
        else:
            self.stat = {}
            self.prepare_data()

    def __len__(self):
        return len(self.idx_lst) * len(self.time_step_lst)

    def prepare_data(self):
        self.stat['mean'] = np.mean(self.all_data[self.train_idx_lst[:]].
            reshape(-1, 1))
        self.stat['scale'] = np.std(self.all_data[self.train_idx_lst[:]].
            reshape(-1, 1))
        data_mean = self.stat['mean']
        data_scale = self.stat['scale']
        print(f'Data statistics, mean: {data_mean}, scale: {data_scale}')

    def preprocess_data(self, data):
        s = data.shape[-1]
        data = (data - self.stat['mean']) / self.stat['scale']
        return data.astype(np.float32)

    def save_data_stats(self, out_dir):
        np.savez(out_dir, mean=self.stat['mean'], scale=self.stat['scale'])

    def __getitem__(self, idx):
        seed = self.idx_lst[idx // len(self.time_step_lst)]
        frame_idx = idx % len(self.time_step_lst)
        id = idx
        if id in self.cache.keys():
            return self.cache[id]
        else:
            frame0 = self.preprocess_data(self.all_data[seed, frame_idx])
            frame1 = self.preprocess_data(self.all_data[seed, frame_idx + 1])
            frame2 = self.preprocess_data(self.all_data[seed, frame_idx + 2])
            frame = np.concatenate((frame0[None, ...], frame1[None, ...],
                frame2[None, ...]), axis=0)
            self.cache[id] = frame
            if len(self.cache) > self.max_cache_len:
                keys_list = list(self.cache.keys())
                # self.cache.pop(self.cache.keys()[np.random.choice(len(self.
                #     cache.keys()))])
                self.cache.pop(keys_list[np.random.choice(len(keys_list))])
            return frame


def inverse_data_transform(config, X):
    if hasattr(config, 'image_mean'):
        X = X + config.image_mean.to(X.place)[None, ...]
    if config.data.logit_transform:
        X = paddle.nn.functional.sigmoid(x=X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0
        

import sys
import paddle
import numpy as np

def float_to_complex(x):
    x = paddle.cast(x, dtype='float32')
    img_x = paddle.zeros_like(x)
    x = paddle.complex(x, img_x)
    return x

def voriticity_residual_train(w, re=1000.0, dt=1 / 32):
    batchsize = w.shape[0]
    w = w.clone()
    out_5 = w
    out_5.stop_gradient = not True
    out_5
    nx = w.shape[2]
    ny = w.shape[3]
    device = w.place
    w_h = paddle.fft.fft2(x=w[:, 1:-1], axes=[2, 3])
    k_max = nx // 2
    N = nx
    k_x = paddle.concat(x=(paddle.arange(start=0, end=k_max, step=1),
        paddle.arange(start=-k_max, end=0, step=1)), axis=0).reshape([N, 1]).tile(repeat_times=[1, N]).reshape([1, 1, N, N])
    k_y = paddle.concat(x=(paddle.arange(start=0, end=k_max, step=1),
        paddle.arange(start=-k_max, end=0, step=1)), axis=0).reshape([1, N]).tile(repeat_times=[N, 1]).reshape([1, 1, N, N])
    
    lap = k_x ** 2 + k_y ** 2
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
    u = paddle.fft.irfft2(x=u_h[(...), :, :k_max + 1], axes=[2, 3])
    v = paddle.fft.irfft2(x=v_h[(...), :, :k_max + 1], axes=[2, 3])
    wx = paddle.fft.irfft2(x=wx_h[(...), :, :k_max + 1], axes=[2, 3])
    wy = paddle.fft.irfft2(x=wy_h[(...), :, :k_max + 1], axes=[2, 3])
    wlap = paddle.fft.irfft2(x=wlap_h[(...), :, :k_max + 1], axes=[2, 3])
    advection = u * wx + v * wy
    wt = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * dt)
    x = paddle.linspace(start=0, stop=2 * np.pi, num=nx + 1)
    x = x[0:-1]
    X, Y = paddle.meshgrid(x, x)
    f = -4 * paddle.cos(x=4 * Y)
    residual = wt + (advection - 1.0 / re * wlap + 0.1 * w[:, 1:-1]) - f
    residual_loss = (residual ** 2).mean()
    dw = paddle.grad(outputs=residual_loss, inputs=w)[0]
    return dw


def noise_estimation_loss(model, x0: paddle.Tensor, t: paddle.Tensor, e:
    paddle.Tensor, b: paddle.Tensor, keepdim=False):
    """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>    a = (1 - b).cumprod(dim=0).index_select(axis=0, index=t).view(-1, 1, 1, 1)
    beta_sub = 1 - b
    # 2. 计算累积乘积
    cumprod_beta = paddle.cumprod(beta_sub, axis=0)

    # 3. 根据t + 1的值选择特定的元素
    # 注意：Paddle中使用gather函数来实现类似的索引选择功能
    # 假设t是一个标量，我们需要将其转换为Tensor，如果t已经是Tensor则不需要转换
    if not isinstance(t, paddle.Tensor):
        t = paddle.to_tensor(t, dtype='int64')
    selected = paddle.gather(cumprod_beta, index=t, axis=0)
    a = paddle.reshape(selected, shape=[-1, 1, 1, 1])
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.astype(dtype='float32'))
    if keepdim:
        return (e - output).square().sum(axis=(1, 2, 3))
    else:
        return (e - output).square().sum(axis=(1, 2, 3)).mean(axis=0)


def conditional_noise_estimation_loss(model, x0: paddle.Tensor, t: paddle.
    Tensor, e: paddle.Tensor, b: paddle.Tensor, x_scale, x_offset, keepdim=
    False, p=0.1):
    """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>    a = (1 - b).cumprod(dim=0).index_select(axis=0, index=t).view(-1, 1, 1, 1)
    beta_sub = 1 - b
    # 2. 计算累积乘积
    cumprod_beta = paddle.cumprod(beta_sub, dim=0)

    # 3. 根据t + 1的值选择特定的元素
    # 注意：Paddle中使用gather函数来实现类似的索引选择功能
    # 假设t是一个标量，我们需要将其转换为Tensor，如果t已经是Tensor则不需要转换
    if not isinstance(t, paddle.Tensor):
        t = paddle.to_tensor(t, dtype='int64')
    selected = paddle.gather(cumprod_beta, index=t, axis=0)
    a = paddle.reshape(selected, shape=[-1, 1, 1, 1])
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    flag = np.random.uniform(0, 1)
    if flag < p:
        output = model(x, t.astype(dtype='float32'))
    else:
        dx = voriticity_residual_train(x * x_scale + x_offset) / x_scale
        output = model(x, t.astype(dtype='float32'), dx)
    if keepdim:
        return (e - output).square().sum(axis=(1, 2, 3))
    else:
        return (e - output).square().sum(axis=(1, 2, 3)).mean(axis=0)


loss_registry = {'simple': noise_estimation_loss, 'conditional':
    conditional_noise_estimation_loss}


from paddle.optimizer import Adam, SGD

def get_optimizer(config, parameters):
    # if config.optim.optimizer == 'Adam':
    #     return Adam(learning_rate=config.optim.lr, parameters=parameters, 
    #         weight_decay=config.optim.weight_decay, betas=(config.optim.
    #         beta1, 0.999), amsgrad=config.optim.amsgrad, eps=config.optim.eps)
    if config.optim.optimizer == 'Adam':
        return Adam(learning_rate=config.optim.lr, parameters=parameters,
                    weight_decay=config.optim.weight_decay, beta1=config.optim.beta1,
                    beta2=0.999, epsilon=config.optim.eps, grad_clip=None, name=None,
                    lazy_mode=False)
    elif config.optim.optimizer == 'RMSProp':
        return paddle.optimizer.RMSProp(parameters=parameters,
            learning_rate=config.optim.lr, weight_decay=config.optim.
            weight_decay, epsilon=1e-08, rho=0.99)
    elif config.optim.optimizer == 'SGD':
        return SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(
            config.optim.optimizer))

def get_beta_schedule_train(beta_schedule, *, beta_start, beta_end,
    num_diffusion_timesteps):

    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)
    if beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5,
            num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps,
            dtype=np.float64)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1,
            num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'sigmoid':
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
                self.shadow[name].data = (1.0 - self.mu
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
                inner_module.config.device)
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
