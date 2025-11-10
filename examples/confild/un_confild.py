# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入必要的库
from abc import ABC, abstractmethod
import copy
import enum
import functools
import math
import hydra
import matplotlib.pyplot as plt
import numpy as np
import paddle
import os
from omegaconf import DictConfig
from resample import UniformSampler, LossAwareSampler

from ppsci.arch import UNetModel
from ppsci.arch import SIRENAutodecoder_film
from ppsci.arch import SpacedDiffusion
from ppsci.arch import ModelVarType
from ppsci.arch import ModelMeanType
from ppsci.utils import logger
from ppsci.arch import LossType


def mean_flat(tensor):
    """
    计算张量除批次维度外所有维度的平均值
    
    参数:
        tensor: 输入张量
        
    返回:
        除批次维度外所有维度的平均值
    """
    return tensor.mean(axis=list(range(1, len(tensor.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    计算两个高斯分布之间的KL散度
    
    参数:
        mean1: 第一个高斯分布的均值
        logvar1: 第一个高斯分布的对数方差
        mean2: 第二个高斯分布的均值
        logvar2: 第二个高斯分布的对数方差
        
    返回:
        两个高斯分布之间的KL散度
    """
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + paddle.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * paddle.exp(-logvar2)
    )


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    从一维numpy数组中为一批索引提取值
    
    参数:
        arr: 一维numpy数组
        timesteps: 时间步索引
        broadcast_shape: 广播形状
        
    返回:
        提取并广播后的张量
    """
    # 修复类型转换：先指定dtype再索引
    res = paddle.to_tensor(arr, dtype=timesteps.dtype)[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


# 添加用于存储训练和验证损失的全局变量
train_losses = []  # 存储训练损失
valid_losses = []  # 存储验证损失


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    dims=2,
    out_channels=1,
    channel_mult=None,
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
):
    """
    创建UNet模型
    
    参数:
        image_size: 图像尺寸
        num_channels: 模型通道数
        num_res_blocks: 每个下采样级别的残差块数
        dims: 数据维度(1=1D, 2=2D, 3=3D)
        out_channels: 输出张量的通道数
        channel_mult: 每个级别的通道乘数
        learn_sigma: 是否学习方差
        class_cond: 是否使用类别条件
        use_checkpoint: 是否启用梯度检查点
        attention_resolutions: 应用注意力的下采样率
        num_heads: 注意力头数
        num_head_channels: 每个注意力头的通道数
        num_heads_upsample: 上采样块的注意力头数
        use_scale_shift_norm: 是否使用FiLM-like调节
        dropout: Dropout概率
        resblock_updown: 是否使用残差块进行重采样
        use_fp16: 是否使用float16精度
        use_new_attention_order: 是否使用优化的注意力模式
        
    返回:
        UNet模型实例
    """
    if channel_mult is None:
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        # 修复channel_mult处理逻辑，确保类型正确
        if isinstance(channel_mult, str):
            channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=out_channels,
        model_channels=num_channels,
        out_channels=(out_channels if not learn_sigma else 2*out_channels),#(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(1000 if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        dims=dims
    )


# class LossType(enum.Enum):
#     """
#     损失类型枚举
#     """
#     MSE = enum.auto()  # 使用原始MSE损失(学习方差时使用KL)
#     RESCALED_MSE = (
#         enum.auto()
#     )  # 使用原始MSE损失(学习方差时使用RESCALED_KL)
#     KL = enum.auto()  # 使用变分下界
#     RESCALED_KL = enum.auto()  # 类似KL，但重新缩放以估计完整的VLB

#     def is_vb(self):
#         """
#         判断是否为变分下界损失
#         """
#         return self == LossType.KL or self == LossType.RESCALED_KL


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    获取命名的beta调度
    
    参数:
        schedule_name: 调度名称("linear"或"cosine")
        num_diffusion_timesteps: 扩散步骤数
        
    返回:
        beta值数组
    """
    if schedule_name == "linear":
        # Ho等人的线性调度，扩展为适用于任何数量的扩散步骤
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    基于alpha_bar创建betas
    
    参数:
        num_diffusion_timesteps: 扩散步骤数
        alpha_bar: 累积alpha值函数
        max_beta: beta的最大值
        
    返回:
        beta值数组
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def space_timesteps(num_timesteps, section_counts):
    """
    在基础扩散过程中跳过步骤的时间步空间化
    
    参数:
        num_timesteps: 原始时间步数
        section_counts: 每个部分的时间步数
        
    返回:
        保留的时间步集合
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    """
    创建高斯扩散过程
    
    参数:
        steps: 扩散步骤数
        learn_sigma: 是否学习方差
        sigma_small: 是否使用小方差
        noise_schedule: 噪声调度("linear"或"cosine")
        use_kl: 是否使用KL损失
        predict_xstart: 是否预测初始x
        rescale_timesteps: 是否重新缩放时间步
        rescale_learned_sigmas: 是否重新缩放学习的sigma
        timestep_respacing: 时间步重新间隔
        
    返回:
        高斯扩散过程实例
    """
    betas = get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = LossType.RESCALED_MSE
    else:
        loss_type = LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            ModelMeanType.EPSILON if not predict_xstart else ModelMeanType.START_X
        ),
        model_var_type=(
            (
                ModelVarType.FIXED_LARGE
                if not sigma_small
                else ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def load_elbow_flow(path):
    """
    加载肘管流数据
    
    参数:
        path: 数据文件路径
        
    返回:
        肘管流数据(从索引1开始)
    """
    return np.load(f"{path}")[1:]


def load_channel_flow(
    path,
    t_start=0,
    t_end=1200,
    t_every=1,
):
    """
    加载通道流数据
    
    参数:
        path: 数据文件路径
        t_start: 起始时间步
        t_end: 结束时间步
        t_every: 采样间隔
        
    返回:
        通道流数据
    """
    return np.load(f"{path}")[t_start:t_end:t_every]


def load_periodic_hill_flow(path):
    """
    加载周期性山丘流数据
    
    参数:
        path: 数据文件路径
        
    返回:
        周期性山丘流数据
    """
    data = np.load(f"{path}")
    return data


def load_3d_flow(path):
    """
    加载3D流数据
    
    参数:
        path: 数据文件路径
        
    返回:
        3D流数据
    """
    data = np.load(f"{path}")
    return data


class Normalizer_ts(object):
    """
    时间序列归一化器
    """
    def __init__(self, params=[], method="-11", dim=None):
        """
        初始化归一化器
        
        参数:
            params: 归一化参数
            method: 归一化方法("-11", "01", "ms", "none")
            dim: 归一化维度
        """
        self.params = params
        self.method = method
        self.dim = dim

    def fit_normalize(self, data):
        """
        拟合并归一化数据
        
        参数:
            data: 输入数据
            
        返回:
            归一化后的数据
        """
        assert type(data) == paddle.Tensor
        if len(self.params) == 0:
            if self.method == "-11" or self.method == "01":
                if self.dim is None:
                    self.params = paddle.max(x=data), paddle.min(x=data)
                else:
                    self.params = (
                        paddle.max(keepdim=True, x=data, axis=self.dim),
                        paddle.argmax(keepdim=True, x=data, axis=self.dim),
                    )[0], (
                        paddle.min(keepdim=True, x=data, axis=self.dim),
                        paddle.argmin(keepdim=True, x=data, axis=self.dim),
                    )[
                        0
                    ]
            elif self.method == "ms":
                if self.dim is None:
                    self.params = paddle.mean(x=data, axis=self.dim), paddle.std(
                        x=data, axis=self.dim
                    )
                else:
                    self.params = paddle.mean(
                        x=data, axis=self.dim, keepdim=True
                    ), paddle.std(x=data, axis=self.dim, keepdim=True)
            elif self.method == "none":
                self.params = None
        return self.fnormalize(data, self.params, self.method)

    def normalize(self, new_data):
        """
        归一化新数据
        
        参数:
            new_data: 新数据
            
        返回:
            归一化后的数据
        """
        if not new_data.place == self.params[0].place:
            self.params = self.params[0], self.params[1]
        return self.fnormalize(new_data, self.params, self.method)

    def denormalize(self, new_data_norm):
        """
        反归一化数据
        
        参数:
            new_data_norm: 归一化后的数据
            
        返回:
            反归一化后的数据
        """
        if not new_data_norm.place == self.params[0].place:
            self.params = self.params[0], self.params[1]
        return self.fdenormalize(new_data_norm, self.params, self.method)

    def get_params(self):
        """
        获取归一化参数
        """
        if self.method == "ms":
            print("returning mean and std")
        elif self.method == "01":
            print("returning max and min")
        elif self.method == "-11":
            print("returning max and min")
        elif self.method == "none":
            print("do nothing")
        return self.params

    @staticmethod
    def fnormalize(data, params, method):
        """
        执行归一化
        
        参数:
            data: 输入数据
            params: 归一化参数
            method: 归一化方法
            
        返回:
            归一化后的数据
        """
        if method == "-11":
            return (data - params[1]) / (
                params[0] - params[1]
            ) * 2 - 1
        elif method == "01":
            return (data - params[1]) / (
                params[0] - params[1]
            )
        elif method == "ms":
            return (data - params[0]) / params[1]
        elif method == "none":
            return data

    @staticmethod
    def fdenormalize(data_norm, params, method):
        """
        执行反归一化
        
        参数:
            data_norm: 归一化后的数据
            params: 归一化参数
            method: 归一化方法
            
        返回:
            反归一化后的数据
        """
        if method == "-11":
            return (data_norm + 1) / 2 * (params[0] - params[1]) + params[1]
        elif method == "01":
            return data_norm * (
                params[0] - params[1]
            ) + params[1]
        elif method == "ms":
            return data_norm * params[1] + params[0]
        elif method == "none":
            return data_norm


def create_slim(cfg):
    """
    创建SLIM模型
    
    参数:
        cfg: 配置对象
        
    返回:
        CNF模型、输入归一化器、输出归一化器和坐标
    """
    ###### read data - fois ######
    if cfg.CNF.load_data_fn == "load_3d_flow":
        fois = load_3d_flow(cfg.CNF.data_path)
    elif cfg.CNF.load_data_fn == "load_elbow_flow":
        fois = load_elbow_flow(cfg.CNF.data_path)
    elif cfg.CNF.load_data_fn == "load_channel_flow":
        fois = load_channel_flow(cfg.CNF.data_path)
    elif cfg.CNF.load_data_fn == "load_periodic_hill_flow":
        fois = load_periodic_hill_flow(cfg.CNF.data_path)
    else:
        fois = np.load(cfg.CNF.data_path)

    # 计算空间形状和轴
    spatio_shape = fois.shape[1:-1]

    ###### read data - coordinate ######
    if cfg.CNF.coor_path is None:
        coord = [np.linspace(0, 1, i) for i in spatio_shape]
        coord = np.stack(np.meshgrid(*coord, indexing="ij"), axis=-1)
    else:
        coord = np.load(cfg.CNF.coor_path)
    coord = coord.astype("float32")
    fois = fois.astype("float32")

    ###### convert to tensor ######
    fois = (
        paddle.to_tensor(fois)
        if not isinstance(fois, paddle.Tensor)
        else fois
    )
    coord = paddle.to_tensor(coord) if not isinstance(coord, paddle.Tensor) else coord
    N_samples = fois.shape[0]

    ###### normalizer ######
    in_normalizer = Normalizer_ts(**cfg.CNF.normalizer)
    out_normalizer = Normalizer_ts(**cfg.CNF.normalizer)
    # 使用最新的模型参数
    norm_params = paddle.load(cfg.CNF.normalizer_params_path)
    in_normalizer.params = norm_params["x_normalizer_params"]
    out_normalizer.params = norm_params["y_normalizer_params"]

    cnf_model = SIRENAutodecoder_film(**cfg.CNF.CONFILD)

    return cnf_model, in_normalizer, out_normalizer, coord


def dl_iter(dl):
    """
    数据加载器迭代器
    
    参数:
        dl: 数据加载器
        
    返回:
        无限迭代数据加载器
    """
    while True:
        yield from dl 


def train(cfg):
    """
    训练函数
    
    参数:
        cfg: 配置对象
    """
    # create parameters
    batch_size = cfg.TRAIN.batch_size
    test_batch_size = cfg.TRAIN.test_batch_size
    ema_rate = cfg.TRAIN.ema_rate
    ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )

    lr_anneal_steps = cfg.TRAIN.lr_anneal_steps
    final_lr = cfg.TRAIN.final_lr
    step = 0
    resume_step = 0
    microbatch = cfg.TRAIN.microbatch if cfg.TRAIN.microbatch > 0 else batch_size

    ## Data Preprocessing
    train_data = np.load(cfg.DATA.train_data)
    valid_data = np.load(cfg.DATA.valid_data)
    max_val, min_val = np.max(train_data, keepdims=True), np.min(train_data, keepdims=True)
    norm_train_data = -1 + (train_data - min_val)*2. / (max_val - min_val)
    norm_valid_data = -1 + (valid_data - min_val)*2. / (max_val - min_val)

    norm_train_data = paddle.to_tensor(norm_train_data[:, None, ...])
    norm_valid_data = paddle.to_tensor(norm_valid_data[:, None, ...])

    dl_train = dl_iter(paddle.io.DataLoader(paddle.io.TensorDataset(norm_train_data), batch_size=batch_size, shuffle=True))
    dl_valid = dl_iter(paddle.io.DataLoader(paddle.io.TensorDataset(norm_valid_data), batch_size=test_batch_size, shuffle=True))

    unet_model = create_model(image_size=cfg.UNET.image_size,
                        num_channels= cfg.UNET.num_channels,
                        num_res_blocks= cfg.UNET.num_res_blocks,
                        num_heads=cfg.UNET.num_heads,
                        num_head_channels=cfg.UNET.num_head_channels,
                        attention_resolutions=cfg.UNET.attention_resolutions,
                        channel_mult=cfg.UNET.channel_mult
                        )
    diff_model = create_gaussian_diffusion(steps=cfg.Diff.steps,
                                        noise_schedule=cfg.Diff.noise_schedule
                                        )

    # 初始化AdamW优化器
    opt = paddle.optimizer.AdamW(
        parameters=unet_model.parameters(), learning_rate=cfg.TRAIN.lr, weight_decay=cfg.TRAIN.weight_decay
    )
    
    schedule_sampler = UniformSampler(diff_model)

    # 初始化EMA参数
    ema_params = []
    for _ in range(len(ema_rate)):
        ema_param_dict = {}
        for name, param in unet_model.named_parameters():
            ema_param_dict[name] = copy.deepcopy(param.detach())
        ema_params.append(ema_param_dict)
    
    global train_losses, valid_losses
    train_losses.clear()
    valid_losses.clear()

    valid_interval = 100
    
    while lr_anneal_steps and (step + resume_step < lr_anneal_steps):
        cond = {}
        # 获取训练批次数据
        train_batch = next(dl_train)
        
        # 前向传播
        unet_model.train()
        # 清零梯度
        opt.clear_grad()
        
        for i in range(0, len(train_batch), microbatch):
            # 获取当前微批次数据
            micro = train_batch[i : i + microbatch]
            micro_cond = {
                k: v[i : i + microbatch]
                for k, v in cond.items()
            }

            t, weights = schedule_sampler.sample(len(micro))

            compute_losses = functools.partial(
                diff_model.training_losses,
                unet_model,
                micro,
                t,
                model_kwargs=micro_cond
            )

            # 计算损失
            losses = compute_losses()

            if isinstance(schedule_sampler, LossAwareSampler):
                schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            # 计算加权平均损失
            loss = (losses["loss"] * weights).mean()
            
            num_microbatches = (len(train_batch) + microbatch - 1) // microbatch
            if num_microbatches > 1:
                loss = loss / num_microbatches
            
            if i == 0:
                log_loss_dict(
                    diff_model, t, {k: v * weights for k, v in losses.items() if isinstance(v, paddle.Tensor)}, is_valid=False
                )
            
            # 反向传播（梯度累积）
            loss.backward()

        # 更新参数
        grad_norm, param_norm = _compute_norms(unet_model)
        opt.step()
        
        # 更新EMA参数
        _update_ema(ema_rate, ema_params, unet_model)
        
        # 更新学习率
        _anneal_lr(lr_anneal_steps, step, resume_step, opt, final_lr, cfg.TRAIN.lr)
        
        step += 1
        
        # 定期执行验证（每valid_interval步）
        if step % valid_interval == 0:
            unet_model.eval()
            with paddle.no_grad():
                # 获取验证批次
                valid_batch = next(dl_valid)
                all_valid_losses = []
                
                # 分解成微批次处理
                for i in range(0, len(valid_batch), microbatch):
                    # 获取当前微批次数据
                    micro = valid_batch[i : i + microbatch]
                    micro_cond = {
                        k: v[i : i + microbatch]
                        for k, v in cond.items()
                    }
                    
                    # 采样时间步
                    t, weights = schedule_sampler.sample(len(micro))

                    # 计算验证损失
                    compute_losses = functools.partial(
                        diff_model.training_losses,
                        unet_model,
                        micro,
                        t,
                        model_kwargs=micro_cond,
                        valid=True
                    )
                    losses = compute_losses()

                    # 收集损失
                    if "loss" in losses:
                        all_valid_losses.append((losses["loss"] * weights).mean().item())
                
                # 聚合并记录验证损失
                if len(all_valid_losses) > 0:
                    avg_valid_loss = sum(all_valid_losses) / len(all_valid_losses)
                    valid_losses.append(avg_valid_loss)
                    print(f"Step {step}: Train Loss: {train_losses[-1]:.6f}, Valid Loss: {avg_valid_loss:.6f}")
            
            # 切换回训练模式
            unet_model.train()
    
    # 保存模型
    paddle.save(unet_model.state_dict(), "unet.pdparams")
    
    # 绘制训练和验证损失曲线
    plot_losses()


def plot_losses():
    """
    绘制训练和验证损失曲线
    """
    if len(train_losses) == 0 or len(valid_losses) == 0:
        print("没有足够的数据来绘制损失曲线")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', alpha=0.8)
    plt.plot(valid_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
    print("损失曲线已保存为 loss_curve.png")
    
    # 显示图像
    plt.show()


def _compute_norms(model, grad_scale=1.0):
    """
    计算模型参数和梯度的范数
    
    参数:
        model: 模型
        grad_scale: 梯度缩放因子
        
    返回:
        梯度范数和参数范数
    """
    grad_norm = 0.0
    param_norm = 0.0
    for p in model.parameters():
        with paddle.no_grad():
            param_norm += paddle.norm(p, p=2, dtype=paddle.float32).item() ** 2
            if p.grad is not None:
                grad_norm += paddle.norm(p.grad, p=2, dtype=paddle.float32).item() ** 2
    return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)


def _update_ema(ema_rate, ema_params, source_model):
        """
        更新EMA(指数移动平均)参数
        EMA有助于提高生成质量，减少模型权重噪声
        
        参数:
            ema_rate: EMA衰减率列表
            ema_params: EMA参数字典列表
            source_model: 源模型
        """
        for rate, target_params_dict in zip(ema_rate, ema_params):
            for name, target_param in target_params_dict.items():
                source_param = dict(source_model.named_parameters())[name]
                updated = target_param.detach() * rate + source_param.detach() * (1 - rate)
                target_param.set_value(updated)


def _anneal_lr(lr_anneal_steps, step, resume_step, opt, final_lr, lr):
        """
        学习率退火调整
        根据训练进度线性降低学习率
        
        参数:
            lr_anneal_steps: 学习率退火步数
            step: 当前步数
            resume_step: 恢复步数
            opt: 优化器
            final_lr: 最终学习率
            lr: 初始学习率
        """
        if not lr_anneal_steps:
            return
        frac_done = (step + resume_step) / lr_anneal_steps
        new_lr = final_lr * (frac_done) + lr * (1 - frac_done)
        opt.set_lr(new_lr)


def log_loss_dict(diffusion, ts, losses, is_valid=False, add_to_list=True):
    """
    记录损失字典的日志
    
    参数:
        diffusion: 扩散模型对象
        ts: 时间步张量
        losses: 损失字典
        is_valid: 是否为验证损失
        add_to_list: 是否将损失添加到全局列表中（用于验证时聚合控制）
    """
    for key, values in losses.items():
        # 记录平均损失值
        mean_loss = values.mean().item()
        logger.info(f"{key}: {mean_loss:.6f}")
        
        # ts_numpy = ts.cpu().numpy() if ts.place.is_gpu_place() else ts.numpy()
        # values_numpy = values.detach().cpu().numpy() if values.place.is_gpu_place() else values.detach().numpy()
        # for sub_t, sub_loss in zip(ts_numpy, values_numpy):
        #     quartile = int(4 * sub_t / diffusion.num_timesteps)
        #     logger.info(f"{key}_q{quartile}: {sub_loss:.6f}")
        
        # 记录训练和验证损失到全局列表
        if key == "loss" and add_to_list:
            if is_valid:
                valid_losses.append(mean_loss)
            else:
                train_losses.append(mean_loss)


def evaluate(cfg):
    """
    评估函数
    
    参数:
        cfg: 配置对象
    """
    ## Create model and diffusion
    unet_model = create_model(image_size=cfg.UNET.image_size,
                            num_channels=cfg.UNET.num_channels,
                            num_res_blocks=cfg.UNET.num_res_blocks,
                            num_heads=cfg.UNET.num_heads,
                            num_head_channels=cfg.UNET.num_head_channels,
                            attention_resolutions=cfg.UNET.attention_resolutions
                            )

    unet_model.set_state_dict(paddle.load(cfg.UNET.ema_path))

    diff_model = create_gaussian_diffusion(steps=cfg.Diff.steps,
                                        noise_schedule=cfg.Diff.noise_schedule
                                        )

    sample_fn = diff_model.p_sample_loop
    gen_latents = sample_fn(unet_model, (cfg.EVAL.test_batch_size, 1, cfg.EVAL.time_length, cfg.EVAL.latent_length))[:, 0]

    max_val, min_val = cfg.DATA.max_val, cfg.DATA.min_val#np.load(cfg.DATA.max_val), np.load(cfg.DATA.min_val)
    max_val, min_val = paddle.to_tensor(max_val), paddle.to_tensor(min_val)
    gen_latents = (gen_latents + 1)*(max_val - min_val)/2. + min_val

    # 获取模型
    nf, in_normalizer, out_normalizer, coord = create_slim(cfg)
    nf.set_state_dict(paddle.load(cfg.CNF.model_path))
    coord = in_normalizer.normalize(coord)

    batch_size = 1 
    n_samples = gen_latents.shape[0]
    gen_fields = []

    for sample_index in range(n_samples):
        for i in range(gen_latents.shape[1]//batch_size):
            new_latents = gen_latents[sample_index, i*batch_size:(i+1)*batch_size]
            # coord = in_normalizer.normalize(coord)
            if len(coord.shape) > 2:
                new_latents = new_latents[:, None, None]
            else:
                new_latents = new_latents[:, None]
            input_data = {
                "confild_x": coord,
                "latent_z": new_latents
            }
            out = nf(input_data)["confild_output"]
            out = out_normalizer.denormalize(out)
            gen_fields.append(out.detach().cpu().numpy())

    gen_fields = np.concatenate(gen_fields)

    np.save(cfg.save_path, gen_fields)


@hydra.main(version_base=None, config_path="./conf", config_name="un_confild_case1.yaml")
def main(cfg: DictConfig):
    """
    主函数
    
    参数:
        cfg: 配置对象
    """
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(
            f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()
