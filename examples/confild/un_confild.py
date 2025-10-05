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

import enum
import math
import hydra
import matplotlib.pyplot as plt
import numpy as np
import paddle
from omegaconf import DictConfig
from paddle.distributed import fleet
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler

import ppsci
from ppsci.arch import UNetModel
from ppsci.arch import LatentContainer
from ppsci.arch import SIRENAutodecoder_film
from ppsci.arch import SpacedDiffusion
from ppsci.arch import ModelVarType
from ppsci.arch import ModelMeanType
from ppsci.utils import logger


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


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
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
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def space_timesteps(num_timesteps, section_counts):
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
    return np.load(f"{path}")[1:]


def load_channel_flow(
    path,
    t_start=0,
    t_end=1200,
    t_every=1,
):
    return np.load(f"{path}")[t_start:t_end:t_every]


def load_periodic_hill_flow(path):
    data = np.load(f"{path}")
    return data


def load_3d_flow(path):
    data = np.load(f"{path}")
    return data


class Normalizer_ts(object):
    def __init__(self, params=[], method="-11", dim=None):
        self.params = params
        self.method = method
        self.dim = dim

    def fit_normalize(self, data):
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
        if not new_data.place == self.params[0].place:
            self.params = self.params[0], self.params[1]
        return self.fnormalize(new_data, self.params, self.method)

    def denormalize(self, new_data_norm):
        if not new_data_norm.place == self.params[0].place:
            self.params = self.params[0], self.params[1]
        return self.fdenormalize(new_data_norm, self.params, self.method)

    def get_params(self):
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
    world_size = cfg.multiGPU
    ###### read data - fois ######
    if cfg.Data.load_data_fn == "load_3d_flow":
        fois = load_3d_flow(cfg.Data.data_path)
    elif cfg.Data.load_data_fn == "load_elbow_flow":
        fois = load_elbow_flow(cfg.Data.data_path)
    elif cfg.Data.load_data_fn == "load_channel_flow":
        fois = load_channel_flow(cfg.Data.data_path)
    elif cfg.Data.load_data_fn == "load_periodic_hill_flow":
        fois = load_periodic_hill_flow(cfg.Data.data_path)
    else:
        fois = np.load(cfg.Data.data_path)

    # 计算空间形状和轴
    spatio_shape = fois.shape[1:-1]
    spatio_axis = list(
        range(
            fois.ndim if isinstance(fois, np.ndarray) else fois.dim()
        )
    )[1:-1]

    ###### read data - coordinate ######
    if cfg.Data.coor_path is None:
        coord = [np.linspace(0, 1, i) for i in spatio_shape]
        coord = np.stack(np.meshgrid(*coord, indexing="ij"), axis=-1)
    else:
        coord = np.load(cfg.Data.coor_path)
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
    in_normalizer = Normalizer_ts(**cfg.Data.normalizer)
    out_normalizer = Normalizer_ts(**cfg.Data.normalizer)
    # 使用最新的模型参数
    norm_params = paddle.load(f"{hyper_para.save_path}/normalizer_params.pt")
    in_normalizer.params = norm_params["x_normalizer_params"]
    out_normalizer.params = norm_params["y_normalizer_params"]

    cnf_model = SIRENAutodecoder_film(**cfg.CONFILD)
    
    normed_coords = in_normalizer.normalize(coord)# 训练集就是测试集

    return cnf_model, in_normalizer, out_normalizer, coord


def evaluate(cfg):
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

    max_val, min_val = np.load(cfg.DATA.max_val), np.load(cfg.DATA.min_val)
    max_val, min_val = paddle.to_tensor(max_val), paddle.to_tensor(min_val)
    gen_latents = (gen_latents + 1)*(max_val - min_val)/2. + min_val

    # 获取模型
    nf, in_normalizer, out_normalizer, coord = create_slim(cfg)
    nf.set_state_dict(paddle.load(cfg.CONFILD.ema_path))
    coord = in_normalizer.normalize(coord)

    batch_size = 1 # if you are limited by your GPU Memory, please change the batch_size variable accordingly
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
            out = nf(coord.to(new_latents.device), new_latents)
            out = out_normalizer.denormalize(out)
            gen_fields.append(out.detach().cpu().numpy())

    gen_fields = np.concatenate(gen_fields)

    np.save(inp.save_path, gen_fields)
    
    # 绘制结果


@hydra.main(version_base=None, config_path="./conf", config_name="un_confild_case1.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(
            f"cfg.mode should in ['eval'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    # main()
    # 构建create_model
    my_model = create_model(image_size=128,
                           num_channels=128,
                           num_res_blocks=2,
                           num_heads=4,
                           num_head_channels=64,
                           attention_resolutions="32,16,8")
    #保存参数
    paddle.save(my_model.state_dict(), "my_model.pdparams")