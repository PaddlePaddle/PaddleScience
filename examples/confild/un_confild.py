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

import copy
import functools
import math

import hydra
import matplotlib.pyplot as plt
import numpy as np
import paddle
from omegaconf import DictConfig
from resample import LossAwareSampler
from resample import UniformSampler

from ppsci.arch import LossType
from ppsci.arch import ModelMeanType
from ppsci.arch import ModelVarType
from ppsci.arch import SIRENAutodecoder_film
from ppsci.arch import SpacedDiffusion
from ppsci.arch import UNetModel
from ppsci.utils import logger


def mean_flat(tensor):
    return tensor.mean(axis=list(range(1, len(tensor.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + paddle.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * paddle.exp(-logvar2)
    )


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = paddle.to_tensor(arr, dtype=timesteps.dtype)[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


train_losses = []
valid_losses = []


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
        if isinstance(channel_mult, str):
            channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=out_channels,
        model_channels=num_channels,
        out_channels=(out_channels if not learn_sigma else 2 * out_channels),
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
        dims=dims,
    )


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
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
            (ModelVarType.FIXED_LARGE if not sigma_small else ModelVarType.FIXED_SMALL)
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
            return (data - params[1]) / (params[0] - params[1]) * 2 - 1
        elif method == "01":
            return (data - params[1]) / (params[0] - params[1])
        elif method == "ms":
            return (data - params[0]) / params[1]
        elif method == "none":
            return data

    @staticmethod
    def fdenormalize(data_norm, params, method):
        if method == "-11":
            return (data_norm + 1) / 2 * (params[0] - params[1]) + params[1]
        elif method == "01":
            return data_norm * (params[0] - params[1]) + params[1]
        elif method == "ms":
            return data_norm * params[1] + params[0]
        elif method == "none":
            return data_norm


def create_slim(cfg):
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
    fois = paddle.to_tensor(fois) if not isinstance(fois, paddle.Tensor) else fois
    coord = paddle.to_tensor(coord) if not isinstance(coord, paddle.Tensor) else coord

    ###### normalizer ######
    in_normalizer = Normalizer_ts(**cfg.CNF.normalizer)
    out_normalizer = Normalizer_ts(**cfg.CNF.normalizer)
    norm_params = paddle.load(cfg.CNF.normalizer_params_path)
    in_normalizer.params = norm_params["x_normalizer_params"]
    out_normalizer.params = norm_params["y_normalizer_params"]

    cnf_model = SIRENAutodecoder_film(**cfg.CNF.CONFILD)

    return cnf_model, in_normalizer, out_normalizer, coord


def dl_iter(dl):
    while True:
        yield from dl


def train(cfg):
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

    train_data = np.load(cfg.DATA.train_data)
    valid_data = np.load(cfg.DATA.valid_data)
    print(
        f"Train data shape: {train_data.shape}, range: [{train_data.min():.3f}, {train_data.max():.3f}]"
    )
    print(
        f"Valid data shape: {valid_data.shape}, range: [{valid_data.min():.3f}, {valid_data.max():.3f}]"
    )

    max_val, min_val = np.max(train_data, keepdims=True), np.min(
        train_data, keepdims=True
    )
    norm_train_data = -1 + (train_data - min_val) * 2.0 / (max_val - min_val)
    norm_valid_data = -1 + (valid_data - min_val) * 2.0 / (max_val - min_val)

    print(
        f"After normalization: train range: [{norm_train_data.min():.3f}, {norm_train_data.max():.3f}]"
    )

    norm_train_data = paddle.to_tensor(norm_train_data[:, None, ...])
    norm_valid_data = paddle.to_tensor(norm_valid_data[:, None, ...])

    dl_train = dl_iter(
        paddle.io.DataLoader(
            paddle.io.TensorDataset(norm_train_data),
            batch_size=batch_size,
            shuffle=True,
        )
    )
    dl_valid = dl_iter(
        paddle.io.DataLoader(
            paddle.io.TensorDataset(norm_valid_data),
            batch_size=test_batch_size,
            shuffle=True,
        )
    )

    unet_model = create_model(
        image_size=cfg.UNET.image_size,
        num_channels=cfg.UNET.num_channels,
        num_res_blocks=cfg.UNET.num_res_blocks,
        num_heads=cfg.UNET.num_heads,
        num_head_channels=cfg.UNET.num_head_channels,
        attention_resolutions=cfg.UNET.attention_resolutions,
        channel_mult=cfg.UNET.channel_mult,
    )
    print(
        f"Model created with {sum(p.numel() for p in unet_model.parameters()):,} parameters"
    )

    diff_model = create_gaussian_diffusion(
        steps=cfg.Diff.steps, noise_schedule=cfg.Diff.noise_schedule
    )
    print(
        f"Diffusion model created with {cfg.Diff.steps} steps, noise schedule: {cfg.Diff.noise_schedule}"
    )

    opt = paddle.optimizer.AdamW(
        parameters=unet_model.parameters(),
        learning_rate=cfg.TRAIN.lr,
        weight_decay=cfg.TRAIN.weight_decay,
    )
    print(
        f"Optimizer initialized with lr={cfg.TRAIN.lr}, weight_decay={cfg.TRAIN.weight_decay}"
    )

    schedule_sampler = UniformSampler(diff_model)

    ema_params = []
    for _ in range(len(ema_rate)):
        ema_param_dict = {}
        for name, param in unet_model.named_parameters():
            ema_param_dict[name] = copy.deepcopy(param.detach())
        ema_params.append(ema_param_dict)

    global train_losses, valid_losses
    train_losses.clear()
    valid_losses.clear()

    valid_interval = 50
    max_steps = cfg.TRAIN.max_steps if hasattr(cfg.TRAIN, "max_steps") else 10000
    print(
        f"Starting training with max_steps={max_steps}, lr_anneal_steps={lr_anneal_steps}"
    )

    while step + resume_step < max_steps:
        cond = {}
        train_batch = next(dl_train)

        unet_model.train()
        opt.clear_grad()

        step_losses = []

        for i in range(0, len(train_batch), microbatch):
            micro = train_batch[i : i + microbatch]
            micro_cond = {k: v[i : i + microbatch] for k, v in cond.items()}

            t, weights = schedule_sampler.sample(len(micro))

            compute_losses = functools.partial(
                diff_model.training_losses,
                unet_model,
                paddle.stack(micro),
                t,
                model_kwargs=micro_cond,
            )
            losses = compute_losses()

            if isinstance(schedule_sampler, LossAwareSampler):
                schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            loss = (losses["loss"] * weights).mean()

            if step == 0 and i == 0:
                print(
                    f"First loss computation - loss: {loss.item():.6f}, losses keys: {list(losses.keys())}"
                )
                if "mse" in losses:
                    print(f"MSE loss: {losses['mse'].mean().item():.6f}")
                if "vb" in losses:
                    print(f"VB loss: {losses['vb'].mean().item():.6f}")

            step_losses.append(loss.item())

            if i == 0:
                log_loss_dict(
                    diff_model,
                    t,
                    {
                        k: v * weights
                        for k, v in losses.items()
                        if isinstance(v, paddle.Tensor)
                    },
                    is_valid=False,
                )

            loss.backward()

        paddle.nn.utils.clip_grad_norm_(unet_model.parameters(), max_norm=1.0)

        grad_norm, param_norm = _compute_norms(unet_model)
        opt.step()

        if step_losses:
            avg_step_loss = sum(step_losses) / len(step_losses)
            train_losses.append(avg_step_loss)

            if step % 50 == 0:
                current_lr = opt.get_lr()
                print(
                    f"Step {step}: Loss={avg_step_loss:.6f}, GradNorm={grad_norm:.6f}, ParamNorm={param_norm:.6f}, LR={current_lr:.2e}"
                )

        _update_ema(ema_rate, ema_params, unet_model)

        if lr_anneal_steps is not None and lr_anneal_steps != 0:
            _anneal_lr(lr_anneal_steps, step, resume_step, opt, final_lr, cfg.TRAIN.lr)

        step += 1

        if step % valid_interval == 0:
            unet_model.eval()
            with paddle.no_grad():
                valid_batch = next(dl_valid)
                all_valid_losses = []

                for i in range(0, len(valid_batch), microbatch):
                    micro = valid_batch[i : i + microbatch]
                    micro_cond = {k: v[i : i + microbatch] for k, v in cond.items()}

                    t, weights = schedule_sampler.sample(len(micro))

                    compute_losses = functools.partial(
                        diff_model.training_losses,
                        unet_model,
                        paddle.stack(micro),
                        t,
                        model_kwargs=micro_cond,
                        valid=True,
                    )
                    losses = compute_losses()

                    if "loss" in losses:
                        all_valid_losses.append(
                            (losses["loss"] * weights).mean().item()
                        )

                if len(all_valid_losses) > 0:
                    avg_valid_loss = sum(all_valid_losses) / len(all_valid_losses)
                    valid_losses.append(avg_valid_loss)
                    print(
                        f"Step {step}: Train Loss: {train_losses[-1]:.6f}, Valid Loss: {avg_valid_loss:.6f}"
                    )

            unet_model.train()

    paddle.save(unet_model.state_dict(), "unet.pdparams")

    plot_losses()


def plot_losses():
    if len(train_losses) == 0 or len(valid_losses) == 0:
        print("没有足够的数据来绘制损失曲线")
        return

    plt.figure(figsize=(10, 6))

    # 绘制训练损失
    plt.plot(train_losses, label="Training Loss", alpha=0.8)

    valid_interval = 100
    valid_steps = [(i + 1) * valid_interval for i in range(len(valid_losses))]
    plt.plot(valid_steps, valid_losses, label="Validation Loss", alpha=0.8, marker="o")

    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图像
    plt.savefig("loss_curve.png", dpi=300, bbox_inches="tight")

    # 显示图像
    plt.show()


def _compute_norms(model, grad_scale=1.0):
    grad_norm = 0.0
    param_norm = 0.0
    for p in model.parameters():
        with paddle.no_grad():
            param_norm += paddle.norm(p, p=2, dtype=paddle.float32).item() ** 2
            if p.grad is not None:
                grad_norm += paddle.norm(p.grad, p=2, dtype=paddle.float32).item() ** 2
    return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)


def _update_ema(ema_rate, ema_params, source_model):
    for rate, target_params_dict in zip(ema_rate, ema_params):
        for name, target_param in target_params_dict.items():
            source_param = dict(source_model.named_parameters())[name]
            updated = target_param.detach() * rate + source_param.detach() * (1 - rate)
            target_param.set_value(updated)


def _anneal_lr(lr_anneal_steps, step, resume_step, opt, final_lr, lr):
    if not lr_anneal_steps:
        return
    frac_done = (step + resume_step) / lr_anneal_steps
    new_lr = final_lr * (frac_done) + lr * (1 - frac_done)
    opt.set_lr(new_lr)


def log_loss_dict(diffusion, ts, losses, is_valid=False, add_to_list=True):
    for key, values in losses.items():
        mean_loss = values.mean().item()
        logger.info(f"{key}: {mean_loss:.6f}")

        if key == "loss" and add_to_list:
            if is_valid:
                valid_losses.append(mean_loss)
            else:
                train_losses.append(mean_loss)


def evaluate(cfg):
    unet_model = create_model(
        image_size=cfg.UNET.image_size,
        num_channels=cfg.UNET.num_channels,
        num_res_blocks=cfg.UNET.num_res_blocks,
        num_heads=cfg.UNET.num_heads,
        num_head_channels=cfg.UNET.num_head_channels,
        attention_resolutions=cfg.UNET.attention_resolutions,
    )

    unet_model.set_state_dict(paddle.load(cfg.UNET.ema_path))

    diff_model = create_gaussian_diffusion(
        steps=cfg.Diff.steps, noise_schedule=cfg.Diff.noise_schedule
    )

    sample_fn = diff_model.p_sample_loop
    gen_latents = sample_fn(
        unet_model,
        (cfg.EVAL.test_batch_size, 1, cfg.EVAL.time_length, cfg.EVAL.latent_length),
    )[:, 0]

    max_val, min_val = (
        cfg.DATA.max_val,
        cfg.DATA.min_val,
    )  # np.load(cfg.DATA.max_val), np.load(cfg.DATA.min_val)
    max_val, min_val = paddle.to_tensor(max_val), paddle.to_tensor(min_val)
    gen_latents = (gen_latents + 1) * (max_val - min_val) / 2.0 + min_val

    nf, in_normalizer, out_normalizer, coord = create_slim(cfg)
    nf.set_state_dict(paddle.load(cfg.CNF.model_path))
    coord = in_normalizer.normalize(coord)

    batch_size = 1
    n_samples = gen_latents.shape[0]
    gen_fields = []

    for sample_index in range(n_samples):
        for i in range(gen_latents.shape[1] // batch_size):
            new_latents = gen_latents[
                sample_index, i * batch_size : (i + 1) * batch_size
            ]
            if len(coord.shape) > 2:
                new_latents = new_latents[:, None, None]
            else:
                new_latents = new_latents[:, None]
            input_data = {"confild_x": coord, "latent_z": new_latents}
            out = nf(input_data)["confild_output"]
            out = out_normalizer.denormalize(out)
            gen_fields.append(out.detach().cpu().numpy())

    gen_fields = np.concatenate(gen_fields)

    np.save(cfg.save_path, gen_fields)


@hydra.main(
    version_base=None, config_path="./conf", config_name="un_confild_case1.yaml"
)
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
