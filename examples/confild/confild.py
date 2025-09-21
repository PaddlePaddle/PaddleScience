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


def rMAE(prediction, target, dims=(1, 2)):
    return paddle.abs(x=prediction - target).mean(axis=dims) / paddle.abs(
        x=target
    ).mean(axis=dims)


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


# build data
def getdata(cfg):
    ###### read data - fois ######
    if cfg.Data.load_data_fn == "load_3d_flow":
        input_data = load_3d_flow(cfg.Data.data_path)
    elif cfg.Data.load_data_fn == "load_elbow_flow":
        input_data = load_elbow_flow(cfg.Data.data_path)
    elif cfg.Data.load_data_fn == "load_channel_flow":
        input_data = load_channel_flow(cfg.Data.data_path)
    elif cfg.Data.load_data_fn == "load_periodic_hill_flow":
        input_data = load_periodic_hill_flow(cfg.Data.data_path)
    else:
        input_data = np.load(cfg.Data.data_path)

    spatio_shape = input_data.shape[1:-1]
    spatio_axis = list(
        range(
            input_data.ndim if isinstance(input_data, np.ndarray) else input_data.dim()
        )
    )[1:-1]

    ###### read data - coordinate ######
    if cfg.Data.coor_path is None:
        coord = [np.linspace(0, 1, i) for i in spatio_shape]
        coord = np.stack(np.meshgrid(*coord, indexing="ij"), axis=-1)
    else:
        coord = np.load(cfg.Data.coor_path)
    coord = coord.astype("float32")
    input_data = input_data.astype("float32")

    ###### convert to tensor ######
    input_data = (
        paddle.to_tensor(input_data)
        if not isinstance(input_data, paddle.Tensor)
        else input_data
    )
    coord = paddle.to_tensor(coord) if not isinstance(coord, paddle.Tensor) else coord
    N_samples = input_data.shape[0]

    ###### normalizer ######
    in_normalizer = Normalizer_ts(**cfg.Data.normalizer)
    in_normalizer.fit_normalize(
        coord if cfg.Latent.lumped else coord.flatten(0, cfg.Latent.dims - 1)
    )
    out_normalizer = Normalizer_ts(**cfg.Data.normalizer)
    out_normalizer.fit_normalize(
        input_data if cfg.Latent.lumped else input_data.flatten(0, cfg.Latent.dims)
    )
    normed_coords = in_normalizer.normalize(coord)
    normed_fois = out_normalizer.normalize(input_data)

    return normed_coords, normed_fois, N_samples, spatio_axis, out_normalizer
    ###### 添加数据集划分 ######
    # split_ratio = cfg.Data.get("split_ratio", 0.8)  # 默认为80%训练集
    # seed = cfg.Data.get("shuffle_seed", 42)         # 随机种子

    # # 生成随机索引并划分
    # np.random.seed(seed)
    # total_samples = N_samples
    # indices = np.random.permutation(total_samples)
    # split_idx = int(total_samples * split_ratio)
    
    # # 划分训练集和测试集
    # train_indices = indices[:split_idx]
    # test_indices = indices[split_idx:]

    # # 根据索引获取训练集和测试集数据
    # train_normed_fois = normed_fois[train_indices]
    # test_normed_fois = normed_fois[test_indices]

    # return (
    #     normed_coords, 
    #     train_normed_fois,  # 训练集数据
    #     test_normed_fois,   # 测试集数据
    #     spatio_axis, 
    #     out_normalizer,
    #     train_indices,      # 训练集索引（用于latent模型）
    #     test_indices        # 测试集索引
    # )


class basic_set(paddle.io.Dataset):
    def __init__(self, fois, coord, global_indices=None, extra_siren_in=None) -> None:
        super().__init__()
        self.fois = fois.numpy()
        self.total_samples = tuple(fois.shape)[0]
        self.coords = coord.numpy()
        # 存储全局索引
        self.global_indices = global_indices if global_indices is not None else np.arange(self.total_samples)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 使用全局索引
        global_idx = self.global_indices[idx]
        if hasattr(self, "extra_in"):
            extra_id = idx % tuple(self.fois.shape)[1]
            idb = idx // tuple(self.fois.shape)[1]
            return (self.coords, self.extra_in[extra_id]), self.fois[idb, extra_id], global_idx
        else:
            return self.coords, self.fois[idx], global_idx


def signal_train(cfg, normed_coords, train_normed_fois, test_normed_fois, spatio_axis, out_normalizer, train_indices, test_indices):
    cnf_model = SIRENAutodecoder_film(**cfg.CONFILD)
    latents_model = LatentContainer(**cfg.Latent)

    # 创建训练集和测试集，传入全局索引
    train_dataset = basic_set(train_normed_fois, normed_coords, train_indices)
    test_dataset = basic_set(test_normed_fois, normed_coords, test_indices)

    criterion = paddle.nn.MSELoss()

    # set loader
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=cfg.TRAIN.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=cfg.TRAIN.test_batch_size, shuffle=False
    )
    # set optimizer
    cnf_optimizer = ppsci.optimizer.Adam(cfg.TRAIN.lr.cnf, weight_decay=0.0)(cnf_model)
    latents_optimizer = ppsci.optimizer.Adam(cfg.TRAIN.lr.latents, weight_decay=0.0)(
        latents_model
    )
    losses = []

    for i in range(cfg.TRAIN.epochs):
        cnf_model.train()
        latents_model.train()
        if i != 0:
            cnf_optimizer.step()
            cnf_optimizer.clear_grad(set_to_zero=False)
        train_loss = []
        for batch_coords, batch_fois, idx in train_loader:
            idx = {"latent_x": idx}
            batch_latent = latents_model(idx)
            if isinstance(batch_coords, list):
                batch_coords = [i for i in batch_coords]
            data = {
                "confild_x": batch_coords,
                "latent_z": batch_latent["latent_z"],
            }
            batch_output = cnf_model(data)
            loss = criterion(batch_output["confild_output"], batch_fois)
            latents_optimizer.clear_grad(set_to_zero=False)
            loss.backward()
            latents_optimizer.step()
            train_loss.append(loss)
        epoch_loss = paddle.stack(x=train_loss).mean().item()
        losses.append(epoch_loss)
        print("epoch {}, train loss {}".format(i + 1, epoch_loss))
        if i % 100 == 0:
            test_error = []
            cnf_model.eval()
            latents_model.eval()
            with paddle.no_grad():
                for test_coords, test_fois, idx in test_loader:
                    if isinstance(test_coords, list):
                        test_coords = [i for i in test_coords]
                    prediction = out_normalizer.denormalize(
                        cnf_model(
                            {
                                "confild_x": test_coords,
                                "latent_z": latents_model({"latent_x": idx})[
                                    "latent_z"
                                ],
                            }
                        )["confild_output"]
                    )
                    target = out_normalizer.denormalize(test_fois)
                    error = rMAE(prediction=prediction, target=target, dims=spatio_axis)
                    test_error.append(error)
                test_error = paddle.concat(x=test_error).mean(axis=0)
                print("test MAE: ", test_error)
        if i % 100 == 0:
            paddle.save(cnf_model.state_dict(), f"cnf_model_{i}.pdparams")
            paddle.save(latents_model.state_dict(), f"latents_model_{i}.pdparams")
    # 绘制损失图
    plt.figure(figsize=(10, 6))
    plt.plot(range(cfg.TRAIN.epochs), losses, label="Training Loss")

    # 添加标题和标签
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epochs")
    plt.xticks(rotation=45)
    plt.ylabel("Loss")

    # 添加图例
    plt.legend()

    # 显示网格线
    plt.grid(True)

    # 保存为 PNG 格式
    plt.savefig("case.png")

    # 显示图形
    plt.show()


def mutil_train(cfg, normed_coords, train_normed_fois, test_normed_fois, spatio_axis, out_normalizer, train_indices, test_indices):
    fleet.init(is_collective=True)
    cnf_model = SIRENAutodecoder_film(**cfg.CONFILD)
    cnf_model = fleet.distributed_model(cnf_model)
    latents_model = LatentContainer(**cfg.Latent)
    latents_model = fleet.distributed_model(latents_model)

    # set optimizer
    cnf_optimizer = ppsci.optimizer.Adam(cfg.TRAIN.lr.cnf, weight_decay=0.0)(cnf_model)
    cnf_optimizer = fleet.distributed_optimizer(cnf_optimizer)
    latents_optimizer = ppsci.optimizer.Adam(cfg.TRAIN.lr.latents, weight_decay=0.0)(
        latents_model
    )
    latents_optimizer = fleet.distributed_optimizer(latents_optimizer)

    # 创建训练集和测试集，传入全局索引
    train_dataset = basic_set(train_normed_fois, normed_coords, train_indices)
    test_dataset = basic_set(test_normed_fois, normed_coords, test_indices)

    train_sampler = DistributedBatchSampler(
        train_dataset, cfg.TRAIN.batch_size, shuffle=True, drop_last=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=cfg.TRAIN.mutil_GPU,
        use_shared_memory=False,
    )
    test_sampler = DistributedBatchSampler(
        test_dataset, cfg.TRAIN.test_batch_size, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        num_workers=cfg.TRAIN.mutil_GPU,
        use_shared_memory=False,
    )

    criterion = paddle.nn.MSELoss()
    losses = []

    for i in range(cfg.TRAIN.epochs):
        cnf_model.train()
        latents_model.train()
        if i != 0:
            cnf_optimizer.step()
            cnf_optimizer.clear_grad(set_to_zero=False)
        train_loss = []
        for batch_coords, batch_fois, idx in train_loader:
            idx = {"latent_x": idx}
            batch_latent = latents_model(idx)
            if isinstance(batch_coords, list):
                batch_coords = [i for i in batch_coords]
            data = {
                "confild_x": batch_coords,
                "latent_z": batch_latent["latent_z"],
            }
            batch_output = cnf_model(data)
            loss = criterion(batch_output["confild_output"], batch_fois)
            latents_optimizer.clear_grad(set_to_zero=False)
            loss.backward()
            latents_optimizer.step()
            train_loss.append(loss)
        epoch_loss = paddle.stack(x=train_loss).mean().item()
        losses.append(epoch_loss)
        print("epoch {}, train loss {}".format(i + 1, epoch_loss))
        if i % 100 == 0:
            test_error = []
            cnf_model.eval()
            latents_model.eval()
            with paddle.no_grad():
                for test_coords, test_fois, idx in test_loader:
                    if isinstance(test_coords, list):
                        test_coords = [i for i in test_coords]
                    prediction = out_normalizer.denormalize(
                        cnf_model(
                            {
                                "confild_x": test_coords,
                                "latent_z": latents_model({"latent_x": idx})[
                                    "latent_z"
                                ],
                            }
                        )["confild_output"]
                    )
                    target = out_normalizer.denormalize(test_fois)
                    error = rMAE(prediction=prediction, target=target, dims=spatio_axis)
                    test_error.append(error)
                test_error = paddle.concat(x=test_error).mean(axis=0)
                print("test MAE: ", test_error)
        if i % 100 == 0:
            paddle.save(cnf_model.state_dict(), f"cnf_model_{i}.pdparams")
            paddle.save(latents_model.state_dict(), f"latents_model_{i}.pdparams")
    # 绘制损失图
    plt.figure(figsize=(10, 6))
    plt.plot(range(cfg.TRAIN.epochs), losses, label="Training Loss")

    # 添加标题和标签
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epochs")
    plt.xticks(rotation=45)
    plt.ylabel("Loss")

    # 添加图例
    plt.legend()

    # 显示网格线
    plt.grid(True)

    # 保存为 PNG 格式
    plt.savefig("case.png")

    # 显示图形
    plt.show()


def train(cfg):
    # 获取分割后的数据集
    # (normed_coords, 
    #  train_normed_fois, 
    #  test_normed_fois, 
    #  spatio_axis, 
    #  out_normalizer,
    #  train_indices,
    #  test_indices) = getdata(cfg)
    normed_coords, normed_fois, N_samples, spatio_axis, out_normalizer = getdata(cfg)
    train_normed_fois = normed_fois
    test_normed_fois = normed_fois
    train_indices = list(range(N_samples))
    test_indices = list(range(N_samples))
    
    if cfg.TRAIN.mutil_GPU > 1:
        import paddle.distributed as dist
        dist.init_parallel_env()
        mutil_train(cfg, normed_coords, train_normed_fois, test_normed_fois, 
                    spatio_axis, out_normalizer, train_indices, test_indices)
    else:
        signal_train(cfg, normed_coords, train_normed_fois, test_normed_fois, 
                     spatio_axis, out_normalizer, train_indices, test_indices)


def evaluate(cfg: DictConfig):
    # set data
    # normed_coords, normed_fois, N_samples, spatio_axis, out_normalizer = getdata(cfg)
    normed_coords, normed_fois, _, spatio_axis, out_normalizer = getdata(cfg)

    if len(normed_coords.shape) + 1 == len(normed_fois.shape):
        normed_coords = paddle.tile(
            normed_coords, [normed_fois.shape[0]] + [1] * len(normed_coords.shape)
        )

    idx = paddle.to_tensor(
        np.array([i for i in range(normed_fois.shape[0])]), dtype="int64"
    )
    # set model
    confild = SIRENAutodecoder_film(**cfg.CONFILD)
    latent = LatentContainer(**cfg.Latent)
    logger.info(
        "Loading pretrained model from {}".format(
            cfg.EVAL.confild_pretrained_model_path
        )
    )
    ppsci.utils.save_load.load_pretrain(
        confild,
        cfg.EVAL.confild_pretrained_model_path,
    )
    logger.info(
        "Loading pretrained model from {}".format(cfg.EVAL.latent_pretrained_model_path)
    )
    ppsci.utils.save_load.load_pretrain(
        latent,
        cfg.EVAL.latent_pretrained_model_path,
    )
    latent_test_pred = latent({"latent_x": idx})
    y_test_pred = []
    for i in range(normed_coords.shape[0]):
        y_test_pred.append(
            confild(
                {
                    "confild_x": normed_coords[i],
                    "latent_z": latent_test_pred["latent_z"][i],
                }
            )["confild_output"].numpy()
        )
    y_test_pred = paddle.to_tensor(np.array(y_test_pred))

    y_test_pred = out_normalizer.denormalize(y_test_pred)
    y_test = out_normalizer.denormalize(normed_fois)

    logger.info("Result is {}".format(y_test.numpy()))


def inference(cfg):
    # 获取分割后的数据集
    normed_coords, normed_fois, _, _, _ = getdata(cfg)
    if len(normed_coords.shape) + 1 == len(normed_fois.shape):
        normed_coords = paddle.tile(
            normed_coords, [normed_fois.shape[0]] + [1] * len(normed_coords.shape)
        )

    fois_len = normed_fois.shape[0]
    idxs = np.array([i for i in range(fois_len)])
    from deploy import python_infer

    latent_predictor = python_infer.GeneralPredictor(cfg.INFER.Latent)
    input_dict = {"latent_x": idxs}
    output_dict = latent_predictor.predict(input_dict, cfg.INFER.batch_size)

    cnf_predictor = python_infer.GeneralPredictor(cfg.INFER.Confild)
    input_dict = {
        "confild_x": normed_coords.numpy(),
        "latent_z": list(output_dict.values())[0],
    }
    output_dict = cnf_predictor.predict(input_dict, cfg.INFER.batch_size)

    logger.info("Result is {}".format(output_dict["fetch_name_0"]))


def uncondiction_infer(cfg):
    test_batch_size = cfg.Uncondiction_INFER.test_batch_size
    time_length = cfg.Uncondiction_INFER.time_length
    latent_length = cfg.Uncondiction_INFER.latent_length
    image_size = cfg.Uncondiction_INFER.image_size
    num_channels = cfg.Uncondiction_INFER.num_channels
    num_res_blocks = cfg.Uncondiction_INFER.num_res_blocks
    num_heads = cfg.Uncondiction_INFER.num_heads
    num_head_channels = cfg.Uncondiction_INFER.num_head_channels
    attention_resolutions = cfg.Uncondiction_INFER.attention_resolutions
    steps = cfg.Uncondiction_INFER.steps
    noise_schedule = cfg.Uncondiction_INFER.noise_schedule

    unet_model = create_model(
        image_size=image_size,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        attention_resolutions=attention_resolutions,
    )
    # ppsci.utils.save_load.load_pretrain(
    #     unet_model,
    #     # cfg.Uncondiction_INFER.ema_path,
    #     "/home/aistudio/ema_0.9999_550000.pdparams",
    # )
    diff_model = create_gaussian_diffusion(steps=steps, noise_schedule=noise_schedule)
    sample_fn = diff_model.p_sample_loop
    gen_latents = sample_fn(unet_model, (test_batch_size, 1, time_length, latent_length))[
        :, 0
    ]
    max_val, min_val = np.load("/home/aistudio/data_max.npy"), np.load("/home/aistudio/data_min.npy")
    # max_val, min_val = np.load(cfg.Uncondiction_INFER.max_val), np.load(cfg.Uncondiction_INFER.min_val)
    max_val, min_val = paddle.to_tensor(data=max_val), paddle.to_tensor(data=min_val)
    gen_latents = (gen_latents + 1) * (max_val - min_val) / 2.0 + min_val
    # 加载cnf模型
    print("加载cnf模型")
    confild = SIRENAutodecoder_film(**cfg.CONFILD)
    ppsci.utils.save_load.load_pretrain(
        confild,
        "https://dataset.bj.bcebos.com/PaddleScience/CoNFiLD/cnf_model_9700.pdparams",# cfg.EVAL.confild_pretrained_model_path,
    )
    confild.eval()
    coord = paddle.to_tensor(np.load("/home/aistudio/data/data321897/case1_coords.npy"), dtype="float32")#(np.load(f"{cfg.Data.coor_path}"), dtype='float32')
    batch_size = 1
    n_samples = tuple(gen_latents.shape)[0]
    out_normalizer = Normalizer_ts(**cfg.Data.normalizer)

    gen_fields = []
    print("开始生成")
    for sample_index in range(n_samples):
        print("第{}个样本", sample_index)
        for i in range(tuple(gen_latents.shape)[1] // batch_size):
            input_dict = {
                "confild_x": coord,
                "latent_z": gen_latents[sample_index, i * batch_size : (i + 1) * batch_size],
            }
            confild_output = confild(input_dict)
            # print(confild_output)
            gen_fields.append(out_normalizer.denormalize(confild_output["confild_output"]).detach()
            .cpu()
            .numpy())
    gen_fields = np.concatenate(gen_fields)
    np.save("./", gen_fields)#cfg.Uncondiction_INFER.save_path


class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


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


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


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
        model_mean_type=ModelMeanType.EPSILON
        if not predict_xstart
        else ModelMeanType.START_X,
        model_var_type=(
            ModelVarType.FIXED_LARGE
            if not sigma_small
            else ModelVarType.FIXED_SMALL
        )
        if not learn_sigma
        else ModelVarType.LEARNED_RANGE,
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


NUM_CLASSES = 1000


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
            channel_mult = 0.5, 1, 1, 2, 2, 4, 4
        elif image_size == 256:
            channel_mult = 1, 1, 2, 2, 4, 4
        elif image_size == 128:
            channel_mult = 1, 1, 2, 3, 4
        elif image_size == 64:
            channel_mult = 1, 2, 3, 4
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
        out_channels=out_channels if not learn_sigma else 2 * out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=NUM_CLASSES if class_cond else None,
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


def export(cfg):
    # set model
    cnf_model = SIRENAutodecoder_film(**cfg.CONFILD)
    latent_model = LatentContainer(**cfg.Latent)
    # initialize solver
    latnet_solver = ppsci.solver.Solver(
        latent_model,
        pretrained_model_path=cfg.INFER.Latent.INFER.pretrained_model_path,
    )
    cnf_solver = ppsci.solver.Solver(
        cnf_model,
        pretrained_model_path=cfg.INFER.Confild.INFER.pretrained_model_path,
    )
    # export model
    from paddle.static import InputSpec

    input_spec = [
        {key: InputSpec([None], "int64", name=key) for key in latent_model.input_keys},
    ]
    cnf_input_spec = [
        {
            cnf_model.input_keys[0]: InputSpec(
                [None] + list(cfg.INFER.Confild.INFER.coord_shape),
                "float32",
                name=cnf_model.input_keys[0],
            ),
            cnf_model.input_keys[1]: InputSpec(
                [None] + list(cfg.INFER.Confild.INFER.latents_shape),
                "float32",
                name=cnf_model.input_keys[1],
            ),
        }
    ]
    cnf_solver.export(cnf_input_spec, cfg.INFER.Confild.INFER.export_path)
    latnet_solver.export(input_spec, cfg.INFER.Latent.INFER.export_path)


@hydra.main(version_base=None, config_path="./conf", config_name="confild_case1.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    elif cfg.mode == "infer":
        if cfg.alis == False:
            inference(cfg)
        else:
            uncondiction_infer(cfg)
    elif cfg.mode == "export":
        export(cfg)
    elif cfg.mode == "uncondition_infer":
        raise ValueError(
            f"cfg.mode should in ['train', 'eval', 'infer', 'export', 'uncondition_infer'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()
