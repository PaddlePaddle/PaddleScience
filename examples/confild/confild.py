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


# build data
def getdata(cfg):
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
    in_normalizer.fit_normalize(
        coord if cfg.Latent.lumped else coord.flatten(0, cfg.Latent.dims - 1)
    )
    out_normalizer = Normalizer_ts(**cfg.Data.normalizer)
    out_normalizer.fit_normalize(
        fois if cfg.Latent.lumped else fois.flatten(0, cfg.Latent.dims)
    )
    normed_coords = in_normalizer.normalize(coord)# 训练集就是测试集
    normed_fois = out_normalizer.normalize(fois)

    return normed_coords, normed_fois, N_samples, spatio_axis, out_normalizer


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

    for epoch in range(cfg.TRAIN.epochs):
        cnf_model.train()
        latents_model.train()
        train_loss = []
        for batch_coords, batch_fois, idx in train_loader:
            idx = {"latent_x": idx}
            batch_latent = latents_model(idx)
            if isinstance(batch_coords, list):
                batch_coords = [coord for coord in batch_coords]
            data = {
                "confild_x": batch_coords,
                "latent_z": batch_latent["latent_z"],
            }
            batch_output = cnf_model(data)
            loss = criterion(batch_output["confild_output"], batch_fois)
            
            # 清空梯度
            cnf_optimizer.clear_grad(set_to_zero=False)
            latents_optimizer.clear_grad(set_to_zero=False)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            cnf_optimizer.step()
            latents_optimizer.step()
            train_loss.append(loss)
        epoch_loss = paddle.stack(x=train_loss).mean().item()
        losses.append(epoch_loss)
        print("epoch {}, train loss {}".format(epoch + 1, epoch_loss))
        if epoch % 100 == 0:
            test_error = []
            cnf_model.eval()
            latents_model.eval()
            with paddle.no_grad():
                for test_coords, test_fois, idx in test_loader:
                    if isinstance(test_coords, list):
                        test_coords = [coord for coord in test_coords]
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
            paddle.save(cnf_model.state_dict(), f"cnf_model_{epoch}.pdparams")
            paddle.save(latents_model.state_dict(), f"latents_model_{epoch}.pdparams")
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

    for epoch in range(cfg.TRAIN.epochs):
        cnf_model.train()
        latents_model.train()
        train_loss = []
        for batch_coords, batch_fois, idx in train_loader:
            idx = {"latent_x": idx}
            batch_latent = latents_model(idx)
            if isinstance(batch_coords, list):
                batch_coords = [coord for coord in batch_coords]
            data = {
                "confild_x": batch_coords,
                "latent_z": batch_latent["latent_z"],
            }
            batch_output = cnf_model(data)
            loss = criterion(batch_output["confild_output"], batch_fois)
            
            # 清空梯度
            cnf_optimizer.clear_grad(set_to_zero=False)
            latents_optimizer.clear_grad(set_to_zero=False)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            cnf_optimizer.step()
            latents_optimizer.step()
            train_loss.append(loss)
        epoch_loss = paddle.stack(x=train_loss).mean().item()
        losses.append(epoch_loss)
        print("epoch {}, train loss {}".format(epoch + 1, epoch_loss))
        if epoch % 100 == 0:
            test_error = []
            cnf_model.eval()
            latents_model.eval()
            with paddle.no_grad():
                for test_coords, test_fois, idx in test_loader:
                    if isinstance(test_coords, list):
                        test_coords = [coord for coord in test_coords]
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
            paddle.save(cnf_model.state_dict(), f"cnf_model_{epoch}.pdparams")
            paddle.save(latents_model.state_dict(), f"latents_model_{epoch}.pdparams")
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
    # 获取GPU数量，检查是否是多卡训练
    world_size = cfg.TRAIN.mutil_GPU
    # 获取数据
    normed_coords, normed_fois, N_samples, spatio_axis, out_normalizer = getdata(cfg)
    train_normed_fois = normed_fois
    test_normed_fois = normed_fois
    train_indices = list(range(N_samples))
    test_indices = list(range(N_samples))
    
    
    if world_size > 1:
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
        np.arange(normed_fois.shape[0]), dtype="int64"
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
    idxs = np.arange(fois_len)
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
        inference(cfg)
    elif cfg.mode == "export":
        export(cfg)
    else:
        raise ValueError(
            f"cfg.mode should in ['train', 'eval', 'infer', 'export'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()
