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

import hydra
import numpy as np
import glob
import re
import paddle
from omegaconf import DictConfig
from packaging import version

import ppsci
from ppsci.arch import LatentContainer, SIRENAutodecoder_film
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
            self.params = self.params[0].to(new_data.place), self.params[1].to(
                new_data.place
            )
        return self.fnormalize(new_data, self.params, self.method)

    def denormalize(self, new_data_norm):
        if not new_data_norm.place == self.params[0].place:
            self.params = self.params[0].to(new_data_norm.place), self.params[1].to(
                new_data_norm.place
            )
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
            return (data - params[1].to(data.place)) / (
                params[0].to(data.place) - params[1].to(data.place)
            ) * 2 - 1
        elif method == "01":
            return (data - params[1].to(data.place)) / (
                params[0].to(data.place) - params[1].to(data.place)
            )
        elif method == "ms":
            return (data - params[0].to(data.place)) / params[1].to(data.place)
        elif method == "none":
            return data

    @staticmethod
    def fdenormalize(data_norm, params, method):
        if method == "-11":
            return (data_norm + 1) / 2 * (
                params[0].to(data_norm.place) - params[1].to(data_norm.place)
            ) + params[1].to(data_norm.place)
        elif method == "01":
            return data_norm * (
                params[0].to(data_norm.place) - params[1].to(data_norm.place)
            ) + params[1].to(data_norm.place)
        elif method == "ms":
            return data_norm * params[1].to(data_norm.place) + params[0].to(
                data_norm.place
            )
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
                range(input_data.ndim if isinstance(input_data, np.ndarray) else input_data.dim())
            )[1:-1]

    ###### read data - coordinate ######
    if cfg.Data.coord_path is None:
        coord = [np.linspace(0, 1, i) for i in spatio_shape]
        coord = np.stack(np.meshgrid(*coord, indexing="ij"), axis=-1)
    else:
        coord = np.load(cfg.Data.coord_path)
    
    ###### convert to tensor ######
    input_data = paddle.to_tensor(input_data) if not isinstance(input_data, paddle.Tensor) else input_data
    coord = paddle.to_tensor(coord) if not isinstance(coord, paddle.Tensor) else coord
    N_samples = input_data.shape[0]

    ###### normalizer ######
    in_normalizer = Normalizer_ts(**cfg.Data.normalizer)
    in_normalizer.fit_normalize(coord if cfg.Latent.lumped_latent else coord.flatten(0, cfg.dims-1))
    out_normalizer = Normalizer_ts(**cfg.Data.normalizer)
    out_normalizer.fit_normalize(input_data if cfg.Latent.lumped_latent else input_data.flatten(0, cfg.dims))
    normed_coords = in_normalizer.normalize(coord)
    normed_fois = out_normalizer.normalize(input_data)

    return normed_coords, normed_fois, N_samples, spatio_axis


def signal_train(cfg):
    pass


def mutil_train(cfg):
    pass


def train(cfg):
    if cfg.TRAIN.mutil_GPU > 1:
        mutil_train(cfg)
    else:
        signal_train(cfg)


def evaluate(cfg: DictConfig):
    # set data
    normed_coords, normed_fois, N_samples, spatio_axis = getdata(cfg)
    
    # set model
    confild = SIRENAutodecoder_film(**cfg.CONFILD)
    latent = LatentContainer(N_samples=N_samples, **cfg.Latent)
    ppsci.utils.save_load.load_pretrain(
        confild,
        cfg.EVAL.confild_pretrained_model_path,
    )
    ppsci.utils.save_load.load_pretrain(
        latent,
        cfg.EVAL.latent_pretrained_model_path,
    )


def inference(cfg):
    normed_coords, normed_fois, _, _ = getdata(cfg)
    fois_len = normed_fois.shape[0]
    idxs = np.array([i for i in range(fois_len)])
    from deploy import python_infer

    latent_predictor = python_infer.GeneralPredictor(cfg.INFER.Latent)
    input_dict = {"latent_x": idxs}
    output_dict = latent_predictor.predict(input_dict, cfg.INFER.batch_size)
    cnf_predictor = python_infer.GeneralPredictor(cfg.INFER.Confild)
    input_dict = {
       "cnf": normed_coords, output_dict.keys()[0]: output_dict.values()[0],
    }
    output_dict = cnf_predictor.predict(input_dict, cfg.INFER.batch_size)
    print(output_dict)


def export(cfg):
    # set model
    cnf_model = SIRENAutodecoder_film(**cfg.CONFILD)
    latent_model = LatentContainer(**cfg.Latent)
    # initialize solver
    latnet_solver = ppsci.solver.Solver(
        latent_model,
        pretrained_model_path=cfg.INFER.Latent.pretrained_model_path,
    )
    cnf_solver = ppsci.solver.Solver(
        cnf_model,
        pretrained_model_path=cfg.INFER.Confild.pretrained_model_path,
    )
    # export model
    from paddle.static import InputSpec

    input_spec = [
        {
            key: InputSpec([None], "float32", name=key)
            for key in latent_model.input_keys
        },
    ]
    cnf_input_spec = [
        {
          cnf_model.input_keys[0]: InputSpec([None]+cfg.Data.shape, "float32", name=cnf_model.input_keys[0]),
          cnf_model.input_keys[1]: InputSpec([None], "float32", name=cnf_model.input_keys[1])
        }
    ]
    cnf_solver.export(cnf_input_spec, cfg.INFER.Confild.export_path)
    latnet_solver.export(input_spec, cfg.INFER.Latent.export_path)


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
        raise ValueError(f"cfg.mode should in ['train', 'eval', 'infer', 'export'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
