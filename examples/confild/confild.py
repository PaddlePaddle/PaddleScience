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
import paddle
from omegaconf import DictConfig
from packaging import version

from ppsci.utils import logger


class Normalizer(object):
    def __init__(self, params=[], method="-11", dim=None):
        self.params = params
        self.method = method
        self.dim = dim

    def fit_normalize(self, data):
        raise NotImplementedError

    def normalize(self, new_data):
        raise NotImplementedError

    def denormalize(self, new_data_norm):
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError


class Normalizer_ts(Normalizer):
    def fit_normalize(self, data):
        assert type(data) == paddle.Tensor
        if len(self.params) == 0:
            if self.method == "-11" or self.method == "01":
                if self.dim == None:
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
                if self.dim == None:
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


def inference(cfg: DictConfig):
    # log paddlepaddle's version
    if version.Version(paddle.__version__) != version.Version("0.0.0"):
        paddle_version = paddle.__version__
        if version.Version(paddle.__version__) < version.Version("2.6.0"):
            logger.warning(
                f"Detected paddlepaddle version is '{paddle_version}', "
                "currently it is recommended to use release 2.6 or develop version."
            )
    else:
        paddle_version = f"develop({paddle.version.commit[:7]})"

    logger.info(f"Using paddlepaddle {paddle_version}")
    # load Data
    cood_data = paddle.to_tensor(np.load(cfg.coor_path))
    # normalize data
    coord = Normalizer_ts(**cfg.normalizer).normalize(cood_data).numpy()

    input_data = np.load(cfg.data_path)
    if len(tuple(input_data.shape)) > 2:
        latents = input_data[:, None, None]
    else:
        latents = input_data[:, None]

    from deploy.python_infer import pinn_predictor

    predictor = pinn_predictor.PINNPredictor(cfg)

    input_dict = {"coords": coord, "latents": latents}
    output_dict = predictor.predict(input_dict, cfg.batch_size)
    # mapping data to cfg.INFER.output_keys
    output_keys = ["output"]
    output_dict = {
        store_key: Normalizer_ts(**cfg.normalizer).denormalize(paddle.to_tensor(output_dict[infer_key]))
        .numpy()
        .flatten()
        for store_key, infer_key in zip(output_keys, output_dict.keys())
    }


@hydra.main(
    version_base=None, config_path="./conf", config_name="confild_case1.yaml"
)
def main(cfg: DictConfig):
    if cfg.mode == "infer":
        inference(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['infer'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()