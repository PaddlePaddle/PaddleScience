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

from __future__ import annotations

from typing import List
from typing import Literal

import numpy as np
import paddle
from einops import rearrange


def _shuffle_along_axis(a: np.ndarray, axis: int) -> np.ndarray:
    """Shuffle numpy array along the specified axis."""
    a = np.swapaxes(a, axis, 0)
    np.random.shuffle(a)
    a = np.swapaxes(a, 0, axis)
    return a


class TMTDataset(paddle.io.Dataset):
    # Whether support batch indexing for speeding up fetching process.
    batch_index: bool = True

    def __init__(
        self,
        input_keys,
        label_keys,
        data_path: str,
        num_train: int,
        mode: Literal["train", "test"] = "train",
        stage: Literal["fae", "dit"] = "fae",
    ):
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.data_path = data_path
        self.num_train = num_train
        self.mode = mode
        self.stage = stage

        train_outputs, test_outputs, mean, std = self._get_dataset(data_path)

        if self.stage == "fae":
            train_outputs = rearrange(
                train_outputs, "b h w c -> (b c) h w"
            )  # [b4, h, w]
            test_outputs = rearrange(test_outputs, "b h w c -> (b c) h w")  # [b4, h, w]
            self.train_outputs = train_outputs[..., None]  # [b4, h, w, 1]
            self.test_outputs = test_outputs[..., None]  # [b4, h, w, 1]
        else:
            self.train_outputs = train_outputs  # [b, h, w, 4]
            self.test_outputs = test_outputs  # [b, h, w, 4]

    def __getitem__(self, idx: int | List[int]) -> np.ndarray:
        if self.mode == "train":
            sample_or_batch = self.train_outputs[idx]
        else:
            sample_or_batch = self.test_outputs[idx]
        return sample_or_batch

    def _get_dataset(self, data_path):
        data = np.load(data_path, allow_pickle=True).item()

        u = data["x_velocity"]
        v = data["y_velocity"]
        p = data["pressure"]
        # udm = data["udm"]
        sdf = data["sdf"]

        outputs = np.stack([u, v, p, sdf], axis=-1)  # (b, h, w, c)

        # Shuffle dataset
        outputs = _shuffle_along_axis(outputs, axis=0)
        outputs = np.array(outputs, dtype=np.float32)

        train_outputs = outputs[: self.num_train]  # [n_train, h, w, 4]
        test_outputs = outputs[self.num_train :]  # [n_test, h, w, 4]

        # Normalize the data by z-score
        mean = train_outputs.mean(axis=(0, 1, 2))  # [4]
        std = train_outputs.std(axis=(0, 1, 2))  # [4]

        train_outputs = (train_outputs - mean) / std  # [n_train, h, w, 4]
        test_outputs = (test_outputs - mean) / std  # [n_test, h, w, 4]

        return train_outputs, test_outputs, mean, std

    def __len__(self):
        if self.mode == "train":
            return len(self.train_outputs)
        else:
            return len(self.test_outputs)


class BatchParser:
    def __init__(self, input_keys, output_keys, num_queries, h, w, solution):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.num_query_points = num_queries
        self.solution = solution

        x_star = np.linspace(0, 1, h)
        y_star = np.linspace(0, 1, w)
        x_star, y_star = np.meshgrid(x_star, y_star, indexing="ij")

        self.coords = np.hstack(
            [x_star.flatten()[:, None], y_star.flatten()[:, None]]
        ).astype(
            paddle.get_default_dtype()
        )  # (h * w, 2)

    def random_query(self, batch: paddle.Tensor, downsample: int = 1):
        batch_inputs = batch  # [b4, h, w, 1]
        b, h, w, c = batch.shape
        batch_outputs = paddle.reshape(batch, [b, -1, c])  # [b4, hw, 1]

        query_index = np.random.choice(
            batch_outputs.shape[1], size=(self.num_query_points,), replace=False
        )  # [num_query_points]

        batch_coords = self.coords[query_index][None, ...]  # [1, num_query_points, 2]
        batch_outputs = batch_outputs[:, query_index]  # [b4, num_query_points, 1]

        # Downsample the inputs
        if len(self.solution) == 1:
            sol = self.solution[0]
        else:
            sol = np.random.choice(self.solution)

        if sol != 1:
            batch_inputs = batch_inputs[:, ::sol, ::sol]  # [b4, h', w', 1]

        # batch_coords: [1, num_query_points, 2]
        # batch_inputs: [b4, h', w', 1]
        # batch_outputs: [b4, num_query_points, 1]
        return (
            {
                self.input_keys[0]: paddle.to_tensor(batch_coords),
                self.input_keys[1]: batch_inputs,
            },
            {
                self.output_keys[0]: batch_outputs,
            },
            None,
        )

    def random_downsample(
        self, batch: paddle.Tensor, downsample: int = 1
    ) -> paddle.Tensor:
        # Downsample the inputs
        if len(self.solution) == 1:
            sol = self.solution[0]
        else:
            sol = np.random.choice(self.solution)

        if sol != 1:
            batch = batch[:, ::downsample, ::downsample]  # [b, h/r, w/r, 4]

        u, v, p, sdf = batch.split(4, axis=-1)  # [b, h/r, w/r, 1]

        return (
            {
                "u": u,
                "v": v,
                "p": p,
                "sdf": sdf,
            },
            {
                "v_t_err": 0,
            },
            None,
        )
