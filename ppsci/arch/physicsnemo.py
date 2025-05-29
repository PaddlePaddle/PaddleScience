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
#
# refs: https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/external_aerodynamics/domino


import glob
import math
import os
import re
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import NewType
from typing import Optional
from typing import Union

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import warp as wp
from paddle.amp import GradScaler
from paddle.optimizer.lr import LRScheduler
from scipy.spatial import KDTree

import ppsci
from ppsci.utils import logger

optimizer = NewType("optimizer", paddle.optimizer)
scheduler = NewType("scheduler", LRScheduler)
scaler = NewType("scaler", GradScaler)


def nd_interpolator(coodinates, field, grid):
    """Function to for nd interpolation"""
    interp_func = KDTree(coodinates[0])
    dd, ii = interp_func.query(grid, k=2)

    field_grid = field[ii]
    field_grid = np.float32(np.mean(field_grid, (3)))
    return field_grid


def pad_inp(arr, npoin, pad_value=0.0):
    """Function for padding arrays"""
    arr_pad = pad_value * np.ones(
        (npoin - arr.shape[0], arr.shape[1], arr.shape[2]), dtype=np.float32
    )
    arr_padded = np.concatenate((arr, arr_pad), axis=0)
    return arr_padded


def shuffle_array_without_sampling(arr):
    """Function for shuffline arrays without sampling."""
    idx = np.arange(arr.shape[0])
    np.random.shuffle(idx)
    return arr[idx], idx


def create_directory(filepath):
    """Function to create directories"""
    if not os.path.exists(filepath):
        os.makedirs(filepath)


def calculate_pos_encoding(nx, d=8):
    """Function for calculating positional encoding"""
    vec = []
    for k in range(int(d / 2)):
        vec.append(np.sin(nx / 10000 ** (2 * (k) / d)))
        vec.append(np.cos(nx / 10000 ** (2 * (k) / d)))
    return vec


def combine_dict(old_dict, new_dict):
    """Function to combine dictionaries"""
    for j in old_dict.keys():
        old_dict[j] += new_dict[j]
    return old_dict


def merge(*lists):
    """Function to merge lists"""
    newlist = lists[:]
    for x in lists:
        if x not in newlist:
            newlist.extend(x)
    return newlist


def mean_std_sampling(field, mean, std, tolerance=3.0):
    """Function for mean/std based sampling"""
    idx_all = []
    for v in range(field.shape[-1]):
        fv = field[:, v]
        idx = np.where(
            (fv > mean[v] + tolerance * std[v]) | (fv < mean[v] - tolerance * std[v])
        )
        if len(idx[0]) != 0:
            idx_all += list(idx[0])

    return idx_all


def dict_to_device(state_dict, device):
    """Function to load dictionary to device"""
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k] = v.to(device)
    return new_state_dict


class BallQuery(paddle.autograd.PyLayer):
    """
    Warp based Ball Query.
    """

    @wp.kernel
    def ball_query(
        points1: wp.array(dtype=wp.vec3),
        points2: wp.array(dtype=wp.vec3),
        grid: wp.uint64,
        k: wp.int32,
        radius: wp.float32,
        mapping: wp.array3d(dtype=wp.int32),
        num_neighbors: wp.array2d(dtype=wp.int32),
    ):

        # Get index of point1
        tid = wp.tid()

        # Get position from points1
        pos = points1[tid]

        # particle contact
        neighbors = wp.hash_grid_query(grid, pos, radius)

        # Keep track of the number of neighbors found
        nr_found = wp.int32(0)

        # loop through neighbors to compute density
        for index in neighbors:
            # Check if outside the radius
            pos2 = points2[index]
            if wp.length(pos - pos2) > radius:
                continue

            # Add neighbor to the list
            mapping[0, tid, nr_found] = index

            # Increment the number of neighbors found
            nr_found += 1

            # Break if we have found enough neighbors
            if nr_found == k:
                num_neighbors[0, tid] = k
                break

        # Set the number of neighbors
        num_neighbors[0, tid] = nr_found

    @wp.kernel
    def sparse_ball_query(
        points2: wp.array(dtype=wp.vec3),
        mapping: wp.array3d(dtype=wp.int32),
        num_neighbors: wp.array2d(dtype=wp.int32),
        outputs: wp.array4d(dtype=wp.float32),
    ):
        # Get index of point1
        p1 = wp.tid()

        # Get number of neighbors
        k = num_neighbors[0, p1]

        # Loop through neighbors
        for _k in range(k):
            # Get point2 index
            index = mapping[0, p1, _k]

            # Get position from points2
            pos = points2[index]

            # Set the output
            outputs[0, p1, _k, 0] = pos[0]
            outputs[0, p1, _k, 1] = pos[1]
            outputs[0, p1, _k, 2] = pos[2]

    @staticmethod
    def forward(
        ctx,
        points1,
        points2,
        lengths1,
        lengths2,
        k,
        radius,
        hash_grid,
    ):
        # Only works for batch size 1
        if points1.shape[0] != 1:
            raise AssertionError("nly works for batch size 1")

        # Convert from paddle to warp
        ctx.points1 = wp.from_paddle(
            points1[0], dtype=wp.vec3, requires_grad=points1.stop_gradient
        )
        ctx.points2 = wp.from_paddle(
            points2[0], dtype=wp.vec3, requires_grad=points2.stop_gradient
        )
        ctx.lengths1 = wp.from_paddle(lengths1, dtype=wp.int32, requires_grad=False)
        ctx.lengths2 = wp.from_paddle(lengths2, dtype=wp.int32, requires_grad=False)
        ctx.k = k
        ctx.radius = radius

        # Allocate the mapping and outputs
        mapping = paddle.zeros([1, points1.shape[1], k], dtype=paddle.int32)
        mapping.stop_gradient = False
        ctx.mapping = wp.from_paddle(mapping, dtype=wp.int32, requires_grad=False)
        num_neighbors = paddle.zeros([1, points1.shape[1]], dtype=paddle.int32)
        num_neighbors.stop_gradient = False
        ctx.num_neighbors = wp.from_paddle(
            num_neighbors, dtype=wp.int32, requires_grad=False
        )
        outputs = paddle.zeros([1, points1.shape[1], k, 3], dtype=paddle.float32)
        outputs.stop_gradient = points1.stop_gradient or points2.stop_gradient
        ctx.outputs = wp.from_paddle(outputs, dtype=wp.float32)
        outputs.stop_gradient = points1.stop_gradient or points2.stop_gradient

        # Make grid
        ctx.hash_grid = hash_grid

        # Build the grid
        ctx.hash_grid.build(ctx.points2, radius)

        # Run the kernel to get mapping
        wp.launch(
            BallQuery.ball_query,
            inputs=[
                ctx.points1,
                ctx.points2,
                ctx.hash_grid.id,
                k,
                radius,
            ],
            outputs=[
                ctx.mapping,
                ctx.num_neighbors,
            ],
            dim=[ctx.points1.shape[0]],
        )

        # Run the kernel to get outputs
        wp.launch(
            BallQuery.sparse_ball_query,
            inputs=[
                ctx.points2,
                ctx.mapping,
                ctx.num_neighbors,
            ],
            outputs=[
                ctx.outputs,
            ],
            dim=[ctx.points1.shape[0]],
        )

        return (
            wp.to_paddle(ctx.mapping),
            wp.to_paddle(ctx.num_neighbors),
            wp.to_paddle(ctx.outputs),
        )

    @staticmethod
    def backward(ctx, grad_mapping, grad_num_neighbors, grad_outputs):
        # Map incoming paddle grads to our output variable
        ctx.outputs.grad = wp.from_paddle(grad_outputs, dtype=wp.float32)

        # Run the kernel in adjoint mode
        wp.launch(
            BallQuery.sparse_ball_query,
            inputs=[
                ctx.points2,
                ctx.mapping,
                ctx.num_neighbors,
            ],
            outputs=[
                ctx.outputs,
            ],
            adj_inputs=[ctx.points2.grad, ctx.mapping.grad, ctx.num_neighbors.grad],
            adj_outputs=[
                ctx.outputs.grad,
            ],
            dim=[ctx.points1.shape[0]],
            adjoint=True,
        )

        # Return the gradients
        return (
            wp.to_paddle(ctx.points1.grad).unsqueeze(0),
            wp.to_paddle(ctx.points2.grad).unsqueeze(0),
            None,
            None,
            None,
            None,
            None,
        )


def kaiming_init(layer):
    if isinstance(layer, (nn.layer.conv._ConvNd, nn.Linear)):
        print(f"layer: {layer} ")
        init_kaimingUniform = paddle.nn.initializer.KaimingUniform(
            nonlinearity="leaky_relu", negative_slope=math.sqrt(5)
        )
        init_kaimingUniform(layer.weight)
        if layer.bias is not None:
            fan_in, _ = ppsci.utils.initializer._calculate_fan_in_and_fan_out(
                layer.weight
            )
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init_uniform = paddle.nn.initializer.Uniform(low=-bound, high=bound)
            init_uniform(layer.bias)


def scale_sdf(sdf):
    """Function to scale SDF"""
    return sdf / (0.4 + abs(sdf))


def calculate_gradient(sdf):
    """Function to calculate the gradients of SDF"""
    m, n, o = sdf.shape[2], sdf.shape[3], sdf.shape[4]
    sdf_x = sdf[:, :, 2:m, :, :] - sdf[:, :, 0 : m - 2, :, :]
    sdf_y = sdf[:, :, :, 2:n, :] - sdf[:, :, :, 0 : n - 2, :]
    sdf_z = sdf[:, :, :, :, 2:o] - sdf[:, :, :, :, 0 : o - 2]

    sdf_x = F.pad(x=sdf_x, pad=(0, 0, 0, 0, 0, 1), mode="constant", value=0.0)
    sdf_x = F.pad(x=sdf_x, pad=(0, 0, 0, 0, 1, 0), mode="constant", value=0.0)
    sdf_y = F.pad(x=sdf_y, pad=(0, 0, 0, 1, 0, 0), mode="constant", value=0.0)
    sdf_y = F.pad(x=sdf_y, pad=(0, 0, 1, 0, 0, 0), mode="constant", value=0.0)
    sdf_z = F.pad(x=sdf_z, pad=(0, 1, 0, 0, 0, 0), mode="constant", value=0.0)
    sdf_z = F.pad(x=sdf_z, pad=(1, 0, 0, 0, 0, 0), mode="constant", value=0.0)

    return sdf_x, sdf_y, sdf_z


def binarize_sdf(sdf):
    """Function to calculate the binarize the SDF"""
    sdf = paddle.where(sdf >= 0, 0.0, 1.0).to(dtype=sdf.dtype)
    return sdf


class BallQueryLayer(paddle.nn.Layer):
    """
    Paddle layer for differentiable and accelerated Ball Query
    operation using Warp.
    Args:
        k (int): Number of neighbors.
        radius (float): Radius of influence.
        grid_size (int): Uniform grid resolution
    """

    def __init__(self, k, radius, grid_size=32):
        super().__init__()
        wp.init()
        self.k = k
        self.radius = radius
        self.hash_grid = wp.HashGrid(grid_size, grid_size, grid_size)

    def forward(self, points1, points2, lengths1, lengths2):
        return BallQuery.apply(
            points1,
            points2,
            lengths1,
            lengths2,
            self.k,
            self.radius,
            self.hash_grid,
        )


def _get_checkpoint_filename(
    path: str,
    base_name: str = "checkpoint",
    index: Union[int, None] = None,
    saving: bool = False,
    model_type: str = "mdlus",
) -> str:
    """Gets the file name /path of checkpoint

    This function has three different ways of providing a checkout filename:
    - If supplied an index this will return the checkpoint name using that index.
    - If index is None and saving is false, this will get the checkpoint with the
    largest index (latest save).
    - If index is None and saving is true, it will return the next valid index file name
    which is calculated by indexing the largest checkpoint index found by one.

    Parameters
    ----------
    path : str
        Path to checkpoints
    base_name: str, optional
        Base file name, by default checkpoint
    index : Union[int, None], optional
        Checkpoint index, by default None
    saving : bool, optional
        Get filename for saving a new checkpoint, by default False
    model_type : str
        Model type, by default "mdlus" for Modulus models and "pdparams" for models


    Returns
    -------
    str
        Checkpoint file name
    """
    # Get model parallel rank so all processes in the first model parallel group
    # can save their checkpoint. In the case without model parallelism,
    # model_parallel_rank should be the same as the process rank itself and
    # only rank 0 saves
    model_parallel_rank = 0

    # Input file name
    checkpoint_filename = str(
        Path(path).resolve() / f"{base_name}.{model_parallel_rank}"
    )

    # File extension for Modulus models or PaddlePaddle models
    file_extension = ".pdparams"

    # If epoch is provided load that file
    if index is not None:
        checkpoint_filename = checkpoint_filename + f".{index}"
        checkpoint_filename += file_extension
    # Otherwise try loading the latest epoch or rolling checkpoint
    else:
        file_names = [
            Path(fname).name
            for fname in glob.glob(
                checkpoint_filename + "*" + file_extension, recursive=False
            )
        ]

        if len(file_names) > 0:
            # If checkpoint from a null index save exists load that
            # This is the most likely line to error since it will fail with
            # invalid checkpoint names
            file_idx = [
                int(
                    re.sub(
                        f"^{base_name}.{model_parallel_rank}.|" + file_extension,
                        "",
                        fname,
                    )
                )
                for fname in file_names
            ]
            file_idx.sort()
            # If we are saving index by 1 to get the next free file name
            if saving:
                checkpoint_filename = checkpoint_filename + f".{file_idx[-1]+1}"
            else:
                checkpoint_filename = checkpoint_filename + f".{file_idx[-1]}"
            checkpoint_filename += file_extension
        else:
            checkpoint_filename += ".0" + file_extension

    return checkpoint_filename


def _unique_model_names(
    models: List[paddle.nn.Layer],
) -> Dict[str, paddle.nn.Layer]:
    """Util to clean model names and index if repeat names, will also strip DDP wrappers
    if they exist.

    Parameters
    ----------
    model :  List[paddle.nn.Layer]
        List of models to generate names for

    Returns
    -------
    Dict[str, paddle.nn.Layer]
        Dictionary of model names and respective modules
    """
    # Loop through provided models and set up base names
    model_dict = {}
    for model0 in models:
        if hasattr(model0, "module"):
            # Strip out DDP layer
            model0 = model0.module
        # Base name of model is meta.name unless paddle model
        base_name = model0.__class__.__name__
        # if isinstance(model0, modulus):
        #     base_name = model0.meta.name
        # If we have multiple models of the same name, introduce another index
        if base_name in model_dict:
            model_dict[base_name].append(model0)
        else:
            model_dict[base_name] = [model0]

    # Set up unique model names if needed
    output_dict = {}
    for key, model in model_dict.items():
        if len(model) > 1:
            for i, model0 in enumerate(model):
                output_dict[key + str(i)] = model0
        else:
            output_dict[key] = model[0]

    return output_dict


def save_checkpoint(
    path: str,
    models: Union[paddle.nn.Layer, List[paddle.nn.Layer], None] = None,
    optimizer: Union[optimizer, None] = None,
    scheduler: Union[scheduler, None] = None,
    scaler: Union[scaler, None] = None,
    epoch: Union[int, None] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Training checkpoint saving utility

    This will save a training checkpoint in the provided path following the file naming
    convention "checkpoint.{model parallel id}.{epoch/index}.mdlus". The load checkpoint
    method in Modulus core can then be used to read this file.

    Parameters
    ----------
    path : str
        Path to save the training checkpoint
    models : Union[paddle.nn.Layer, List[paddle.nn.Layer], None], optional
        A single or list of PaddlePaddle models, by default None
    optimizer : Union[optimizer, None], optional
        Optimizer, by default None
    scheduler : Union[scheduler, None], optional
        Learning rate scheduler, by default None
    scaler : Union[scaler, None], optional
        AMP grad scaler. Will attempt to save on in static capture if none provided, by
        default None
    epoch : Union[int, None], optional
        Epoch checkpoint to load. If none this will save the checkpoint in the next
        valid index, by default None
    metadata : Optional[Dict[str, Any]], optional
        Additional metadata to save, by default None
    """
    # Create checkpoint directory if it does not exist
    if not Path(path).is_dir():
        logger.warning(
            f"Output directory {path} does not exist, will " "attempt to create"
        )
        Path(path).mkdir(parents=True, exist_ok=True)

    # == Saving model checkpoint ==
    if models:
        if not isinstance(models, list):
            models = [models]
        models = _unique_model_names(models)
        for name, model in models.items():
            # Get model type
            model_type = "pdparams"

            # Get full file path / name
            file_name = _get_checkpoint_filename(
                path, name, index=epoch, saving=True, model_type=model_type
            )

            # Save state dictionary
            paddle.save(model.state_dict(), file_name)
            logger.info(f"Saved model state dictionary: {file_name}")

    # == Saving training checkpoint ==
    checkpoint_dict = {}
    # Optimizer state dict
    if optimizer:
        checkpoint_dict["optimizer_state_dict"] = optimizer.state_dict()

    # Scheduler state dict
    if scheduler:
        checkpoint_dict["scheduler_state_dict"] = scheduler.state_dict()

    # Scheduler state dict
    if scaler:
        checkpoint_dict["scaler_state_dict"] = scaler.state_dict()
    # Static capture is being used, save its grad scaler
    # if _StaticCapture._amp_scalers:
    #     checkpoint_dict["static_capture_state_dict"] = _StaticCapture.state_dict()

    # Output file name
    output_filename = _get_checkpoint_filename(
        path, index=epoch, saving=True, model_type="pdparams"
    )
    if epoch:
        checkpoint_dict["epoch"] = epoch
    if metadata:
        checkpoint_dict["metadata"] = metadata
    # Save checkpoint to memory
    if bool(checkpoint_dict):
        paddle.save(
            checkpoint_dict,
            output_filename,
        )
        logger.info(f"Saved training checkpoint: {output_filename}")


def load_checkpoint(
    path: str,
    models: Union[paddle.nn.Layer, List[paddle.nn.Layer], None] = None,
    optimizer: Union[optimizer, None] = None,
    scheduler: Union[scheduler, None] = None,
    scaler: Union[scaler, None] = None,
    epoch: Union[int, None] = None,
    metadata_dict: Optional[Dict[str, Any]] = {},
) -> int:
    """Checkpoint loading utility

    This loader is designed to be used with the save checkpoint utility in Modulus
    Launch. Given a path, this method will try to find a checkpoint and load state
    dictionaries into the provided training objects.

    Parameters
    ----------
    path : str
        Path to training checkpoint
    models : Union[paddle.nn.Layer, List[paddle.nn.Layer], None], optional
        A single or list of models, by default None
    optimizer : Union[optimizer, None], optional
        Optimizer, by default None
    scheduler : Union[scheduler, None], optional
        Learning rate scheduler, by default None
    scaler : Union[scaler, None], optional
        AMP grad scaler, by default None
    epoch : Union[int, None], optional
        Epoch checkpoint to load. If none is provided this will attempt to load the
        checkpoint with the largest index, by default None
    metadata_dict: Optional[Dict[str, Any]], optional
        Dictionary to store metadata from the checkpoint, by default None

    Returns
    -------
    int
        Loaded epoch
    """
    # Check if checkpoint directory exists
    if not Path(path).is_dir():
        logger.warning(
            f"Provided checkpoint directory {path} does not exist, skipping load"
        )
        return 0

    # == Loading model checkpoint ==
    if models:
        if not isinstance(models, list):
            models = [models]
        models = _unique_model_names(models)
        for name, model in models.items():
            # Get model type
            model_type = "pdparams"

            # Get full file path / name
            file_name = _get_checkpoint_filename(
                path, name, index=epoch, model_type=model_type
            )
            if not Path(file_name).exists():
                logger.error(
                    f"Could not find valid model file {file_name}, skipping load"
                )
                continue
            # Load state dictionary
            model.set_state_dict(paddle.load(file_name))

            logger.info(f"Loaded model state dictionary {file_name}")

    # == Loading training checkpoint ==
    checkpoint_filename = _get_checkpoint_filename(
        path, index=epoch, model_type="pdparams"
    )
    if not Path(checkpoint_filename).is_file():
        logger.warning("Could not find valid checkpoint file, skipping load")
        return 0

    checkpoint_dict = paddle.load(checkpoint_filename)
    logger.info(f"Loaded checkpoint file {checkpoint_filename}")

    # Optimizer state dict
    if optimizer and "optimizer_state_dict" in checkpoint_dict:
        optimizer.set_state_dict(checkpoint_dict["optimizer_state_dict"])
        logger.info("Loaded optimizer state dictionary")

    # Scheduler state dict
    if scheduler and "scheduler_state_dict" in checkpoint_dict:
        scheduler.set_state_dict(checkpoint_dict["scheduler_state_dict"])
        logger.info("Loaded scheduler state dictionary")

    # Scaler state dict
    if scaler and "scaler_state_dict" in checkpoint_dict:
        scaler.load_state_dict(checkpoint_dict["scaler_state_dict"])
        logger.info("Loaded grad scaler state dictionary")

    epoch = 0
    if "epoch" in checkpoint_dict:
        epoch = checkpoint_dict["epoch"]
    # Update metadata if exists and the dictionary object is provided
    metadata = checkpoint_dict.get("metadata", {})
    for key, value in metadata.items():
        metadata_dict[key] = value

    return epoch


class BQWarp(nn.Layer):
    """Warp based ball-query layer"""

    def __init__(
        self,
        input_features,
        grid_resolution=[256, 96, 64],
        radius=0.25,
        neighbors_in_radius=10,
    ):
        super().__init__()
        self.ball_query_layer = BallQueryLayer(neighbors_in_radius, radius)
        self.grid_resolution = grid_resolution

    def forward(self, x, p_grid, reverse_mapping=True):
        batch_size = x.shape[0]
        nx, ny, nz = (
            self.grid_resolution[0],
            self.grid_resolution[1],
            self.grid_resolution[2],
        )

        p_grid = paddle.reshape(p_grid, (batch_size, nx * ny * nz, 3))
        p1 = nx * ny * nz
        p2 = x.shape[1]

        if reverse_mapping:
            lengths1 = paddle.full((batch_size,), p1, dtype=paddle.int32)
            lengths2 = paddle.full((batch_size,), p2, dtype=paddle.int32)
            mapping, num_neighbors, outputs = self.ball_query_layer(
                p_grid,
                x,
                lengths1,
                lengths2,
            )
        else:
            lengths1 = paddle.full((batch_size,), p2, dtype=paddle.int32)
            lengths2 = paddle.full((batch_size,), p1, dtype=paddle.int32)
            mapping, num_neighbors, outputs = self.ball_query_layer(
                x,
                p_grid,
                lengths1,
                lengths2,
            )

        return mapping, outputs


class GeoConvOut(nn.Layer):
    """Geometry layer to project STLs on grids"""

    def __init__(self, input_features, model_parameters, grid_resolution=[256, 96, 64]):
        super().__init__()
        base_neurons = model_parameters.base_neurons

        self.fc1 = nn.Linear(input_features, base_neurons)
        self.fc2 = nn.Linear(base_neurons, int(base_neurons / 2))
        self.fc3 = nn.Linear(int(base_neurons / 2), model_parameters.base_neurons_out)

        self.grid_resolution = grid_resolution

        self.activation = F.relu

    def forward(self, x, radius=0.025, neighbors_in_radius=10):
        batch_size = x.shape[0]
        nx, ny, nz = (
            self.grid_resolution[0],
            self.grid_resolution[1],
            self.grid_resolution[2],
        )

        mask = abs(x - 0) > 1e-6

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = F.tanh(self.fc3(x))
        mask = mask[:, :, :, 0:1].expand(
            [mask.shape[0], mask.shape[1], mask.shape[2], x.shape[-1]]
        )

        # paddle does not support multiplication with boolean tensors,
        # so we convert the mask to float
        x = paddle.sum(x * mask.to(dtype=x.dtype), 2)

        x = paddle.reshape(x, (batch_size, x.shape[-1], nx, ny, nz))
        return x


class GeoProcessor(nn.Layer):
    """Geometry processing layer using CNNs"""

    def __init__(self, input_filters, model_parameters):
        super().__init__()
        base_filters = model_parameters.base_filters
        self.conv1 = nn.Conv3D(
            input_filters, base_filters, kernel_size=3, padding="same"
        )
        self.conv_bn1 = nn.BatchNorm3D(int(base_filters))
        self.conv2 = nn.Conv3D(
            base_filters, 2 * base_filters, kernel_size=3, padding="same"
        )
        self.conv_bn2 = nn.BatchNorm3D(int(2 * base_filters))
        self.conv3 = nn.Conv3D(
            2 * base_filters, 4 * base_filters, kernel_size=3, padding="same"
        )
        self.conv_bn3 = nn.BatchNorm3D(int(4 * base_filters))
        self.conv3_1 = nn.Conv3D(
            4 * base_filters, 4 * base_filters, kernel_size=3, padding="same"
        )
        self.conv4 = nn.Conv3D(
            4 * base_filters, 2 * base_filters, kernel_size=3, padding="same"
        )
        self.conv_bn4 = nn.BatchNorm3D(int(2 * base_filters))
        self.conv5 = nn.Conv3D(
            4 * base_filters, base_filters, kernel_size=3, padding="same"
        )
        self.conv_bn5 = nn.BatchNorm3D(int(base_filters))
        self.conv6 = nn.Conv3D(
            2 * base_filters, input_filters, kernel_size=3, padding="same"
        )
        self.conv_bn6 = nn.BatchNorm3D(int(input_filters))
        self.conv7 = nn.Conv3D(
            2 * input_filters, input_filters, kernel_size=3, padding="same"
        )
        self.conv8 = nn.Conv3D(input_filters, 1, kernel_size=3, padding="same")
        self.avg_pool = paddle.nn.AvgPool3D((2, 2, 2))
        self.max_pool = nn.MaxPool3D(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.activation = F.relu
        self.batch_norm = False

    def forward(self, x):
        # Encoder
        x0 = x
        if self.batch_norm:
            x = self.activation(self.conv_bn1(self.conv1(x)))
        else:
            x = self.activation(self.conv1(x))
        x = self.max_pool(x)
        x1 = x
        if self.batch_norm:
            x = self.activation(self.conv_bn2(self.conv2(x)))
        else:
            x = self.activation((self.conv2(x)))
        x = self.max_pool(x)

        x2 = x
        if self.batch_norm:
            x = self.activation(self.conv_bn3(self.conv2(x)))
        else:
            x = self.activation((self.conv3(x)))
        x = self.max_pool(x)

        # Processor loop
        x = F.relu(self.conv3_1(x))

        # Decoder
        if self.batch_norm:
            x = self.activation(self.conv_bn4(self.conv4(x)))
        else:
            x = self.activation((self.conv4(x)))
        x = self.upsample(x)
        x = paddle.concat((x, x2), axis=1)

        if self.batch_norm:
            x = self.activation(self.conv_bn5(self.conv5(x)))
        else:
            x = self.activation((self.conv5(x)))
        x = self.upsample(x)
        x = paddle.concat((x, x1), axis=1)
        if self.batch_norm:
            x = self.activation(self.conv_bn6(self.conv6(x)))
        else:
            x = self.activation((self.conv6(x)))
        x = self.upsample(x)
        x = paddle.concat((x, x0), axis=1)

        x = self.activation(self.conv7(x))
        x = self.conv8(x)

        return x


class GeometryRep(nn.Layer):
    """Geometry representation from STLs block"""

    def __init__(self, input_features, model_parameters=None):
        super().__init__()
        geometry_rep = model_parameters.geometry_rep

        self.bq_warp_short = BQWarp(
            input_features=input_features,
            grid_resolution=model_parameters.interp_res,
            radius=geometry_rep.geo_conv.radius_short,
        )

        self.bq_warp_long = BQWarp(
            input_features=input_features,
            grid_resolution=model_parameters.interp_res,
            radius=geometry_rep.geo_conv.radius_long,
        )

        self.geo_conv_out = GeoConvOut(
            input_features=input_features,
            model_parameters=geometry_rep.geo_conv,
            grid_resolution=model_parameters.interp_res,
        )

        self.geo_processor_short_range = GeoProcessor(
            input_filters=geometry_rep.geo_conv.base_neurons_out,
            model_parameters=geometry_rep.geo_processor,
        )
        self.geo_processor_long_range = GeoProcessor(
            input_filters=geometry_rep.geo_conv.base_neurons_out,
            model_parameters=geometry_rep.geo_processor,
        )
        self.geo_processor_sdf = GeoProcessor(
            input_filters=6, model_parameters=geometry_rep.geo_processor
        )
        self.activation = F.relu
        self.radius_short = geometry_rep.geo_conv.radius_short
        self.radius_long = geometry_rep.geo_conv.radius_long
        self.hops = geometry_rep.geo_conv.hops

    def forward(self, x, p_grid, sdf):

        # Expand SDF
        sdf = paddle.unsqueeze(sdf, 1)

        # Calculate short-range geoemtry dependency
        mapping, k_short = self.bq_warp_short(x, p_grid)
        x_encoding_short = self.geo_conv_out(k_short)

        # Calculate long-range geometry dependency
        mapping, k_long = self.bq_warp_long(x, p_grid)
        x_encoding_long = self.geo_conv_out(k_long)

        # Scaled sdf to emphasis on surface
        scaled_sdf = scale_sdf(sdf)
        # Binary sdf
        binary_sdf = binarize_sdf(sdf)
        # Gradients of SDF
        sdf_x, sdf_y, sdf_z = calculate_gradient(sdf)

        # Propagate information in the geometry enclosed BBox
        for _ in range(self.hops):
            dx = self.geo_processor_short_range(x_encoding_short) / self.hops
            x_encoding_short = x_encoding_short + dx

        # Propagate information in the computational domain BBox
        for _ in range(self.hops):
            dx = self.geo_processor_long_range(x_encoding_long) / self.hops
            x_encoding_long = x_encoding_long + dx

        # Process SDF and its computed features
        sdf = paddle.concat((sdf, scaled_sdf, binary_sdf, sdf_x, sdf_y, sdf_z), 1)
        sdf_encoding = self.geo_processor_sdf(sdf)

        # Geometry encoding comprised of short-range, long-range and SDF features
        encoding_g = paddle.concat((x_encoding_short, sdf_encoding, x_encoding_long), 1)

        return encoding_g


class NNBasisFunctions(nn.Layer):
    """Basis function layer for point clouds"""

    def __init__(self, input_features, model_parameters=None):
        super(NNBasisFunctions, self).__init__()
        self.input_features = input_features

        base_layer = model_parameters.base_layer
        self.fc1 = nn.Linear(self.input_features, base_layer)
        self.fc2 = nn.Linear(base_layer, int(base_layer))
        self.fc3 = nn.Linear(int(base_layer), int(base_layer))
        self.bn1 = nn.BatchNorm1D(base_layer)
        self.bn2 = nn.BatchNorm1D(int(base_layer))
        self.bn3 = nn.BatchNorm1D(int(base_layer))

        self.activation = F.relu

    def forward(self, x, padded_value=-10):
        facets = x
        facets = self.activation(self.fc1(facets))
        facets = self.activation(self.fc2(facets))
        facets = self.fc3(facets)

        return facets


class ParameterModel(nn.Layer):
    """Layer to encode parameters such as inlet velocity and air density"""

    def __init__(self, input_features, model_parameters=None):
        super(ParameterModel, self).__init__()
        self.input_features = input_features

        base_layer = model_parameters.base_layer
        self.fc1 = nn.Linear(self.input_features, base_layer)
        self.fc2 = nn.Linear(base_layer, int(base_layer))
        self.fc3 = nn.Linear(int(base_layer), int(base_layer))
        self.bn1 = nn.BatchNorm1D(base_layer)
        self.bn2 = nn.BatchNorm1D(int(base_layer))
        self.bn3 = nn.BatchNorm1D(int(base_layer))

        self.activation = F.relu

    def forward(self, x, padded_value=-10):
        params = x
        params = self.activation(self.fc1(params))
        params = self.activation(self.fc2(params))
        params = self.fc3(params)

        return params


class AggregationModel(nn.Layer):
    """Layer to aggregate local geometry encoding with basis functions"""

    def __init__(
        self, input_features, output_features, model_parameters=None, new_change=True
    ):
        super(AggregationModel, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.new_change = new_change
        base_layer = model_parameters.base_layer
        self.fc1 = nn.Linear(self.input_features, base_layer)
        self.fc2 = nn.Linear(base_layer, int(base_layer))
        self.fc3 = nn.Linear(int(base_layer), int(base_layer))
        self.fc4 = nn.Linear(int(base_layer), int(base_layer))
        self.fc5 = nn.Linear(int(base_layer), self.output_features)
        self.bn1 = nn.BatchNorm1D(base_layer)
        self.bn2 = nn.BatchNorm1D(int(base_layer))
        self.bn3 = nn.BatchNorm1D(int(base_layer))
        self.bn4 = nn.BatchNorm1D(int(base_layer))
        self.activation = F.relu

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.activation(self.fc2(out))
        out = self.activation(self.fc3(out))
        out = self.activation(self.fc4(out))

        out = self.fc5(out)

        return out


class DoMINO(nn.Layer):
    """DoMINO model architecture
    Parameters
    ----------
    input_features : int
        Number of point input features
    output_features_vol : int
        Number of output features in volume
    output_features_surf : int
        Number of output features on surface
    model_parameters: dict
        Dictionary of model parameters controlled by config.yaml

    Example
    -------
    >>> from modulus.models.domino.model import DoMINO
    >>> import os
    >>> from hydra import compose, initialize
    >>> from omegaconf import OmegaConf
    >>> cfg = OmegaConf.register_new_resolver("eval", eval)
    >>> with initialize(version_base="1.3", config_path="examples/cfd/external_aerodynamics/domino/src/conf"):
    ...    cfg = compose(config_name="config")
    >>> cfg.model.model_type = "combined"
    >>> model = DoMINO(
    ...         input_features=3,
    ...         output_features_vol=5,
    ...         output_features_surf=4,
    ...         model_parameters=cfg.model
    ...     )

    Warp ...
    >>> bsize = 1
    >>> nx, ny, nz = 128, 64, 48
    >>> num_neigh = 7
    >>> pos_normals_closest_vol = paddle.randn([bsize, 100, 3])
    >>> pos_normals_com_vol = paddle.randn([bsize, 100, 3])
    >>> pos_normals_com_surface = paddle.randn([bsize, 100, 3])
    >>> geom_centers = paddle.randn([bsize, 100, 3])
    >>> grid = paddle.randn([bsize, nx, ny, nz, 3])
    >>> surf_grid = paddle.randn([bsize, nx, ny, nz, 3])
    >>> sdf_grid = paddle.randn([bsize, nx, ny, nz])
    >>> sdf_surf_grid = paddle.randn([bsize, nx, ny, nz])
    >>> sdf_nodes = paddle.randn([bsize, 100, 1])
    >>> surface_coordinates = paddle.randn([bsize, 100, 3])
    >>> surface_neighbors = paddle.randn([bsize, 100, num_neigh, 3])
    >>> surface_normals = paddle.randn([bsize, 100, 3])
    >>> surface_neighbors_normals = paddle.randn([bsize, 100, num_neigh, 3])
    >>> surface_sizes = paddle.randn([bsize, 100, 3])
    >>> surface_neighbors_sizes = paddle.randn([bsize, 100, num_neigh, 3])
    >>> volume_coordinates = paddle.randn([bsize, 100, 3])
    >>> vol_grid_max_min = paddle.randn([bsize, 2, 3])
    >>> surf_grid_max_min = paddle.randn([bsize, 2, 3])
    >>> stream_velocity = paddle.randn([bsize, 1])
    >>> air_density = paddle.randn([bsize, 1])
    >>> input_dict = {
    ...            "pos_volume_closest": pos_normals_closest_vol,
    ...            "pos_volume_center_of_mass": pos_normals_com_vol,
    ...            "pos_surface_center_of_mass": pos_normals_com_surface,
    ...            "geometry_coordinates": geom_centers,
    ...            "grid": grid,
    ...            "surf_grid": surf_grid,
    ...            "sdf_grid": sdf_grid,
    ...            "sdf_surf_grid": sdf_surf_grid,
    ...            "sdf_nodes": sdf_nodes,
    ...            "surface_mesh_centers": surface_coordinates,
    ...            "surface_mesh_neighbors": surface_neighbors,
    ...            "surface_normals": surface_normals,
    ...            "surface_neighbors_normals": surface_neighbors_normals,
    ...            "surface_areas": surface_sizes,
    ...            "surface_neighbors_areas": surface_neighbors_sizes,
    ...            "volume_mesh_centers": volume_coordinates,
    ...            "volume_min_max": vol_grid_max_min,
    ...            "surface_min_max": surf_grid_max_min,
    ...             "stream_velocity": stream_velocity,
    ...             "air_density": air_density,
    ...        }
    >>> output = model(input_dict)
    Module ...
    >>> print(f"{output[0].shape}, {output[1].shape}")
    """

    def __init__(
        self,
        input_features,
        output_features_vol=None,
        output_features_surf=None,
        model_parameters=None,
    ):
        super(DoMINO, self).__init__()
        self.input_features = input_features
        self.output_features_vol = output_features_vol
        self.output_features_surf = output_features_surf

        if self.output_features_vol is None and self.output_features_surf is None:
            raise ValueError("Need to specify number of volume or surface features")

        self.num_variables_vol = output_features_vol
        self.num_variables_surf = output_features_surf
        self.grid_resolution = model_parameters.interp_res
        self.surface_neighbors = model_parameters.surface_neighbors
        self.use_surface_normals = model_parameters.use_surface_normals
        self.use_only_normals = model_parameters.use_only_normals
        self.encode_parameters = model_parameters.encode_parameters
        self.param_scaling_factors = model_parameters.parameter_model.scaling_params

        if self.use_surface_normals:
            if self.use_only_normals:
                input_features_surface = input_features + 3
            else:
                input_features_surface = input_features + 4
        else:
            input_features_surface = input_features

        if self.encode_parameters:
            # Defining the parameter model
            base_layer_p = model_parameters.parameter_model.base_layer
            self.parameter_model = ParameterModel(
                input_features=2, model_parameters=model_parameters.parameter_model
            )
        else:
            base_layer_p = 0

        self.geo_rep = GeometryRep(
            input_features=input_features,
            model_parameters=model_parameters,
        )

        # Basis functions for surface and volume
        base_layer_nn = model_parameters.nn_basis_functions.base_layer
        if self.output_features_surf is not None:
            self.nn_basis_surf = nn.LayerList()
            for _ in range(self.num_variables_surf):
                self.nn_basis_surf.append(
                    NNBasisFunctions(
                        input_features=input_features_surface,
                        model_parameters=model_parameters.nn_basis_functions,
                    )
                )

        if self.output_features_vol is not None:
            self.nn_basis_vol = nn.LayerList()
            for _ in range(self.num_variables_vol):
                self.nn_basis_vol.append(
                    NNBasisFunctions(
                        input_features=input_features,
                        model_parameters=model_parameters.nn_basis_functions,
                    )
                )

        # Positional encoding
        position_encoder_base_neurons = model_parameters.position_encoder.base_neurons
        if self.output_features_vol is not None:
            if model_parameters.positional_encoding:
                inp_pos_vol = 25 if model_parameters.use_sdf_in_basis_func else 12
            else:
                inp_pos_vol = 7 if model_parameters.use_sdf_in_basis_func else 3

            self.fc_p_vol = nn.Linear(inp_pos_vol, position_encoder_base_neurons)

        if self.output_features_surf is not None:
            if model_parameters.positional_encoding:
                inp_pos_surf = 12
            else:
                inp_pos_surf = 3

            self.fc_p_surf = nn.Linear(inp_pos_surf, position_encoder_base_neurons)

        # Positional encoding hidden layers
        self.fc_p1 = nn.Linear(
            position_encoder_base_neurons, position_encoder_base_neurons
        )
        self.fc_p2 = nn.Linear(
            position_encoder_base_neurons, position_encoder_base_neurons
        )

        # BQ for surface and volume
        self.neighbors_in_radius = model_parameters.geometry_local.neighbors_in_radius
        self.radius = model_parameters.geometry_local.radius
        self.bq_warp = BQWarp(
            input_features=input_features,
            grid_resolution=model_parameters.interp_res,
            radius=self.radius,
            neighbors_in_radius=self.neighbors_in_radius,
        )

        base_layer_geo = model_parameters.geometry_local.base_layer
        self.fc_1 = nn.Linear(self.neighbors_in_radius * 3, base_layer_geo)
        self.fc_2 = nn.Linear(base_layer_geo, base_layer_geo)
        self.activation = F.relu

        # Aggregation model
        if self.output_features_surf is not None:
            # Surface
            self.agg_model_surf = nn.LayerList()
            for _ in range(self.num_variables_surf):
                self.agg_model_surf.append(
                    AggregationModel(
                        input_features=position_encoder_base_neurons
                        + base_layer_nn
                        + base_layer_geo
                        + base_layer_p,
                        output_features=1,
                        model_parameters=model_parameters.aggregation_model,
                    )
                )

        if self.output_features_vol is not None:
            # Volume
            self.agg_model_vol = nn.LayerList()
            for _ in range(self.num_variables_vol):
                self.agg_model_vol.append(
                    AggregationModel(
                        input_features=position_encoder_base_neurons
                        + base_layer_nn
                        + base_layer_geo
                        + base_layer_p,
                        output_features=1,
                        model_parameters=model_parameters.aggregation_model,
                    )
                )

        self.apply(kaiming_init)

    def geometry_encoder(self, geo_centers, p_grid, sdf):
        """Function to return local geometry encoding"""
        return self.geo_rep(geo_centers, p_grid, sdf)

    def position_encoder(self, encoding_node, eval_mode="volume"):
        """Function to calculate positional encoding"""
        if eval_mode == "volume":
            x = self.activation(self.fc_p_vol(encoding_node))
        elif eval_mode == "surface":
            x = self.activation(self.fc_p_surf(encoding_node))
        x = self.activation(self.fc_p1(x))
        x = self.fc_p2(x)
        return x

    def geo_encoding_local_surface(self, encoding_g, volume_mesh_centers, p_grid):
        """Function to calculate local geometry encoding from global encoding for surface"""
        batch_size = volume_mesh_centers.shape[0]
        nx, ny, nz = (
            self.grid_resolution[0],
            self.grid_resolution[1],
            self.grid_resolution[2],
        )
        p_grid = paddle.reshape(p_grid, (batch_size, nx * ny * nz, 3))
        mapping, outputs = self.bq_warp(
            volume_mesh_centers, p_grid, reverse_mapping=False
        )
        mapping = mapping.astype(paddle.int64)
        mask = mapping != 0

        geo_encoding = paddle.reshape(encoding_g[:, 0], (batch_size, 1, nx * ny * nz))
        geo_encoding = geo_encoding.expand(
            [batch_size, volume_mesh_centers.shape[1], geo_encoding.shape[2]]
        )
        sdf_encoding = paddle.reshape(encoding_g[:, 1], (batch_size, 1, nx * ny * nz))
        sdf_encoding = sdf_encoding.expand(
            [batch_size, volume_mesh_centers.shape[1], sdf_encoding.shape[2]]
        )
        geo_encoding_long = paddle.reshape(
            encoding_g[:, 2], (batch_size, 1, nx * ny * nz)
        )
        geo_encoding_long = geo_encoding_long.expand(
            [batch_size, volume_mesh_centers.shape[1], geo_encoding_long.shape[2]]
        )

        geo_encoding_sampled = paddle.take_along_axis(
            geo_encoding, axis=2, indices=mapping
        ) * mask.to(dtype=geo_encoding.dtype)
        sdf_encoding_sampled = paddle.take_along_axis(
            sdf_encoding, axis=2, indices=mapping
        ) * mask.to(dtype=geo_encoding.dtype)
        geo_encoding_long_sampled = paddle.take_along_axis(
            geo_encoding_long, axis=2, indices=mapping
        ) * mask.to(dtype=geo_encoding.dtype)

        encoding_g = paddle.concat(
            (geo_encoding_sampled, sdf_encoding_sampled, geo_encoding_long_sampled),
            axis=2,
        )
        encoding_g = self.activation(self.fc_1(encoding_g))
        encoding_g = self.fc_2(encoding_g)

        return encoding_g

    def geo_encoding_local(self, encoding_g, volume_mesh_centers, p_grid):
        """Function to calculate local geometry encoding from global encoding"""
        batch_size = volume_mesh_centers.shape[0]
        nx, ny, nz = (
            self.grid_resolution[0],
            self.grid_resolution[1],
            self.grid_resolution[2],
        )
        p_grid = paddle.reshape(p_grid, (batch_size, nx * ny * nz, 3))
        mapping, outputs = self.bq_warp(
            volume_mesh_centers, p_grid, reverse_mapping=False
        )
        mapping = mapping.astype(paddle.int64)
        mask = mapping != 0

        geo_encoding = paddle.reshape(encoding_g[:, 0], (batch_size, 1, nx * ny * nz))
        geo_encoding = geo_encoding.expand(
            [batch_size, volume_mesh_centers.shape[1], geo_encoding.shape[2]]
        )
        sdf_encoding = paddle.reshape(encoding_g[:, 1], (batch_size, 1, nx * ny * nz))
        sdf_encoding = sdf_encoding.expand(
            [batch_size, volume_mesh_centers.shape[1], sdf_encoding.shape[2]]
        )
        geo_encoding_long = paddle.reshape(
            encoding_g[:, 2], (batch_size, 1, nx * ny * nz)
        )
        geo_encoding_long = geo_encoding_long.expand(
            [batch_size, volume_mesh_centers.shape[1], geo_encoding_long.shape[2]]
        )

        geo_encoding_sampled = paddle.take_along_axis(
            geo_encoding, axis=2, indices=mapping
        ) * mask.to(dtype=geo_encoding.dtype)
        sdf_encoding_sampled = paddle.take_along_axis(
            sdf_encoding, axis=2, indices=mapping
        ) * mask.to(dtype=geo_encoding.dtype)
        geo_encoding_long_sampled = paddle.take_along_axis(
            geo_encoding_long, axis=2, indices=mapping
        ) * mask.to(dtype=geo_encoding.dtype)

        encoding_g = paddle.concat(
            (geo_encoding_sampled, sdf_encoding_sampled, geo_encoding_long_sampled),
            axis=2,
        )
        encoding_g = self.activation(self.fc_1(encoding_g))
        encoding_g = self.fc_2(encoding_g)

        return encoding_g

    def calculate_solution_with_neighbors(
        self,
        surface_mesh_centers,
        encoding_g,
        encoding_node,
        surface_mesh_neighbors,
        surface_normals,
        surface_neighbors_normals,
        surface_areas,
        surface_neighbors_areas,
        inlet_velocity,
        air_density,
    ):
        """Function to approximate solution given the neighborhood information"""
        num_variables = self.num_variables_surf
        nn_basis = self.nn_basis_surf
        agg_model = self.agg_model_surf
        num_sample_points = surface_mesh_neighbors.shape[2] + 1

        if self.encode_parameters:
            inlet_velocity = paddle.unsqueeze(inlet_velocity, 1)
            inlet_velocity = inlet_velocity.expand(
                [
                    inlet_velocity.shape[0],
                    surface_mesh_centers.shape[1],
                    inlet_velocity.shape[2],
                ]
            )
            inlet_velocity = inlet_velocity / self.param_scaling_factors[0]

            air_density = paddle.unsqueeze(air_density, 1)
            air_density = air_density.expand(
                [
                    air_density.shape[0],
                    surface_mesh_centers.shape[1],
                    air_density.shape[2],
                ]
            )
            air_density = air_density / self.param_scaling_factors[1]

            params = paddle.concat((inlet_velocity, air_density), axis=-1)
            param_encoding = self.parameter_model(params)

        if self.use_surface_normals:
            if self.use_only_normals:
                surface_mesh_centers = paddle.concat(
                    (surface_mesh_centers, surface_normals),
                    axis=-1,
                )
                surface_mesh_neighbors = paddle.concat(
                    (
                        surface_mesh_neighbors,
                        surface_neighbors_normals,
                    ),
                    axis=-1,
                )

            else:
                surface_mesh_centers = paddle.concat(
                    (surface_mesh_centers, surface_normals, 10**5 * surface_areas),
                    axis=-1,
                )
                surface_mesh_neighbors = paddle.concat(
                    (
                        surface_mesh_neighbors,
                        surface_neighbors_normals,
                        10**5 * surface_neighbors_areas,
                    ),
                    axis=-1,
                )

        for f in range(num_variables):
            for p in range(num_sample_points):
                if p == 0:
                    volume_m_c = surface_mesh_centers
                else:
                    volume_m_c = surface_mesh_neighbors[:, :, p - 1]
                    noise = surface_mesh_centers - volume_m_c
                    dist = paddle.sqrt(
                        noise[:, :, 0:1] ** 2.0
                        + noise[:, :, 1:2] ** 2.0
                        + noise[:, :, 2:3] ** 2.0
                    )
                basis_f = nn_basis[f](volume_m_c)
                output = paddle.concat((basis_f, encoding_node, encoding_g), axis=-1)
                if self.encode_parameters:
                    output = paddle.concat((output, param_encoding), axis=-1)
                if p == 0:
                    output_center = agg_model[f](output)
                else:
                    if p == 1:
                        output_neighbor = agg_model[f](output) * (1.0 / dist)
                        dist_sum = 1.0 / dist
                    else:
                        output_neighbor += agg_model[f](output) * (1.0 / dist)
                        dist_sum += 1.0 / dist
            if num_sample_points > 1:
                output_res = 0.5 * output_center + 0.5 * output_neighbor / dist_sum
            else:
                output_res = output_center
            if f == 0:
                output_all = output_res
            else:
                output_all = paddle.concat((output_all, output_res), axis=-1)

        return output_all

    def calculate_solution(
        self,
        volume_mesh_centers,
        encoding_g,
        encoding_node,
        inlet_velocity,
        air_density,
        eval_mode,
        num_sample_points=20,
        noise_intensity=50,
    ):
        """Function to approximate solution sampling the neighborhood information"""
        if eval_mode == "volume":
            num_variables = self.num_variables_vol
            nn_basis = self.nn_basis_vol
            agg_model = self.agg_model_vol
        elif eval_mode == "surface":
            num_variables = self.num_variables_surf
            nn_basis = self.nn_basis_surf
            agg_model = self.agg_model_surf

        if self.encode_parameters:
            inlet_velocity = paddle.unsqueeze(inlet_velocity, 1)
            inlet_velocity = inlet_velocity.expand(
                [
                    inlet_velocity.shape[0],
                    volume_mesh_centers.shape[1],
                    inlet_velocity.shape[2],
                ]
            )
            inlet_velocity = inlet_velocity / self.param_scaling_factors[0]

            air_density = paddle.unsqueeze(air_density, 1)
            air_density = air_density.expand(
                [
                    air_density.shape[0],
                    volume_mesh_centers.shape[1],
                    air_density.shape[2],
                ]
            )
            air_density = air_density / self.param_scaling_factors[1]

            params = paddle.concat((inlet_velocity, air_density), axis=-1)
            param_encoding = self.parameter_model(params)

        for f in range(num_variables):
            for p in range(num_sample_points):
                if p == 0:
                    volume_m_c = volume_mesh_centers
                else:
                    noise = paddle.rand(
                        shape=volume_mesh_centers.shape, dtype=volume_mesh_centers.dtype
                    )
                    noise = 2 * (noise - 0.5)
                    noise = noise / noise_intensity
                    dist = paddle.sqrt(
                        noise[:, :, 0:1] ** 2.0
                        + noise[:, :, 1:2] ** 2.0
                        + noise[:, :, 2:3] ** 2.0
                    )
                    volume_m_c = volume_mesh_centers + noise
                basis_f = nn_basis[f](volume_m_c)
                output = paddle.concat((basis_f, encoding_node, encoding_g), axis=-1)
                if self.encode_parameters:
                    output = paddle.concat((output, param_encoding), axis=-1)
                if p == 0:
                    output_center = agg_model[f](output)
                else:
                    if p == 1:
                        output_neighbor = agg_model[f](output) * (1.0 / dist)
                        dist_sum = 1.0 / dist
                    else:
                        output_neighbor += agg_model[f](output) * (1.0 / dist)
                        dist_sum += 1.0 / dist
            if num_sample_points > 1:
                output_res = 0.5 * output_center + 0.5 * output_neighbor / dist_sum
            else:
                output_res = output_center
            if f == 0:
                output_all = output_res
            else:
                output_all = paddle.concat((output_all, output_res), axis=-1)

        return output_all

    def forward(
        self,
        data_dict,
    ):
        # Loading STL inputs, bounding box grids, precomputed SDF and scaling factors

        # STL nodes
        geo_centers = data_dict["geometry_coordinates"]

        # Bounding box grid
        s_grid = data_dict["surf_grid"]
        sdf_surf_grid = data_dict["sdf_surf_grid"]
        # Scaling factors
        surf_max = data_dict["surface_min_max"][:, 1]
        surf_min = data_dict["surface_min_max"][:, 0]

        # Parameters
        stream_velocity = data_dict["stream_velocity"]
        air_density = data_dict["air_density"]

        if self.output_features_vol is not None:
            # Represent geometry on computational grid
            # Computational domain grid
            p_grid = data_dict["grid"]
            sdf_grid = data_dict["sdf_grid"]
            # Scaling factors
            vol_max = data_dict["volume_min_max"][:, 1]
            vol_min = data_dict["volume_min_max"][:, 0]

            # Normalize based on computational domain
            geo_centers_vol = 2.0 * (geo_centers - vol_min) / (vol_max - vol_min) - 1
            encoding_g_vol = self.geo_rep(geo_centers_vol, p_grid, sdf_grid)

            # Normalize based on BBox around surface (car)
            geo_centers_surf = (
                2.0 * (geo_centers - surf_min) / (surf_max - surf_min) - 1
            )
            encoding_g_surf = self.geo_rep(geo_centers_surf, s_grid, sdf_surf_grid)

            # SDF on volume mesh nodes
            sdf_nodes = data_dict["sdf_nodes"]
            # Positional encoding based on closest point on surface to a volume node
            pos_volume_closest = data_dict["pos_volume_closest"]
            # Positional encoding based on center of mass of geometry to volume node
            pos_volume_center_of_mass = data_dict["pos_volume_center_of_mass"]
            encoding_node_vol = paddle.concat(
                (sdf_nodes, pos_volume_closest, pos_volume_center_of_mass), axis=-1
            )

            # Calculate positional encoding on volume nodes
            encoding_node_vol = self.position_encoder(
                encoding_node_vol, eval_mode="volume"
            )

        if self.output_features_surf is not None:
            # Represent geometry on bounding box
            geo_centers_surf = (
                2.0 * (geo_centers - surf_min) / (surf_max - surf_min) - 1
            )
            encoding_g_surf = self.geo_rep(geo_centers_surf, s_grid, sdf_surf_grid)

            # Positional encoding based on center of mass of geometry to surface node
            pos_surface_center_of_mass = data_dict["pos_surface_center_of_mass"]
            encoding_node_surf = pos_surface_center_of_mass

            # Calculate positional encoding on surface centers
            encoding_node_surf = self.position_encoder(
                encoding_node_surf, eval_mode="surface"
            )

        encoding_g = 0.5 * encoding_g_surf
        # Average the encodings
        if self.output_features_vol is not None:
            encoding_g += 0.5 * encoding_g_vol

        if self.output_features_vol is not None:
            # Calculate local geometry encoding for volume
            # Sampled points on volume
            volume_mesh_centers = data_dict["volume_mesh_centers"]
            encoding_g_vol = self.geo_encoding_local(
                encoding_g, volume_mesh_centers, p_grid
            )

            # Approximate solution on volume node
            output_vol = self.calculate_solution(
                volume_mesh_centers,
                encoding_g_vol,
                encoding_node_vol,
                stream_velocity,
                air_density,
                eval_mode="volume",
            )
        else:
            output_vol = None

        if self.output_features_surf is not None:
            # Sampled points on surface
            surface_mesh_centers = data_dict["surface_mesh_centers"]
            surface_normals = data_dict["surface_normals"]
            surface_areas = data_dict["surface_areas"]

            # Neighbors of sampled points on surface
            surface_mesh_neighbors = data_dict["surface_mesh_neighbors"]
            surface_neighbors_normals = data_dict["surface_neighbors_normals"]
            surface_neighbors_areas = data_dict["surface_neighbors_areas"]
            surface_areas = paddle.unsqueeze(surface_areas, -1)
            surface_neighbors_areas = paddle.unsqueeze(surface_neighbors_areas, -1)
            # Calculate local geometry encoding for surface
            encoding_g_surf = self.geo_encoding_local_surface(
                0.5 * encoding_g_surf, surface_mesh_centers, s_grid
            )

            # Approximate solution on surface cell center
            if not self.surface_neighbors:
                output_surf = self.calculate_solution(
                    surface_mesh_centers,
                    encoding_g_surf,
                    encoding_node_surf,
                    stream_velocity,
                    air_density,
                    eval_mode="surface",
                    num_sample_points=1,
                    noise_intensity=500,
                )
            else:
                output_surf = self.calculate_solution_with_neighbors(
                    surface_mesh_centers,
                    encoding_g_surf,
                    encoding_node_surf,
                    surface_mesh_neighbors,
                    surface_normals,
                    surface_neighbors_normals,
                    surface_areas,
                    surface_neighbors_areas,
                    stream_velocity,
                    air_density,
                )
        else:
            output_surf = None

        return output_vol, output_surf


if __name__ == "__main__":
    from hydra import compose
    from hydra import initialize
    from omegaconf import OmegaConf

    if paddle.device.cuda.device_count() >= 1:
        paddle.set_device("gpu")
    else:
        paddle.set_device("cpu")
    cfg = OmegaConf.register_new_resolver("eval", eval)
    with initialize(version_base="1.3", config_path="../../scripts/conf"):
        cfg = compose(config_name="config")
    cfg.model.model_type = "combined"
    model = DoMINO(
        input_features=3,
        output_features_vol=5,
        output_features_surf=4,
        model_parameters=cfg.model,
    )

    bsize = 1
    nx, ny, nz = 128, 64, 48
    num_neigh = 7
    pos_normals_closest_vol = paddle.randn([bsize, 100, 3])
    pos_normals_com_vol = paddle.randn([bsize, 100, 3])
    pos_normals_com_surface = paddle.randn([bsize, 100, 3])
    geom_centers = paddle.randn([bsize, 100, 3])
    grid = paddle.randn([bsize, nx, ny, nz, 3])
    surf_grid = paddle.randn([bsize, nx, ny, nz, 3])
    sdf_grid = paddle.randn([bsize, nx, ny, nz])
    sdf_surf_grid = paddle.randn([bsize, nx, ny, nz])
    sdf_nodes = paddle.randn([bsize, 100, 1])
    surface_coordinates = paddle.randn([bsize, 100, 3])
    surface_neighbors = paddle.randn([bsize, 100, num_neigh, 3])
    surface_normals = paddle.randn([bsize, 100, 3])
    surface_neighbors_normals = paddle.randn([bsize, 100, num_neigh, 3])
    surface_sizes = paddle.randn([bsize, 100, 3])
    surface_neighbors_sizes = paddle.randn([bsize, 100, num_neigh, 3])
    volume_coordinates = paddle.randn([bsize, 100, 3])
    vol_grid_max_min = paddle.randn([bsize, 2, 3])
    surf_grid_max_min = paddle.randn([bsize, 2, 3])
    stream_velocity = paddle.randn([bsize, 1])
    air_density = paddle.randn([bsize, 1])
    input_dict = {
        "pos_volume_closest": pos_normals_closest_vol,
        "pos_volume_center_of_mass": pos_normals_com_vol,
        "pos_surface_center_of_mass": pos_normals_com_surface,
        "geometry_coordinates": geom_centers,
        "grid": grid,
        "surf_grid": surf_grid,
        "sdf_grid": sdf_grid,
        "sdf_surf_grid": sdf_surf_grid,
        "sdf_nodes": sdf_nodes,
        "surface_mesh_centers": surface_coordinates,
        "surface_mesh_neighbors": surface_neighbors,
        "surface_normals": surface_normals,
        "surface_neighbors_normals": surface_neighbors_normals,
        "surface_areas": surface_sizes,
        "surface_neighbors_areas": surface_neighbors_sizes,
        "volume_mesh_centers": volume_coordinates,
        "volume_min_max": vol_grid_max_min,
        "surface_min_max": surf_grid_max_min,
        "stream_velocity": stream_velocity,
        "air_density": air_density,
    }
    output = model(input_dict)
    print(f"{output[0].shape}, {output[1].shape}")
