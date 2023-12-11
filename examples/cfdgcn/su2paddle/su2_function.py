# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Tuple

import paddle
from su2paddle import common
from su2paddle import su2_function_mpi

from mpi4py import MPI  # isort:skip
import pysu2  # isort:skip


_global_max_ppe = -1


class SU2Module(paddle.nn.Layer):
    def __init__(
        self, config_file: str, mesh_file: str, dims: int = 2, num_zones: int = 1
    ) -> None:
        """Initialize the SU2 configurations for the provided config file.

        Args:
            config_file: str - The SU2 configuration file name.
            mesh_file: str - Optional parameter, if not set defaults to the mesh filename set in the config file.
                Can be used to run a batch with different meshes for each sample.
                Passing in mesh_file with batch_index parameter in string format (e.g., 'b{batch_index}_mesh.su2') causes each element in batch to get assigned to the correct mesh file (0 indexed).
                If running multiple processes in parallel, take care to name each mesh file uniquely to avoid conflicts
                (e.g., unique = str(os.getpid()); mesh_file = 'b{batch_index}_' + unique + '_mesh.su2').
            dims: int - Number of dimensions for the problem (2D or 3D).
            num_zones: int - Number of zones in the simulation (only 1 supported currently).
        """
        super().__init__()
        if num_zones != 1:
            raise ValueError("Only supports 1 zone for now.")
        if MPI.COMM_WORLD.Get_rank() != 0:
            raise ValueError("Not rank 0 in comm")
        if not _global_max_ppe > 0:
            raise ValueError(
                "Before running SU2Function, a (single) call to activate_su2_mpi is needed."
            )
        self.num_zones = num_zones
        self.dims = dims
        self.mesh_file = mesh_file

        self.forward_config = config_file
        self.forward_driver = None

    def forward(self, *inputs: paddle.Tensor) -> Tuple[paddle.Tensor, ...]:
        return SU2Function.apply(
            *inputs,
            self.forward_config,
            self.mesh_file,
            self.num_zones,
            self.dims,
            self.set_forward_driver,
        )

    def get_forward_driver(self):
        if self.forward_driver is None:
            raise AttributeError("Forward driver is only set after running forward()")
        return self.forward_driver

    def set_forward_driver(self, f):
        if self.forward_driver is not None:
            self.forward_driver.Postprocessing()
        self.forward_driver = f

    def __del__(self):
        """Close existing drivers and MPI communicators."""
        if hasattr(self, "forward_driver") and self.forward_driver is not None:
            self.forward_driver.Postprocessing()


class SU2Function(paddle.autograd.PyLayer):
    num_params = 5

    @staticmethod
    def forward(ctx, *inputs):
        su2_function_mpi.non_busy_post(MPI.COMM_WORLD)
        x = inputs[: -SU2Function.num_params]
        forward_config, mesh_file, num_zones, dims, set_forward_driver_hook = inputs[
            -SU2Function.num_params :
        ]

        if x[0].dim() < 2:
            raise ValueError(
                "Input is expected to have first dimension for batch, "
                "e.g. x[0, :] is first item in batch."
            )
        batch_size = x[0].shape[0]
        max_ppe = _global_max_ppe
        workers = MPI.COMM_WORLD.Get_size() - 1
        if 0 <= workers < batch_size:
            raise ValueError(
                "Batch size is larger than number of workers, not enough processes to run batch."
            )

        MPI.COMM_WORLD.bcast(su2_function_mpi.RunCode.RUN_FORWARD, root=0)
        procs_per_example = min(max_ppe, math.ceil(workers / batch_size))

        x = tuple([i.numpy() for i in x])

        MPI.COMM_WORLD.bcast(
            [num_zones, dims, forward_config, mesh_file, procs_per_example, x], root=0
        )

        # instantiate forward_driver while workers work
        worker_forward_config = MPI.COMM_WORLD.recv(source=1)
        forward_driver = pysu2.CSinglezoneDriver(
            worker_forward_config, num_zones, dims, MPI.COMM_SELF
        )
        num_diff_inputs = forward_driver.GetnDiff_Inputs()
        num_diff_outputs = forward_driver.GetnDiff_Outputs()
        if num_diff_inputs <= 0 or num_diff_outputs <= 0:
            raise ValueError(
                "Need to define at least one differentiable input and output. To run without differentiation, use the SU2Numpy class."
            )

        if len(x) != num_diff_inputs:
            raise ValueError(
                f"{len(x)} inputs were provided, but the config file ({forward_config}) defines {num_diff_inputs} diff inputs."
            )
        set_forward_driver_hook(forward_driver)
        ctx.num_diff_inputs = num_diff_inputs

        outputs = []
        su2_function_mpi.non_busy_wait(MPI.COMM_WORLD)
        for i in range(batch_size):
            output = MPI.COMM_WORLD.recv(source=1 + i * procs_per_example)
            outputs.append(output)
        outputs = tuple(
            common.pad_sequence(
                [paddle.to_tensor(o[i], dtype=paddle.float32) for o in outputs],
                batch_first=True,
            )
            for i in range(num_diff_outputs)
        )
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        su2_function_mpi.non_busy_post(MPI.COMM_WORLD)
        max_ppe = _global_max_ppe
        workers = MPI.COMM_WORLD.Get_size() - 1
        MPI.COMM_WORLD.bcast(su2_function_mpi.RunCode.RUN_ADJOINT, root=0)
        grad_outputs = tuple([i.numpy() for i in grad_outputs])
        MPI.COMM_WORLD.bcast(grad_outputs, root=0)
        batch_size = grad_outputs[0].shape[0]
        procs_per_example = min(max_ppe, math.ceil(workers / batch_size))
        su2_function_mpi.non_busy_wait(MPI.COMM_WORLD)
        grads = []
        for i in range(batch_size):
            grad = MPI.COMM_WORLD.recv(source=1 + i * procs_per_example)
            grads.append(grad)
        print("grads", len(grads), flush=True)
        grads = tuple(
            common.pad_sequence(
                [paddle.to_tensor(g[i], dtype=paddle.float32) for g in grads],
                batch_first=True,
            )
            for i in range(ctx.num_diff_inputs)
        )
        return tuple(
            [grads[0], grads[1], None, None]
        )  # + (None,) * SU2Function.num_params
