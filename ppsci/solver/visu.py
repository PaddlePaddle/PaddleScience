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

import os
import os.path as osp

import numpy as np
import paddle
import paddle.amp as amp

import ppsci.data.dataset as dataset
from ppsci.utils import expression
from ppsci.utils import misc


def visualize_func(solver, epoch_id):
    """Visualization program

    Args:
        solver (Solver): Main Solver.
        epoch_id (int): Epoch id.

    Returns:
        Dict[str, Any]: Metric collected during visualization.
    """
    for _, _visualizer in solver.visualizer.items():
        all_input = misc.Prettydefaultdict(list)
        all_output = misc.Prettydefaultdict(list)

        input_dict = _visualizer.input_dict
        for key in input_dict:
            if not paddle.is_tensor(input_dict[key]):
                input_dict[key] = paddle.to_tensor(input_dict[key], stop_gradient=False)

        evaluator = expression.ExpressionSolver(
            _visualizer.input_keys, _visualizer.output_keys, solver.model
        )
        for output_key, output_expr in _visualizer.output_expr.items():
            evaluator.add_target_expr(output_expr, output_key)

        # forward
        if solver.use_amp:
            with amp.auto_cast(level=solver.amp_level):
                output_dict = evaluator(input_dict)
        else:
            output_dict = evaluator(input_dict)

        # collect batch data
        for key, input in input_dict.items():
            all_input[key].append(
                input.detach()
                if solver.world_size == 1
                else misc.all_gather(input.detach())
            )
        for key, output in output_dict.items():
            all_output[key].append(
                output.detach()
                if solver.world_size == 1
                else misc.all_gather(output.detach())
            )

        for key in all_input:
            all_input[key] = paddle.concat(all_input[key])
        for key in all_output:
            all_output[key] = paddle.concat(all_output[key])

        if solver.rank == 0:
            visual_dir = osp.join(solver.output_dir, "visual", f"epoch_{epoch_id}")
            os.makedirs(visual_dir, exist_ok=True)
            _visualizer.save(
                osp.join(visual_dir, _visualizer.prefix), {**all_input, **all_output}
            )


@paddle.no_grad()
def visualize_func_3D(solver, epoch_id):
    """Visualization program, perform forward here

    Args:
        solver (Solver): Main Solver.
        epoch_id (int): Epoch id.

    """
    # construct input
    _visualizer = next(iter(solver.visualizer.values()))
    input_dict, label, onestep_xyz = _visualizer.construct_input()

    # reconstruct input
    n = len(next(iter(input_dict.values())))
    splited_shares = int(n / _visualizer.visualizer_batch_size)
    input_split = {
        key: paddle.to_tensor(
            np.array_split(value, splited_shares),
            dtype=paddle.float32,
            stop_gradient=False,
        )
        for key, value in input_dict.items()
    }

    # forward and predict
    solution = {key: np.zeros((n, 1)).astype(np.float32) for key in dataset.Label}
    for i in range(splited_shares):
        input_i = {key: value[i] for key, value in input_split.items()}
        output = solver.model(input_i)
        for key in output.keys():
            m = output[key].shape[0]
            solution[key][i * m : (i + 1) * m] = output[key].numpy()

    solution = dataset.denormalization(solution, _visualizer.factor_dict)

    # save vtu
    if solver.rank == 0:
        visual_dir = osp.join(solver.output_dir, "visual", f"epoch_{epoch_id}")
    _visualizer.save(visual_dir, onestep_xyz, solution)
