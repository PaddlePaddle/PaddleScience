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

from typing import Callable
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle
from paddle import nn
from paddle.distributed.fleet.utils import hybrid_parallel_util as hpu
from skimage import measure

import ppsci
from ppsci import geometry
from ppsci.autodiff import clear
from ppsci.solver import Solver
from ppsci.utils import logger
from ppsci.utils import misc
from ppsci.utils import save_load


class Trainer:
    def __init__(self, solver: "Solver") -> None:
        self.solver = solver

    def run_forward(
        self,
        expr_dicts: Tuple[Dict[str, Callable], ...],
        input_dicts: Tuple[Dict[str, "paddle.Tensor"], ...],
        model: nn.Layer,
    ) -> Tuple["paddle.Tensor", ...]:
        """Forward computation for training, including model forward and equation
        forward.

        Args:
            expr_dicts (Tuple[Dict[str, Callable], ...]): Tuple of expression dicts.
            input_dicts (Tuple[Dict[str, paddle.Tensor], ...]): Tuple of input dicts.
            model (nn.Layer): NN model.

        Returns:
            Tuple[paddle.Tensor, ...]: Tuple of output for each expression.
        """
        output_dicts = []
        for i, expr_dict in enumerate(expr_dicts):
            # model forward
            output_dict = model(input_dicts[i])

            # equation forward
            data_dict = {k: v for k, v in input_dicts[i].items()}
            data_dict.update(output_dict)
            for name, expr in expr_dict.items():
                output_dict[name] = expr(data_dict)

            # put field 'area' into output_dict
            if "area" in input_dicts[i]:
                output_dict["area"] = input_dicts[i]["area"]

            output_dicts.append(output_dict)

            # clear differentiation cache
            clear()
        return output_dicts

    def train_forward(self):
        input_dicts_list = []
        label_dicts_list = []
        weight_dicts_list = []
        output_dicts_list = []
        for _ in range(1, self.solver.iters_per_epoch + 1):
            loss_dict = misc.Prettydefaultdict(float)
            loss_dict["loss"] = 0.0

            input_dicts = []
            label_dicts = []
            weight_dicts = []
            for _, _constraint in self.solver.constraint.items():
                try:
                    input_dict, label_dict, weight_dict = next(_constraint.data_iter)
                except StopIteration:
                    _constraint.data_iter = iter(_constraint.data_loader)
                    input_dict, label_dict, weight_dict = next(_constraint.data_iter)

                for v in input_dict.values():
                    v.stop_gradient = False

                # gather each constraint's input, label, weight to a list
                input_dicts.append(input_dict)
                label_dicts.append(label_dict)
                weight_dicts.append(weight_dict)

            output_dicts = self.run_forward(
                (
                    _constraint.output_expr
                    for _constraint in self.solver.constraint.values()
                ),
                input_dicts,
                self.solver.model,
            )
            input_dicts_list.append(input_dicts)
            label_dicts_list.append(label_dicts)
            weight_dicts_list.append(weight_dicts)
            output_dicts_list.append(output_dicts)
        return input_dicts_list, label_dicts_list, weight_dicts_list, output_dicts_list

    def train_backward(self, constraint_losses_list):
        for iter_id in range(1, self.solver.iters_per_epoch + 1):
            # compute loss for each constraint according to its' own output, label and weight
            total_loss = 0
            for i, _ in enumerate(self.solver.constraint.values()):
                loss = constraint_losses_list[i]
                if isinstance(loss, Callable):
                    total_loss += loss(self.solver.model, iter_id - 1)
                elif isinstance(loss, List):
                    total_loss += loss[iter_id - 1]

            if self.solver.use_amp:
                total_loss_scaled = self.solver.scaler.scale(total_loss)
                total_loss_scaled.backward()
            else:
                total_loss.backward()

            # update parameters
            if (
                iter_id % self.solver.update_freq == 0
                or iter_id == self.solver.iters_per_epoch
            ):
                if self.solver.world_size > 1:
                    # fuse + allreduce manually before optimization if use DDP + no_sync
                    # details in https://github.com/PaddlePaddle/Paddle/issues/48898#issuecomment-1343838622
                    hpu.fused_allreduce_gradients(
                        list(self.solver.model.parameters()), None
                    )
                if self.solver.use_amp:
                    self.solver.scaler.minimize(
                        self.solver.optimizer, total_loss_scaled
                    )
                else:
                    self.solver.optimizer.step()
                self.solver.optimizer.clear_grad()

            # update learning rate by step
            if (
                self.solver.lr_scheduler is not None
                and not self.solver.lr_scheduler.by_epoch
            ):
                self.solver.lr_scheduler.step()

            # update and log training information
            self.solver.global_step += 1

    def train_batch(self):
        """Training."""
        self.solver.global_step = (
            self.solver.best_metric["epoch"] * self.solver.iters_per_epoch
        )

        for epoch_id in range(
            self.solver.best_metric["epoch"] + 1, self.solver.epochs + 1
        ):
            # forward
            (
                input_dicts_list,
                label_dicts_list,
                weight_dicts_list,
                output_dicts_list,
            ) = self.train_forward()

            # batch loss
            constraint_losses_list = []
            for _, _constraint in enumerate(self.solver.constraint.values()):
                if not isinstance(_constraint.loss, FunctionalLossBatch):
                    raise TypeError(
                        "Loss function of constraint should be FunctionalLossBatch when using train_batch"
                    )
                constraint_loss_list = _constraint.loss(
                    output_dicts_list,
                    label_dicts_list,
                    weight_dicts_list,
                    input_dicts_list,
                )
                constraint_losses_list.append(constraint_loss_list)

            # backward
            self.train_backward(constraint_losses_list)

            # log training summation at end of a epoch
            metric_msg = ", ".join(
                [
                    self.solver.train_output_info[key].avg_info
                    for key in self.solver.train_output_info
                ]
            )
            logger.info(
                f"[Train][Epoch {epoch_id}/{self.solver.epochs}][Avg] {metric_msg}"
            )
            self.solver.train_output_info.clear()

            cur_metric = float("inf")
            # evaluate during training
            if (
                self.solver.eval_during_train
                and epoch_id % self.solver.eval_freq == 0
                and epoch_id >= self.solver.start_eval_epoch
            ):
                cur_metric, metric_dict_group = self.eval(epoch_id)
                if cur_metric < self.solver.best_metric["metric"]:
                    self.solver.best_metric["metric"] = cur_metric
                    self.solver.best_metric["epoch"] = epoch_id
                    save_load.save_checkpoint(
                        self.solver.model,
                        self.solver.optimizer,
                        self.solver.best_metric,
                        self.solver.scaler,
                        self.solver.output_dir,
                        "best_model",
                        self.solver.equation,
                    )
                logger.info(
                    f"[Eval][Epoch {epoch_id}]"
                    f"[best metric: {self.solver.best_metric['metric']}]"
                    f"[best epoch: {self.solver.best_metric['epoch']}]"
                    f"[current metric: {cur_metric}]"
                )
                for metric_dict in metric_dict_group.values():
                    logger.scaler(
                        {f"eval/{k}": v for k, v in metric_dict.items()},
                        epoch_id,
                        self.vdl_writer,
                        self.wandb_writer,
                    )

                # visualize after evaluation
                if self.solver.visualizer is not None:
                    self.solver.visualize(epoch_id)

            # update learning rate by epoch
            if (
                self.solver.lr_scheduler is not None
                and self.solver.lr_scheduler.by_epoch
            ):
                self.solver.lr_scheduler.step()

            # save epoch model every save_freq epochs
            if self.solver.save_freq > 0 and epoch_id % self.solver.save_freq == 0:
                save_load.save_checkpoint(
                    self.solver.model,
                    self.solver.optimizer,
                    {"metric": cur_metric, "epoch": epoch_id},
                    self.solver.scaler,
                    self.solver.output_dir,
                    f"epoch_{epoch_id}",
                    self.solver.equation,
                )

            # save the latest model for convenient resume training
            save_load.save_checkpoint(
                self.solver.model,
                self.solver.optimizer,
                {"metric": cur_metric, "epoch": epoch_id},
                self.solver.scaler,
                self.solver.output_dir,
                "latest",
                self.solver.equation,
            )

        # close VisualDL
        if self.solver.vdl_writer is not None:
            self.solver.vdl_writer.close()


class FunctionalLossBatch(ppsci.loss.base.Loss):
    def __init__(
        self,
        loss_expr: Callable,
        reduction: Literal["mean", "sum"] = "mean",
        weight: Optional[Union[float, Dict[str, float]]] = None,
    ):
        if reduction not in ["mean", "sum"]:
            raise ValueError(
                f"reduction should be 'mean' or 'sum', but got {reduction}"
            )
        super().__init__(reduction, weight)
        self.loss_expr = loss_expr

    def forward(
        self,
        output_dicts_list,
        label_dicts_list=None,
        weight_dicts_list=None,
        input_dicts_list=None,
    ):
        return self.loss_expr(
            output_dicts_list, label_dicts_list, weight_dicts_list, input_dicts_list
        )


class Sampler:
    def __init__(
        self,
        geom: geometry.Geometry,
        bounds: Tuple[Tuple[float, float], ...],
        criteria: Optional[Callable] = None,
    ) -> None:
        self.geom = geom
        self.dim = geom.ndim
        self.dim_keys = geom.dim_keys
        self.bounds = bounds
        self.criteria = criteria

    def stratified_random(self, n_samples: Tuple[int, ...]) -> np.ndarray:
        """Stratified random."""
        # Divide the geometry into uniformly sized chunks
        # and randomly sample within each chunk.

        zeros = np.zeros(n_samples)
        grid_points = np.transpose(np.where(zeros == 0)).astype(
            paddle.get_default_dtype()
        )
        all_samples = np.prod(n_samples)
        for i in range(self.dim):
            random = np.random.uniform(0.0, 1.0, (all_samples))
            grid_size = (self.bounds[i][1] - self.bounds[i][0]) / n_samples[i]
            grid_points[:, i] = (grid_points[:, i] + random) * grid_size
        return grid_points

    def sample_interior_stratified(self, n_samples: Tuple[int, ...], n_iter: int):
        """Sample random points in the geometry and return those meet criteria."""
        points = self.stratified_random(n_samples)
        for _ in range(1, n_iter):
            points = np.concatenate((points, self.stratified_random(n_samples)), axis=0)

        if self.criteria is not None:
            criteria_mask = self.criteria(*np.split(points, self.dim, axis=1)).flatten()
            points = points[criteria_mask]
        else:
            criteria_mask = self.geom.is_inside(points)
            points = points[criteria_mask]
        points_dict = {}
        for i in range(self.dim):
            points_dict[self.dim_keys[i]] = points[:, i].reshape([-1, 1])

        return points_dict, criteria_mask


class Plot:
    def __init__(self, filename, problem, n_cells, threshold=0.25) -> None:
        self.filename = filename
        self.problem = problem
        self.n_cells = n_cells
        self.threshold = threshold

    def prepare_data(self):
        cx = 0.5 * self.problem.geo_dim[0] / self.n_cells[0]
        cy = 0.5 * self.problem.geo_dim[1] / self.n_cells[1]
        cz = 0.5 * self.problem.geo_dim[2] / self.n_cells[2]

        x = np.linspace(
            self.problem.geo_origin[0] + cx,
            self.problem.geo_dim[0] - cx,
            num=self.n_cells[0],
            dtype=paddle.get_default_dtype(),
        )
        y = np.linspace(
            self.problem.geo_origin[1] + cy,
            self.problem.geo_dim[1] - cy,
            num=self.n_cells[1],
            dtype=paddle.get_default_dtype(),
        )
        z = np.linspace(
            self.problem.geo_origin[2] + cz,
            self.problem.geo_dim[2] - cz,
            num=self.n_cells[2],
            dtype=paddle.get_default_dtype(),
        )
        xs, ys, zs = np.meshgrid(x, y, z, indexing="ij")

        input_dict = {}
        input_dict["x"] = paddle.to_tensor(
            xs.reshape(-1, 1), dtype=paddle.get_default_dtype()
        )
        input_dict["y"] = paddle.to_tensor(
            ys.reshape(-1, 1), dtype=paddle.get_default_dtype()
        )
        input_dict["z"] = paddle.to_tensor(
            zs.reshape(-1, 1), dtype=paddle.get_default_dtype()
        )

        self.input_dict = input_dict

    def compute_densities(self):
        self.densities = (
            self.problem.density_net(self.input_dict)["densities"]
            .numpy()
            .reshape(self.n_cells)
        )

    def compute_mirror(self):
        if self.problem.mirror[0]:
            self.densities = np.concatenate(
                (
                    self.densities,
                    self.densities[::-1, :, :],
                ),
                axis=0,
            )
        if self.problem.mirror[1]:
            self.densities = np.concatenate(
                (
                    self.densities,
                    self.densities[:, ::-1, :],
                ),
                axis=1,
            )
        if self.problem.mirror[2]:
            self.densities = np.concatenate(
                (
                    self.densities,
                    self.densities[:, :, ::-1],
                ),
                axis=2,
            )

    def pad_with_zeros(self):
        self.density_grid = np.pad(
            self.density_grid, ((1, 1), (1, 1), (1, 1)), "constant", constant_values=0.0
        )

    def save_solid(self):
        if self.problem.mirror:
            for i in range(len(self.problem.mirror)):
                self.n_cells[i] *= 2 if self.problem.mirror[i] else 1
        self.density_grid = np.reshape(
            self.densities, (self.n_cells[0], self.n_cells[1], self.n_cells[2])
        )
        self.pad_with_zeros()

        if (
            np.amax(self.density_grid) < self.threshold
            or np.amin(self.density_grid) > self.threshold
        ):
            print("Warning! Cannot save density grid cause the levelset is empty")
            return

        verts, faces, normals, _ = measure.marching_cubes(
            self.density_grid,
            level=self.threshold,
            spacing=[0.005, 0.005, 0.005],
            gradient_direction="ascent",
            method="lewiner",
        )

        with open(self.filename, "w") as file:
            for item in verts:
                file.write(f"v {item[0]} {item[1]} {item[2]}\n")

            for item in normals:
                file.write(f"vn {item[0]} {item[1]} {item[2]}\n")

            for item in faces:
                idx_0 = item[0] + 1
                idx_1 = item[1] + 1
                idx_2 = item[2] + 1
                file.write(f"f {idx_0}//{idx_0} {idx_1}//{idx_1} {idx_2}//{idx_2}\n")

    def plot_3d(self):
        self.prepare_data()
        self.compute_densities()
        if self.problem.mirror:
            self.compute_mirror()
        self.save_solid()
