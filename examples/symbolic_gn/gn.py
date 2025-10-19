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

import os
import sys
import numpy as np
import paddle
from omegaconf import DictConfig
import hydra
import ppsci
from ppsci.utils import logger

from simulate import SimulationDataset
from ppsci.arch.symbolic_gn import OGN, VarOGN, HGN, get_edge_index



class OGNDataset:
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        self.mode = mode
        self.data = None
        self.labels = None
        self.input_keys = tuple(cfg.MODEL.input_keys)
        self.label_keys = tuple(cfg.MODEL.output_keys)
        self._prepare_data()

    def _prepare_data(self):
        # Get time step size for physics system
        dt = 1e-2
        for sim_set in self.cfg.DATATYPE_TYPE:
            if sim_set['sim'] == self.cfg.DATA.type:
                dt = sim_set['dt'][0]
                break

        # Generate simulation data
        sim_dataset = SimulationDataset(
            sim=self.cfg.DATA.type,
            n=self.cfg.DATA.num_nodes,
            dim=self.cfg.DATA.dimension,
            dt=dt,
            nt=self.cfg.DATA.time_steps,
            seed=self.cfg.seed
        )
        sim_dataset.simulate(self.cfg.DATA.num_samples)
        
        # Compute target data based on model architecture
        if self.cfg.MODEL.arch == "HGN":
            raw_target_data = sim_dataset.get_derivative()  # [v, a] for HGN
        else:
            raw_target_data = sim_dataset.get_acceleration()  # Acceleration for OGN/VarOGN
        
        # Downsample data
        downsample_factor = self.cfg.DATA.downsample_factor
        input_data = np.concatenate([sim_dataset.data[:, i] for i in range(0, sim_dataset.data.shape[1], downsample_factor)])
        target_data = np.concatenate([raw_target_data[:, i] for i in range(0, sim_dataset.data.shape[1], downsample_factor)])

        # Split train/validation set
        from sklearn.model_selection import train_test_split
        test_size = 1.0 - getattr(self.cfg.TRAIN, 'train_split', 0.8)
        
        if self.mode == "train":
            input_data, _, target_data, _ = train_test_split(
                input_data, target_data, test_size=test_size, shuffle=False
            )
        elif self.mode == "eval":
            _, input_data, _, target_data = train_test_split(
                input_data, target_data, test_size=test_size, shuffle=False
            )
        
        # Generate edge indices
        edge_index = get_edge_index(self.cfg.DATA.num_nodes, self.cfg.DATA.type)

        self.data = {
            "node_features": input_data.astype(np.float32),
            "edge_index": np.tile(edge_index.T.numpy(), (len(input_data), 1, 1))
        }
        self.labels = {
            list(self.cfg.MODEL.output_keys)[0]: target_data.astype(np.float32)
        }

    def __len__(self):
        return len(self.data["node_features"])

    def __getitem__(self, idx):
        return {
            "input": {
                "node_features": self.data["node_features"][idx],
                "edge_index": self.data["edge_index"][idx]
            },
            "label": {
                list(self.cfg.MODEL.output_keys)[0]: self.labels[list(self.cfg.MODEL.output_keys)[0]][idx]
            }
        }
        
    def get_data_and_labels(self):
        return self.data, self.labels
        
    def __call__(self):
        return self


def create_loss_function(cfg):
    def loss_function(input_dict, label_dict, model_output):
        # Get loss type from config (MAE or MSE)
        loss_type = getattr(cfg.TRAIN.loss, 'type', 'MAE')
        
        # Get output key
        output_key = list(cfg.MODEL.output_keys)[0] if cfg.MODEL.arch != "HGN" else "derivative"
        
        # Compute base loss
        if cfg.MODEL.arch == "HGN":
            output_key = "derivative"
            pred_accel = model_output[output_key][:, cfg.DATA.dimension:]
            target_accel = label_dict[output_key][:, cfg.DATA.dimension:]
            if loss_type == "MSE":
                base_loss = paddle.mean(paddle.square(pred_accel - target_accel))
            else:  # MAE
                base_loss = paddle.mean(paddle.abs(pred_accel - target_accel))
        else:
            pred = model_output[output_key]
            target = label_dict[output_key]
            if loss_type == "MSE":
                base_loss = paddle.mean(paddle.square(pred - target))
            else:  # MAE
                base_loss = paddle.mean(paddle.abs(pred - target))

        # Add regularization if configured
        if cfg.MODEL.regularization_type == "l1" and cfg.MODEL.arch in ["OGN", "VarOGN"]:
            reg_loss = cfg.MODEL.l1_strength * paddle.mean(paddle.abs(input_dict["node_features"]))
            return base_loss + reg_loss

        return base_loss
    
    return loss_function


def train(cfg):
    # Set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    
    # Parse automatic configuration parameters
    if cfg.MODEL.n_f == "auto":
        cfg.MODEL.n_f = cfg.DATA.dimension * 2 + 2
    if cfg.MODEL.ndim == "auto":
        cfg.MODEL.ndim = cfg.DATA.dimension
    if cfg.TRAIN.batch_size == "auto":
        cfg.TRAIN.batch_size = int(64 * (4 / cfg.DATA.num_nodes) ** 2)
    
    # Update output keys for HGN
    if cfg.MODEL.arch == "HGN":
        cfg.MODEL.output_keys = ["derivative"]
    
    # Create datasets
    logger.message(f"Creating {cfg.DATA.type} dataset using simulate.py...")
    train_dataset = OGNDataset(cfg, mode="train")
    eval_dataset = OGNDataset(cfg, mode="eval")
    
    # Get training and evaluation data
    train_data, train_labels = train_dataset.get_data_and_labels()
    eval_data, eval_labels = eval_dataset.get_data_and_labels()
    
    # Create model based on architecture
    logger.message(f"Creating model: {cfg.MODEL.arch}")
    if cfg.MODEL.arch == "OGN":
        model = OGN(
            input_keys=tuple(cfg.MODEL.input_keys),
            output_keys=tuple(cfg.MODEL.output_keys),
            n_f=cfg.MODEL.n_f,
            msg_dim=cfg.MODEL.msg_dim,
            ndim=cfg.MODEL.ndim,
            dt=cfg.MODEL.dt,
            hidden=cfg.MODEL.hidden,
            aggr=cfg.MODEL.aggr
        )
    elif cfg.MODEL.arch == "VarOGN":
        model = VarOGN(
            input_keys=tuple(cfg.MODEL.input_keys),
            output_keys=tuple(cfg.MODEL.output_keys),
            n_f=cfg.MODEL.n_f,
            msg_dim=cfg.MODEL.msg_dim,
            ndim=cfg.MODEL.ndim,
            dt=cfg.MODEL.dt,
            hidden=cfg.MODEL.hidden,
            aggr=cfg.MODEL.aggr
        )
    elif cfg.MODEL.arch == "HGN":
        model = HGN(
            input_keys=tuple(cfg.MODEL.input_keys),
            output_keys=tuple(cfg.MODEL.output_keys),
            n_f=cfg.MODEL.n_f,
            ndim=cfg.MODEL.ndim,
            hidden=cfg.MODEL.hidden,
            aggr=cfg.MODEL.aggr
        )
    else:
        raise ValueError(f"Unsupported model architecture: {cfg.MODEL.arch}")
    
    # Create loss function and constraints
    loss_fn = create_loss_function(cfg)
    train_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": train_data,
                "label": train_labels
            },
            "batch_size": cfg.TRAIN.batch_size,
            "sampler": {"name": "BatchSampler", "drop_last": False, "shuffle": True},
        },
        loss=loss_fn,
        name="train_constraint",
    )
    
    eval_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": eval_data,
                "label": eval_labels
            },
            "batch_size": cfg.EVAL.batch_size,
            "sampler": {"name": "BatchSampler", "drop_last": False, "shuffle": False},
        },
        loss=loss_fn,
        name="eval_constraint",
    )
    
    constraint = {
        train_constraint.name: train_constraint,
        eval_constraint.name: eval_constraint,
    }
    
    # Calculate iterations per epoch
    batch_per_epoch = int(1000*10 / (cfg.TRAIN.batch_size/32.0))
    
    # Create optimizer and learning rate scheduler
    if cfg.TRAIN.lr_scheduler.name == "OneCycleLR":
        lr_scheduler = paddle.optimizer.lr.OneCycleLR(
            max_learning_rate=cfg.TRAIN.lr_scheduler.max_learning_rate,
            total_steps=cfg.TRAIN.epochs*batch_per_epoch,
            divide_factor=cfg.TRAIN.lr_scheduler.final_div_factor
        )
    else:
        lr_scheduler = paddle.optimizer.lr.ExponentialDecay(
            learning_rate=cfg.TRAIN.optimizer.learning_rate,
            gamma=0.9
        )
    
    optimizer = paddle.optimizer.Adam(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=cfg.TRAIN.optimizer.weight_decay
    )
    
    # Create PaddleScience Solver and start training
    solver = ppsci.solver.Solver(
        model,
        constraint,
        optimizer=optimizer,
        cfg=cfg,
    )
    solver.train()


def evaluate(cfg: DictConfig):
    # Parse automatic configuration parameters
    if cfg.MODEL.n_f == "auto":
        cfg.MODEL.n_f = cfg.DATA.dimension * 2 + 2
    if cfg.MODEL.ndim == "auto":
        cfg.MODEL.ndim = cfg.DATA.dimension
    if cfg.MODEL.arch == "HGN":
        cfg.MODEL.output_keys = ["derivative"]
    
    # Create model based on architecture
    if cfg.MODEL.arch == "OGN":
        model = OGN(
            input_keys=tuple(cfg.MODEL.input_keys),
            output_keys=tuple(cfg.MODEL.output_keys),
            n_f=cfg.MODEL.n_f,
            msg_dim=cfg.MODEL.msg_dim,
            ndim=cfg.MODEL.ndim,
            dt=cfg.MODEL.dt,
            hidden=cfg.MODEL.hidden,
            aggr=cfg.MODEL.aggr
        )
    elif cfg.MODEL.arch == "VarOGN":
        model = VarOGN(
            input_keys=tuple(cfg.MODEL.input_keys),
            output_keys=tuple(cfg.MODEL.output_keys),
            n_f=cfg.MODEL.n_f,
            msg_dim=cfg.MODEL.msg_dim,
            ndim=cfg.MODEL.ndim,
            dt=cfg.MODEL.dt,
            hidden=cfg.MODEL.hidden,
            aggr=cfg.MODEL.aggr
        )
    elif cfg.MODEL.arch == "HGN":
        model = HGN(
            input_keys=tuple(cfg.MODEL.input_keys),
            output_keys=tuple(cfg.MODEL.output_keys),
            n_f=cfg.MODEL.n_f,
            ndim=cfg.MODEL.ndim,
            hidden=cfg.MODEL.hidden,
            aggr=cfg.MODEL.aggr
        )
    else:
        raise ValueError(f"Unsupported model architecture: {cfg.MODEL.arch}")
    
    # Create evaluation dataset
    eval_dataset = OGNDataset(cfg, mode="eval")
    eval_data, eval_labels = eval_dataset.get_data_and_labels()
    
    # Create loss function and constraint
    loss_fn = create_loss_function(cfg)
    eval_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": eval_data,
                "label": eval_labels
            },
            "batch_size": cfg.EVAL.batch_size,
            "sampler": {"name": "BatchSampler", "drop_last": False, "shuffle": False},
        },
        loss=loss_fn,
        name="eval_constraint",
    )
    
    constraint = {
        eval_constraint.name: eval_constraint,
    }
    
    # Create PaddleScience Solver
    solver = ppsci.solver.Solver(model, constraint, cfg=cfg)
    
    # Run evaluation
    logger.message("Starting evaluation...")
    eval_result = solver.eval()
    logger.message(f"Evaluation completed. Result: {eval_result}")


def inference(cfg: DictConfig):
    # Parse automatic configuration parameters
    if cfg.MODEL.n_f == "auto":
        cfg.MODEL.n_f = cfg.DATA.dimension * 2 + 2
    if cfg.MODEL.ndim == "auto":
        cfg.MODEL.ndim = cfg.DATA.dimension
    if cfg.MODEL.arch == "HGN":
        cfg.MODEL.output_keys = ["derivative"]
    
    # Create model based on architecture
    if cfg.MODEL.arch == "OGN":
        model = OGN(
            input_keys=tuple(cfg.MODEL.input_keys),
            output_keys=tuple(cfg.MODEL.output_keys),
            n_f=cfg.MODEL.n_f,
            msg_dim=cfg.MODEL.msg_dim,
            ndim=cfg.MODEL.ndim,
            dt=cfg.MODEL.dt,
            hidden=cfg.MODEL.hidden,
            aggr=cfg.MODEL.aggr
        )
    elif cfg.MODEL.arch == "VarOGN":
        model = VarOGN(
            input_keys=tuple(cfg.MODEL.input_keys),
            output_keys=tuple(cfg.MODEL.output_keys),
            n_f=cfg.MODEL.n_f,
            msg_dim=cfg.MODEL.msg_dim,
            ndim=cfg.MODEL.ndim,
            dt=cfg.MODEL.dt,
            hidden=cfg.MODEL.hidden,
            aggr=cfg.MODEL.aggr
        )
    elif cfg.MODEL.arch == "HGN":
        model = HGN(
            input_keys=tuple(cfg.MODEL.input_keys),
            output_keys=tuple(cfg.MODEL.output_keys),
            n_f=cfg.MODEL.n_f,
            ndim=cfg.MODEL.ndim,
            hidden=cfg.MODEL.hidden,
            aggr=cfg.MODEL.aggr
        )
    else:
        raise ValueError(f"Unsupported model architecture: {cfg.MODEL.arch}")
    
    # Create PaddleScience Solver
    solver = ppsci.solver.Solver(model, cfg=cfg)

    # Generate test data
    logger.message("Generating inference data using simulate.py...")
    dt = 1e-2
    for sim_set in cfg.DATATYPE_TYPE:
        if sim_set['sim'] == cfg.DATA.type:
            dt = sim_set['dt'][0]
            break
    
    sim_dataset = SimulationDataset(
        sim=cfg.DATA.type,
        n=cfg.DATA.num_nodes,
        dim=cfg.DATA.dimension,
        dt=dt,
        nt=100,
        seed=cfg.seed
    )
    sim_dataset.simulate(1)

    # Prepare input data (first time step)
    node_features = sim_dataset.data[0, 0]
    edge_index = get_edge_index(cfg.DATA.num_nodes, cfg.DATA.type)
    
    input_dict = {
        "node_features": paddle.to_tensor(node_features, dtype='float32').unsqueeze(0),
        "edge_index": edge_index.unsqueeze(0)
    }

    # Run inference
    logger.message("Running inference...")
    output_dict = solver.predict(input_dict, return_numpy=True)

    # Display results
    if cfg.MODEL.arch == "HGN":
        dq_dt = output_dict["derivative"][0, :cfg.DATA.dimension]
        dv_dt = output_dict["derivative"][0, cfg.DATA.dimension:]
        logger.message(f"HGN inference - dq/dt: {dq_dt}, dv/dt (acceleration): {dv_dt}")
    else:
        acceleration = output_dict[list(cfg.MODEL.output_keys)[0]][0]
        logger.message(f"{cfg.MODEL.arch} inference - acceleration: {acceleration}")


def export(cfg: DictConfig):
    # Parse automatic configuration parameters
    if cfg.MODEL.n_f == "auto":
        cfg.MODEL.n_f = cfg.DATA.dimension * 2 + 2
    if cfg.MODEL.ndim == "auto":
        cfg.MODEL.ndim = cfg.DATA.dimension
    if cfg.MODEL.arch == "HGN":
        cfg.MODEL.output_keys = ["derivative"]
    
    # Create model based on architecture
    if cfg.MODEL.arch == "OGN":
        model = OGN(
            input_keys=tuple(cfg.MODEL.input_keys),
            output_keys=tuple(cfg.MODEL.output_keys),
            n_f=cfg.MODEL.n_f,
            msg_dim=cfg.MODEL.msg_dim,
            ndim=cfg.MODEL.ndim,
            dt=cfg.MODEL.dt,
            hidden=cfg.MODEL.hidden,
            aggr=cfg.MODEL.aggr
        )
    elif cfg.MODEL.arch == "VarOGN":
        model = VarOGN(
            input_keys=tuple(cfg.MODEL.input_keys),
            output_keys=tuple(cfg.MODEL.output_keys),
            n_f=cfg.MODEL.n_f,
            msg_dim=cfg.MODEL.msg_dim,
            ndim=cfg.MODEL.ndim,
            dt=cfg.MODEL.dt,
            hidden=cfg.MODEL.hidden,
            aggr=cfg.MODEL.aggr
        )
    elif cfg.MODEL.arch == "HGN":
        model = HGN(
            input_keys=tuple(cfg.MODEL.input_keys),
            output_keys=tuple(cfg.MODEL.output_keys),
            n_f=cfg.MODEL.n_f,
            ndim=cfg.MODEL.ndim,
            hidden=cfg.MODEL.hidden,
            aggr=cfg.MODEL.aggr
        )
    else:
        raise ValueError(f"Unsupported model architecture: {cfg.MODEL.arch}")
    
    # Create PaddleScience Solver
    solver = ppsci.solver.Solver(model, cfg=cfg)
    
    # Define input specifications for export
    from paddle.static import InputSpec
    input_spec = [
        {
            "node_features": InputSpec([None, cfg.DATA.num_nodes, cfg.MODEL.n_f], "float32", "node_features"),
            "edge_index": InputSpec([None, 2, cfg.DATA.num_nodes * (cfg.DATA.num_nodes - 1)], "int64", "edge_index")
        }
    ]
    
    # Export model to static graph
    logger.message(f"Exporting model to {cfg.INFER.export_path}")
    solver.export(input_spec, cfg.INFER.export_path, with_onnx=False)
    logger.message("Model export completed!")


@hydra.main(version_base=None, config_path="./conf", config_name="config.yaml")
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
        raise ValueError(f"Unsupported mode: {cfg.mode}")


if __name__ == "__main__":
    main()