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
from typing import List
from typing import Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import paddle
from omegaconf import DictConfig
from simulate import SimulationDataset

import ppsci
from ppsci.arch.symbolic_gn import HGN
from ppsci.arch.symbolic_gn import OGN
from ppsci.arch.symbolic_gn import VarOGN
from ppsci.arch.symbolic_gn import get_edge_index
from ppsci.utils import logger


def create_ppsci_model(
    model_type: str = "OGN",
    input_keys: List[str] = ["x", "edge_index"],
    output_keys: List[str] = ["acceleration"],
    n_f: int = 6,
    msg_dim: int = 100,
    ndim: int = 2,
    hidden: int = 300,
    edge_index: Optional[np.ndarray] = None,
    l1_strength: float = 0.0,
):
    if model_type == "OGN":
        model = OGN(
            input_keys=input_keys,
            output_keys=output_keys,
            n_f=n_f,
            msg_dim=msg_dim,
            ndim=ndim,
            hidden=hidden,
            edge_index=edge_index,
            l1_strength=l1_strength,
        )
    elif model_type == "HGN":
        model = HGN(
            input_keys=input_keys,
            output_keys=output_keys,
            n_f=n_f,
            ndim=ndim,
            hidden=hidden,
            edge_index=edge_index,
        )
    elif model_type == "VarOGN":
        model = VarOGN(
            input_keys=input_keys,
            output_keys=output_keys,
            n_f=n_f,
            msg_dim=msg_dim,
            ndim=ndim,
            hidden=hidden,
            edge_index=edge_index,
            l1_strength=l1_strength,
        )
    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    return model


def replicate_array(arr, n):
    return np.tile(arr, (n,) + (1,) * arr.ndim)


def create_loss_function(cfg):
    def loss_function(output_dict, label_dict, weight_dict=None):
        loss_type = getattr(cfg.TRAIN.loss, "type", "MAE")

        output_key = list(cfg.MODEL.output_keys)[0]

        target = label_dict[output_key]
        pred = output_dict[output_key]
        if loss_type == "MSE":
            base_loss = paddle.mean(paddle.square(pred - target))
        else:
            base_loss = paddle.mean(paddle.abs(pred - target))

        if "l1_regularization" in output_dict:
            base_loss = base_loss + output_dict["l1_regularization"]

        return {output_key: base_loss}

    return loss_function


def train(cfg):
    ppsci.utils.misc.set_random_seed(cfg.seed)

    if cfg.MODEL.arch == "HGN":
        cfg.MODEL.output_keys = ["acceleration"]

    sim = SimulationDataset(
        sim=cfg.DATA.type,
        n=cfg.DATA.num_nodes,
        dim=cfg.DATA.dimension,
        nt=cfg.DATA.time_steps,
        dt=cfg.DATA.time_step_size,
    )
    sim.simulate(cfg.DATA.num_samples)
    accel_data = sim.get_acceleration()

    X_list = []
    y_list = []
    for sample_idx in range(cfg.DATA.num_samples):
        for t in range(0, sim.data.shape[1], cfg.DATA.sample_interval):
            X_list.append(sim.data[sample_idx, t])
            y_list.append(accel_data[sample_idx, t])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    if cfg.MODEL.n_f == "auto":
        cfg.MODEL.n_f = cfg.DATA.dimension * 2 + 2
    if cfg.MODEL.ndim == "auto":
        cfg.MODEL.ndim = cfg.DATA.dimension

    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    edge_index = get_edge_index(cfg.DATA.num_nodes, cfg.DATA.type)
    edge_index_train = replicate_array(edge_index, len(X_train))
    edge_index_val = replicate_array(edge_index, len(X_val))

    train_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": {"x": X_train, "edge_index": edge_index_train},
                "label": {cfg.MODEL.output_keys[0]: y_train},
            },
            "batch_size": cfg.TRAIN.batch_size,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": True,
            },
        },
        create_loss_function(cfg),
        name="sup_constraint",
    )
    eval_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": {"x": X_val, "edge_index": edge_index_val},
                "label": {cfg.MODEL.output_keys[0]: y_val},
            },
            "batch_size": cfg.TRAIN.batch_size,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": False,
            },
        },
        create_loss_function(cfg),
        name="eval_constraint",
    )
    constraint = {
        train_constraint.name: train_constraint,
        eval_constraint.name: eval_constraint,
    }
    l1_strength = (
        cfg.MODEL.l1_strength
        if cfg.MODEL.regularization_type == "l1" and cfg.MODEL.arch in ["OGN", "VarOGN"]
        else 0.0
    )
    model = create_ppsci_model(
        model_type=cfg.MODEL.arch,
        n_f=cfg.MODEL.n_f,
        msg_dim=cfg.MODEL.msg_dim,
        ndim=cfg.MODEL.ndim,
        hidden=cfg.MODEL.hidden,
        edge_index=edge_index,
        l1_strength=l1_strength,
    )

    batch_per_epoch = int(train_size / cfg.TRAIN.batch_size)
    if cfg.TRAIN.lr_scheduler.name == "OneCycleLR":
        lr_scheduler = paddle.optimizer.lr.OneCycleLR(
            max_learning_rate=cfg.TRAIN.lr_scheduler.max_learning_rate,
            total_steps=int(cfg.TRAIN.epochs * batch_per_epoch),
            divide_factor=cfg.TRAIN.lr_scheduler.final_div_factor,
        )
        lr_scheduler.by_epoch = False
    else:
        lr_scheduler = paddle.optimizer.lr.ExponentialDecay(
            learning_rate=cfg.TRAIN.optimizer.learning_rate, gamma=0.9
        )
        lr_scheduler.by_epoch = True

    optimizer = paddle.optimizer.Adam(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=cfg.TRAIN.optimizer.weight_decay,
    )

    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        epochs=cfg.TRAIN.epochs,
        save_freq=cfg.TRAIN.save_freq,
    )

    solver.train()
    solver.plot_loss_history(by_epoch=True, smooth_step=1)
    paddle.save(model.state_dict(), os.path.join(f"{cfg.MODEL.arch}.pdparams"))


def evaluate(cfg: DictConfig):
    if cfg.MODEL.n_f == "auto":
        cfg.MODEL.n_f = cfg.DATA.dimension * 2 + 2
    if cfg.MODEL.ndim == "auto":
        cfg.MODEL.ndim = cfg.DATA.dimension

    edge_index = get_edge_index(cfg.DATA.num_nodes, cfg.DATA.type)
    l1_strength = (
        cfg.MODEL.l1_strength
        if cfg.MODEL.regularization_type == "l1" and cfg.MODEL.arch in ["OGN", "VarOGN"]
        else 0.0
    )

    model = create_ppsci_model(
        model_type=cfg.MODEL.arch,
        n_f=cfg.MODEL.n_f,
        msg_dim=cfg.MODEL.msg_dim,
        ndim=cfg.MODEL.ndim,
        hidden=cfg.MODEL.hidden,
        edge_index=edge_index,
        l1_strength=l1_strength,
    )

    ppsci.utils.save_load.load_pretrain(
        model,
        cfg.EVAL.pretrained_model_path,
    )

    sim = SimulationDataset(
        sim=cfg.DATA.type,
        n=cfg.DATA.num_nodes,
        dim=cfg.DATA.dimension,
        nt=cfg.DATA.time_steps,
        dt=cfg.DATA.time_step_size,
    )
    sim.simulate(cfg.DATA.num_samples)
    accel_data = sim.get_acceleration()

    sample_indices = [0, 1] if cfg.DATA.num_samples > 1 else [0]

    for sample_idx in sample_indices:
        sample_data = sim.data[sample_idx, 0:1]
        true_accel = accel_data[sample_idx, 0:1]

        input_dict = {
            "x": paddle.to_tensor(sample_data, dtype="float32"),
            "edge_index": paddle.to_tensor(edge_index, dtype="int64"),
        }

        with paddle.no_grad():
            pred_output = model(input_dict)
            pred_accel = pred_output[cfg.MODEL.output_keys[0]]

        error = np.mean(np.abs(pred_accel.numpy() - true_accel))
        logger.info(f"Sample {sample_idx} - MAE error: {error:.6f}")

        rel_error = np.linalg.norm(pred_accel.numpy() - true_accel) / np.linalg.norm(
            true_accel
        )
        logger.info(f"Sample {sample_idx} - Relative error: {rel_error:.6f}")

        plt.figure(figsize=(10, 8))
        sim.plot(sample_idx, animate=False, plot_size=True, s_size=2)
        plt.title(
            f"{cfg.DATA.type.capitalize()} System - Sample {sample_idx}\nMAE: {error:.4f}, Rel Error: {rel_error:.4f}"
        )
        plt.tight_layout()

        plot_path = os.path.join(cfg.output_dir, f"evaluation_sample_{sample_idx}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Evaluation plot saved to {plot_path}")


def export(cfg: DictConfig):
    if cfg.MODEL.arch == "HGN":
        raise ValueError("HGN is not supported for export")

    ppsci.utils.misc.set_random_seed(cfg.seed)

    if cfg.MODEL.n_f == "auto":
        cfg.MODEL.n_f = cfg.DATA.dimension * 2 + 2
    if cfg.MODEL.ndim == "auto":
        cfg.MODEL.ndim = cfg.DATA.dimension
    edge_index = get_edge_index(cfg.DATA.num_nodes, cfg.DATA.type)
    l1_strength = (
        cfg.MODEL.l1_strength
        if cfg.MODEL.regularization_type == "l1" and cfg.MODEL.arch in ["OGN", "VarOGN"]
        else 0.0
    )
    model = create_ppsci_model(
        model_type=cfg.MODEL.arch,
        n_f=cfg.MODEL.n_f,
        msg_dim=cfg.MODEL.msg_dim,
        ndim=cfg.MODEL.ndim,
        hidden=cfg.MODEL.hidden,
        edge_index=edge_index,
        l1_strength=l1_strength,
    )

    solver = ppsci.solver.Solver(
        model,
        pretrained_model_path=cfg.INFER.pretrained_model_path,
    )

    from paddle.static import InputSpec

    input_spec = [
        {
            key: (
                InputSpec(
                    [None, cfg.DATA.num_nodes, cfg.MODEL.n_f], "float32", name=key
                )
                if key == "x"
                else InputSpec([2, 30], "int64", name=key)
            )
            for key in model.input_keys
        },
    ]
    solver.export(input_spec, cfg.INFER.export_path)


def inference(cfg: DictConfig):
    from deploy import python_infer

    try:
        predictor = python_infer.GeneralPredictor(cfg)
        use_predictor = True
    except Exception as e:
        logger.error(f"GeneralPredictor failed: {e}")
        logger.info("Switching to direct model inference...")
        use_predictor = False

    sim = SimulationDataset(
        sim=cfg.DATA.type,
        n=cfg.DATA.num_nodes,
        dim=cfg.DATA.dimension,
        nt=cfg.DATA.time_steps,
        dt=cfg.DATA.time_step_size,
    )
    sim.simulate(cfg.DATA.num_samples)

    sample_idx = 0
    sample_data = sim.data[sample_idx, 0]
    edge_index = get_edge_index(cfg.DATA.num_nodes, cfg.DATA.type)

    if use_predictor:
        input_dict = {"x": sample_data, "edge_index": edge_index}
        output_dict = predictor.predict(input_dict, cfg.INFER.batch_size)
        pred_acceleration = output_dict[cfg.MODEL.output_keys[0]]
    else:
        if cfg.MODEL.n_f == "auto":
            cfg.MODEL.n_f = cfg.DATA.dimension * 2 + 2
        if cfg.MODEL.ndim == "auto":
            cfg.MODEL.ndim = cfg.DATA.dimension

        l1_strength = (
            cfg.MODEL.l1_strength
            if cfg.MODEL.regularization_type == "l1"
            and cfg.MODEL.arch in ["OGN", "VarOGN"]
            else 0.0
        )

        model = create_ppsci_model(
            model_type=cfg.MODEL.arch,
            n_f=cfg.MODEL.n_f,
            msg_dim=cfg.MODEL.msg_dim,
            ndim=cfg.MODEL.ndim,
            hidden=cfg.MODEL.hidden,
            edge_index=edge_index,
            l1_strength=l1_strength,
        )

        ppsci.utils.save_load.load_pretrain(
            model,
            cfg.INFER.pretrained_model_path,
        )

        input_dict = {
            "x": paddle.to_tensor(sample_data, dtype="float32"),
            "edge_index": paddle.to_tensor(edge_index, dtype="int64"),
        }

        with paddle.no_grad():
            pred_output = model(input_dict)
            pred_acceleration = pred_output[cfg.MODEL.output_keys[0]].numpy()

    accel_data = sim.get_acceleration()
    true_acceleration = accel_data[sample_idx, 0]

    error = np.mean(np.abs(pred_acceleration - true_acceleration))
    logger.info(f"Inference error (MAE): {error:.6f}")

    rel_error = np.linalg.norm(pred_acceleration - true_acceleration) / np.linalg.norm(
        true_acceleration
    )
    logger.info(f"Inference relative error: {rel_error:.6f}")

    plt.figure(figsize=(10, 8))
    sim.plot(sample_idx, animate=False, plot_size=True, s_size=2)
    plt.title(
        f"{cfg.DATA.type.capitalize()} System - Inference\nMAE: {error:.4f}, Rel Error: {rel_error:.4f}"
    )
    plt.tight_layout()

    plot_path = os.path.join(cfg.output_dir, "inference.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Inference plot saved to {plot_path}")


@hydra.main(version_base=None, config_path="./conf", config_name="config_hgn")
def main(cfg: DictConfig) -> None:
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    elif cfg.mode == "export":
        export(cfg)
    elif cfg.mode == "infer":
        inference(cfg)
    else:
        raise ValueError(
            "cfg.mode should in [train, eval, export, infer], but got {}".format(
                cfg.mode
            )
        )


if __name__ == "__main__":
    main()
