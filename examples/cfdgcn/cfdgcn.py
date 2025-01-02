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
from typing import Dict
from typing import List

import hydra
import paddle
import pgl
import su2paddle
import utils
from omegaconf import DictConfig
from paddle.nn import functional as F

import ppsci
from ppsci.utils import logger


def train_mse_func(
    output_dict: Dict[str, "paddle.Tensor"],
    label_dict: Dict[str, "pgl.Graph"],
    *args,
) -> paddle.Tensor:
    return {"pred": F.mse_loss(output_dict["pred"], label_dict["label"].y)}


def eval_rmse_func(
    output_dict: Dict[str, List["paddle.Tensor"]],
    label_dict: Dict[str, List["pgl.Graph"]],
    *args,
) -> Dict[str, paddle.Tensor]:
    mse_losses = [
        F.mse_loss(pred, label.y)
        for (pred, label) in zip(output_dict["pred"], label_dict["label"])
    ]
    return {"RMSE": (sum(mse_losses) / len(mse_losses)) ** 0.5}


def train(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", os.path.join(cfg.output_dir, "train.log"), "info")

    # set dataloader config
    train_dataloader_cfg = {
        "dataset": {
            "name": "MeshAirfoilDataset",
            "input_keys": ("input",),
            "label_keys": ("label",),
            "data_dir": cfg.TRAIN_DATA_DIR,
            "mesh_graph_path": cfg.TRAIN_MESH_GRAPH_PATH,
            "transpose_edges": True,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "num_workers": 1,
    }

    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        output_expr={"pred": lambda out: out["pred"]},
        loss=ppsci.loss.FunctionalLoss(train_mse_func),
        name="Sup",
    )
    # wrap constraints together
    constraint = {sup_constraint.name: sup_constraint}
    process_sim = sup_constraint.data_loader.dataset._preprocess
    fine_marker_dict = sup_constraint.data_loader.dataset.marker_dict

    # set model
    model = ppsci.arch.CFDGCN(
        **cfg.MODEL,
        process_sim=process_sim,
        fine_marker_dict=fine_marker_dict,
        su2_module=su2paddle.SU2Module,
    )

    # set optimizer
    optimizer = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model)

    # set validator
    eval_dataloader_cfg = {
        "dataset": {
            "name": "MeshAirfoilDataset",
            "input_keys": ("input",),
            "label_keys": ("label",),
            "data_dir": cfg.EVAL_DATA_DIR,
            "mesh_graph_path": cfg.EVAL_MESH_GRAPH_PATH,
            "transpose_edges": True,
        },
        "batch_size": cfg.EVAL.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }
    rmse_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(train_mse_func),
        output_expr={"pred": lambda out: out["pred"].unsqueeze(0)},
        metric={"RMSE": ppsci.metric.FunctionalMetric(eval_rmse_func)},
        name="RMSE_validator",
    )
    validator = {rmse_validator.name: rmse_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        None,
        cfg.TRAIN.epochs,
        cfg.TRAIN.iters_per_epoch,
        save_freq=cfg.TRAIN.save_freq,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_freq=cfg.TRAIN.eval_freq,
        validator=validator,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
        checkpoint_path=cfg.TRAIN.checkpoint_path,
    )

    # train model
    solver.train()

    # visualize prediction
    with solver.no_grad_context_manager(True):
        for index, (input_, label, _) in enumerate(rmse_validator.data_loader):
            truefield = label["label"].y
            prefield = model(input_)
            utils.log_images(
                input_["input"].pos,
                prefield["pred"],
                truefield,
                rmse_validator.data_loader.dataset.elems_list,
                index,
                "cylinder",
            )


def export(cfg: DictConfig):
    # set model
    model = ppsci.arch.CFDGCN(
        **cfg.MODEL,
        process_sim=process_sim,
        fine_marker_dict=fine_marker_dict,
        su2_module=su2paddle.SU2Module,
    )

    solver = ppsci.solver.Solver(
        model, pretrained_model_path=cfg.EXPORT.pretrained_model_path
    )

    # export model
    from paddle.static import InputSpec

    input_spec = [
        {key: InputSpec([None, 1], "float32", name=key) for key in model.input_keys},
    ]

    solver.export(input_spec, cfg.INFER.export_path)


def inference(cfg: DictConfig):

    # 初始化预测器
    from deploy.python_infer import pinn_predictor

    predictor = pinn_predictor.PINNPredictor(cfg)

    # 设置 dataloader 配置
    infer_dataloader_cfg = {
        "dataset": {
            "name": "MeshAirfoilDataset",
            "input_keys": ("input",),
            "label_keys": ("label",),
            "data_dir": cfg.INFER_DATA_DIR,
            "mesh_graph_path": cfg.INFER_MESH_GRAPH_PATH,
            "transpose_edges": True,
        },
        "batch_size": cfg.INFER.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }

    # 初始化数据集和 dataloader
    infer_dataset = ppsci.data.MeshAirfoilDataset(
        input_keys=infer_dataloader_cfg["dataset"]["input_keys"],
        label_keys=infer_dataloader_cfg["dataset"]["label_keys"],
        data_dir=infer_dataloader_cfg["dataset"]["data_dir"],
        mesh_graph_path=infer_dataloader_cfg["dataset"]["mesh_graph_path"],
        transpose_edges=infer_dataloader_cfg["dataset"]["transpose_edges"],
    )
    infer_dataloader = ppsci.dataloader.DataLoader(
        infer_dataset,
        batch_size=infer_dataloader_cfg["batch_size"],
        shuffle=infer_dataloader_cfg["sampler"]["shuffle"],
        drop_last=infer_dataloader_cfg["sampler"]["drop_last"],
        num_workers=infer_dataloader_cfg.get("num_workers", 1),
    )

    # 进行推理并可视化结果
    with predictor.no_grad_context_manager(True):
        for index, (input_dict, label_dict, _) in enumerate(infer_dataloader):
            # 获取真实值
            truefield = label_dict["label"].y
            # 模型预测
            output_dict = predictor.predict(input_dict, cfg.INFER.batch_size)
            prefield = output_dict["pred"]

            # 可视化结果
            utils.log_images(
                input_dict["input"].pos,
                prefield,
                truefield,
                infer_dataset.elems_list,
                index,
                "cylinder",
            )


def evaluate(cfg: DictConfig):
    # set dataloader config
    train_dataloader_cfg = {
        "dataset": {
            "name": "MeshAirfoilDataset",
            "input_keys": ("input",),
            "label_keys": ("label",),
            "data_dir": cfg.TRAIN_DATA_DIR,
            "mesh_graph_path": cfg.TRAIN_MESH_GRAPH_PATH,
            "transpose_edges": True,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "num_workers": 1,
    }

    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        output_expr={"pred": lambda out: out["pred"]},
        loss=ppsci.loss.FunctionalLoss(train_mse_func),
        name="Sup",
    )

    process_sim = sup_constraint.data_loader.dataset._preprocess
    fine_marker_dict = sup_constraint.data_loader.dataset.marker_dict

    # set airfoil model
    model = ppsci.arch.CFDGCN(
        **cfg.MODEL,
        process_sim=process_sim,
        fine_marker_dict=fine_marker_dict,
        su2_module=su2paddle.SU2Module,
    )

    # set validator
    eval_dataloader_cfg = {
        "dataset": {
            "name": "MeshAirfoilDataset",
            "input_keys": ("input",),
            "label_keys": ("label",),
            "data_dir": cfg.EVAL_DATA_DIR,
            "mesh_graph_path": cfg.EVAL_MESH_GRAPH_PATH,
            "transpose_edges": True,
        },
        "batch_size": cfg.EVAL.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }
    rmse_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(train_mse_func),
        output_expr={"pred": lambda out: out["pred"].unsqueeze(0)},
        metric={"RMSE": ppsci.metric.FunctionalMetric(eval_rmse_func)},
        name="RMSE_validator",
    )
    validator = {rmse_validator.name: rmse_validator}

    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        log_freq=cfg.log_freq,
        seed=cfg.seed,
        validator=validator,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )

    # evaluate model
    solver.eval()

    # visualize prediction
    with solver.no_grad_context_manager(True):
        for index, (input_, label, _) in enumerate(rmse_validator.data_loader):
            truefield = label["label"].y
            prefield = model(input_)
            utils.log_images(
                input_["input"].pos,
                prefield["pred"],
                truefield,
                rmse_validator.data_loader.dataset.elems_list,
                index,
                "cylinder",
            )


@hydra.main(version_base=None, config_path="./conf", config_name="cfdgcn.yaml")
def main(cfg: DictConfig):
    su2paddle.activate_su2_mpi(remove_temp_files=True)
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
            f"cfg.mode should in ['train', 'eval', 'export', 'infer'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()
