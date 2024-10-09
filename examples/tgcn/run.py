import os

import hydra
from omegaconf import DictConfig

import ppsci
from ppsci.arch.tgcn import TGCN
from ppsci.data.dataset.pems_dataset import get_edge_index


def train(cfg: DictConfig):
    # set train dataloader config
    train_dataloader_cfg = {
        "dataset": {
            "name": "PEMSDataset",
            "file_path": cfg.data_path,
            "split": "train",
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.MODEL.label_keys,
            "norm_input": cfg.norm_input,
            "norm_label": cfg.norm_label,
            "input_len": cfg.input_len,
            "label_len": cfg.label_len,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": True,
        },
        "batch_size": cfg.TRAIN.batch_size,
    }

    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg, ppsci.loss.L1Loss(), name="train"
    )
    constraint = {sup_constraint.name: sup_constraint}

    # set eval dataloader config
    eval_dataloader_cfg = {
        "dataset": {
            "name": "PEMSDataset",
            "file_path": cfg.data_path,
            "split": "val",
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.MODEL.label_keys,
            "norm_input": cfg.norm_input,
            "norm_label": cfg.norm_label,
            "input_len": cfg.input_len,
            "label_len": cfg.label_len,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "batch_size": cfg.EVAL.batch_size,
    }

    # set validator
    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        ppsci.loss.L1Loss(),
        metric={"MAE": ppsci.metric.MAE(), "RMSE": ppsci.metric.RMSE()},
        name="val",
    )
    validator = {sup_validator.name: sup_validator}

    # get adj
    _, _, adj = get_edge_index(cfg.data_path, reduce=cfg.reduce)
    # set model
    model = TGCN(adj=adj, cfg=cfg)
    # init optimizer
    optimizer = ppsci.optimizer.Adam(learning_rate=cfg.TRAIN.learning_rate)(model)
    # set iters_per_epoch by dataloader length
    iters_per_epoch = len(sup_constraint.data_loader)

    # initialize solver
    solver = ppsci.solver.Solver(
        model=model,
        constraint=constraint,
        output_dir=cfg.output_dir,
        optimizer=optimizer,
        # lr_scheduler=None,
        epochs=cfg.TRAIN.epochs,
        iters_per_epoch=iters_per_epoch,
        # update_freq=1,
        # save_freq=0,
        log_freq=cfg.log_freq,
        eval_during_train=True,
        # eval_freq=1,
        # start_eval_epoch=1,
        seed=cfg.seed,
        device=cfg.device,
        validator=validator,
        pretrained_model_path=cfg.TRAIN.pretrained_model_path,
        # compute_metric_by_batch=False,
        eval_with_no_grad=True,
    )
    # train model
    solver.train()
    # evaluate after training
    eval(cfg)


def eval(cfg: DictConfig):
    # set eval dataloader config
    test_dataloader_cfg = {
        "dataset": {
            "name": "PEMSDataset",
            "file_path": cfg.data_path,
            "split": "test",
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.MODEL.label_keys,
            "norm_input": cfg.norm_input,
            "norm_label": cfg.norm_label,
            "input_len": cfg.input_len,
            "label_len": cfg.label_len,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "batch_size": cfg.EVAL.batch_size,
    }

    # set validator
    sup_validator = ppsci.validate.SupervisedValidator(
        test_dataloader_cfg,
        ppsci.loss.L1Loss(),
        metric={"MAE": ppsci.metric.MAE(), "RMSE": ppsci.metric.RMSE()},
        name="test",
    )
    validator = {sup_validator.name: sup_validator}

    # get adj
    _, _, adj = get_edge_index(cfg.data_path, reduce=cfg.reduce)
    # set model
    model = TGCN(adj=adj, cfg=cfg)
    # best model
    if cfg.mode == "train":
        best_model_path = os.path.join(
            cfg.output_dir, "checkpoints", "best_model.pdparams"
        )  # call in train()
    else:
        best_model_path = cfg.data_name + "_pretrained_model.pdparams"

    # initialize solver
    solver = ppsci.solver.Solver(
        model=model,
        # constraint=None,
        output_dir=cfg.output_dir,
        # optimizer=None,
        # lr_scheduler=None,
        # epochs=5,
        # iters_per_epoch=20,
        # update_freq=1,
        # save_freq=0,
        log_freq=cfg.log_freq,
        # eval_during_train=False,
        # eval_freq=1,
        # start_eval_epoch=1,
        seed=cfg.seed,
        device=cfg.device,
        validator=validator,
        pretrained_model_path=best_model_path,
        # compute_metric_by_batch=False,
        eval_with_no_grad=True,
    )
    # evaluate
    solver.eval()


@hydra.main(version_base=None, config_path="./conf", config_name="run.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        eval(cfg)
    else:
        raise ValueError(
            "cfg.mode should in [train, eval], but got {}".format(cfg.mode)
        )


if __name__ == "__main__":
    main()
