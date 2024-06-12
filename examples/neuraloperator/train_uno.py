import hydra
import metric
import numpy as np
import paddle
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger


def train(cfg: DictConfig):
    # set train dataloader config
    train_dataloader_cfg = {
        "dataset": {
            "name": "DarcyFlowDataset",
            "data_dir": cfg.FILE_PATH,
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.DATASET.label_keys,
            "train_resolution": cfg.DATASET.train_resolution,
            "test_resolutions": cfg.DATASET.test_resolutions,
            "grid_boundaries": cfg.DATASET.grid_boundaries,
            "encode_input": cfg.DATASET.encode_input,
            "encode_output": cfg.DATASET.encode_output,
            "encoding": cfg.DATASET.encoding,
            "channel_dim": cfg.DATASET.channel_dim,
            "data_split": "train",
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": True,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "num_workers": 0,
    }

    # set loss
    l2loss = metric.LpLoss_train(d=2, p=2)
    h1loss = metric.H1Loss_train(d=2)
    if cfg.TRAIN.training_loss == "l2":
        train_loss = l2loss
    if cfg.TRAIN.training_loss == "h1":
        train_loss = h1loss

    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(train_loss),
        name="Sup",
    )
    constraint = {sup_constraint.name: sup_constraint}

    # set iters_per_epoch by dataloader length
    ITERS_PER_EPOCH = len(sup_constraint.data_loader)
    # set eval dataloader config
    eval_dataloader_cfg_16 = {
        "dataset": {
            "name": "DarcyFlowDataset",
            "data_dir": cfg.FILE_PATH,
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.DATASET.label_keys,
            "train_resolution": cfg.DATASET.train_resolution,
            "test_resolutions": cfg.DATASET.test_resolutions,
            "grid_boundaries": cfg.DATASET.grid_boundaries,
            "encode_input": cfg.DATASET.encode_input,
            "encode_output": cfg.DATASET.encode_output,
            "encoding": cfg.DATASET.encoding,
            "channel_dim": cfg.DATASET.channel_dim,
            "data_split": "test_16x16",
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "batch_size": cfg.EVAL.batch_size,
        "num_workers": 0,
    }

    eval_dataloader_cfg_32 = {
        "dataset": {
            "name": "DarcyFlowDataset",
            "data_dir": cfg.FILE_PATH,
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.DATASET.label_keys,
            "train_resolution": cfg.DATASET.train_resolution,
            "test_resolutions": cfg.DATASET.test_resolutions,
            "grid_boundaries": cfg.DATASET.grid_boundaries,
            "encode_input": cfg.DATASET.encode_input,
            "encode_output": cfg.DATASET.encode_output,
            "encoding": cfg.DATASET.encoding,
            "channel_dim": cfg.DATASET.channel_dim,
            "data_split": "test_32x32",
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "batch_size": cfg.EVAL.batch_size,
        "num_workers": 0,
    }

    h1_eval_metric = metric.H1Loss(d=2)
    l2_eval_metric = metric.LpLoss(d=2, p=2)
    sup_validator_16 = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg_16,
        loss=ppsci.loss.FunctionalLoss(train_loss),
        metric={
            "h1": ppsci.metric.FunctionalMetric(h1_eval_metric),
            "l2": ppsci.metric.FunctionalMetric(l2_eval_metric),
        },
        name="Sup_Validator_16x16",
    )

    sup_validator_32 = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg_32,
        loss=ppsci.loss.FunctionalLoss(train_loss),
        metric={
            "h1": ppsci.metric.FunctionalMetric(h1_eval_metric),
            "l2": ppsci.metric.FunctionalMetric(l2_eval_metric),
        },
        name="Sup_Validator_32x32",
    )

    validator = {
        sup_validator_16.name: sup_validator_16,
        sup_validator_32.name: sup_validator_32,
    }

    model = ppsci.arch.UNONet(
        **cfg.MODEL,
    )
    # init optimizer and lr scheduler
    if cfg.TRAIN.lr_scheduler.type == "ReduceOnPlateau":
        lr_scheduler = paddle.optimizer.lr.ReduceOnPlateau(
            learning_rate=cfg.TRAIN.lr_scheduler.learning_rate,
            factor=cfg.TRAIN.lr_scheduler.gamma,
            patience=cfg.TRAIN.lr_scheduler.scheduler_patience,
            mode="min",
        )
    elif cfg.TRAIN.lr_scheduler.type == "CosineAnnealingDecay":
        lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=cfg.TRAIN.lr_scheduler.learning_rate,
            T_max=cfg.TRAIN.lr_scheduler.scheduler_T_max,
        )()
    elif cfg.TRAIN.lr_scheduler.type == "StepDecay":
        lr_scheduler = ppsci.optimizer.lr_scheduler.Step(
            epochs=cfg.TRAIN.lr_scheduler.epochs,
            iters_per_epoch=ITERS_PER_EPOCH,
            learning_rate=cfg.TRAIN.lr_scheduler.learning_rate,
            step_size=cfg.TRAIN.lr_scheduler.step_size,
            gamma=cfg.TRAIN.lr_scheduler.gamma,
        )()
    else:
        raise ValueError(f"Got scheduler={cfg.TRAIN.lr_scheduler.type}")
    optimizer = ppsci.optimizer.Adam(lr_scheduler, weight_decay=cfg.TRAIN.wd)(model)

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        lr_scheduler,
        cfg.TRAIN.epochs,
        ITERS_PER_EPOCH,
        eval_during_train=cfg.TRAIN.eval_during_train,
        seed=cfg.seed,
        validator=validator,
        compute_metric_by_batch=cfg.EVAL.compute_metric_by_batch,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
        pretrained_model_path=cfg.TRAIN.pretrained_model_path,
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()


def evaluate(cfg: DictConfig):
    # set eval dataloader config
    eval_dataloader_cfg_16 = {
        "dataset": {
            "name": "DarcyFlowDataset",
            "data_dir": cfg.FILE_PATH,
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.DATASET.label_keys,
            "train_resolution": cfg.DATASET.train_resolution,
            "test_resolutions": cfg.DATASET.test_resolutions,
            "grid_boundaries": cfg.DATASET.grid_boundaries,
            "encode_input": cfg.DATASET.encode_input,
            "encode_output": cfg.DATASET.encode_output,
            "encoding": cfg.DATASET.encoding,
            "channel_dim": cfg.DATASET.channel_dim,
            "data_split": "test_16x16",
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "batch_size": cfg.EVAL.batch_size,
        "num_workers": 0,
    }

    eval_dataloader_cfg_32 = {
        "dataset": {
            "name": "DarcyFlowDataset",
            "data_dir": cfg.FILE_PATH,
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.DATASET.label_keys,
            "train_resolution": cfg.DATASET.train_resolution,
            "test_resolutions": cfg.DATASET.test_resolutions,
            "grid_boundaries": cfg.DATASET.grid_boundaries,
            "encode_input": cfg.DATASET.encode_input,
            "encode_output": cfg.DATASET.encode_output,
            "encoding": cfg.DATASET.encoding,
            "channel_dim": cfg.DATASET.channel_dim,
            "data_split": "test_32x32",
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "batch_size": cfg.EVAL.batch_size,
        "num_workers": 0,
    }

    # set loss
    l2loss = metric.LpLoss_train(d=2, p=2)
    h1loss = metric.H1Loss_train(d=2)
    if cfg.TRAIN.training_loss == "l2":
        train_loss = l2loss
    if cfg.TRAIN.training_loss == "h1":
        train_loss = h1loss

    h1_eval_metric = metric.H1Loss(d=2)
    l2_eval_metric = metric.LpLoss(d=2, p=2)
    sup_validator_16 = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg_16,
        loss=ppsci.loss.FunctionalLoss(train_loss),
        metric={
            "h1": ppsci.metric.FunctionalMetric(h1_eval_metric),
            "l2": ppsci.metric.FunctionalMetric(l2_eval_metric),
        },
        name="Sup_Validator_16x16",
    )

    sup_validator_32 = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg_32,
        loss=ppsci.loss.FunctionalLoss(train_loss),
        metric={
            "h1": ppsci.metric.FunctionalMetric(h1_eval_metric),
            "l2": ppsci.metric.FunctionalMetric(l2_eval_metric),
        },
        name="Sup_Validator_32x32",
    )
    validator = {
        sup_validator_16.name: sup_validator_16,
        sup_validator_32.name: sup_validator_32,
    }

    model = ppsci.arch.UNONet(
        **cfg.MODEL,
    )

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        log_freq=cfg.log_freq,
        seed=cfg.seed,
        validator=validator,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
        compute_metric_by_batch=cfg.EVAL.compute_metric_by_batch,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )
    # evaluate
    solver.eval()


def export(cfg: DictConfig):
    # set model
    model = ppsci.arch.UNONet(
        **cfg.MODEL,
    )

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        pretrained_model_path=cfg.INFER.pretrained_model_path,
    )
    # export model
    from paddle.static import InputSpec

    input_spec = [
        {
            key: InputSpec([None, 3, 16, 16], "float32", name=key)
            for key in model.input_keys
        },
    ]
    solver.export(input_spec, cfg.INFER.export_path)


def inference(cfg: DictConfig):
    import matplotlib.pyplot as plt
    import predictor

    predictor = predictor.FNOPredictor(cfg)

    data = np.load(cfg.INFER.data_path, allow_pickle=True).item()

    input_data = data["x"][0].reshape(-1, 1, *data["x"].shape[1:]).astype("float32")
    label = data["y"][0].astype("float32")

    pred_data = predictor.predict(input_data, cfg.INFER.batch_size)

    fig = plt.figure(figsize=(7, 7))

    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(input_data.squeeze(), cmap="gray")
    ax.set_title("k(x)")
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(label)
    ax.set_title("Ground-truth y")
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(pred_data.squeeze())
    ax.set_title("Model prediction")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig(cfg.output_dir)
    logger.message("save success")
    plt.close(fig)


@hydra.main(
    version_base=None,
    config_path="./conf",
    config_name="uno_darcyflow_pretrain.yaml",
)
def main(cfg: DictConfig):
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
