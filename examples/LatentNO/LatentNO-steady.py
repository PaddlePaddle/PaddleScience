import hydra
import paddle
from omegaconf import DictConfig
from utils import RelLpLoss

import ppsci


def train(cfg: DictConfig):
    model = ppsci.arch.LatentNO(**cfg.MODEL)

    train_dataloader_cfg = {
        "dataset": {
            "name": "LatentNODataset",
            "data_name": cfg.data_name,
            "data_mode": "train",
            "data_normalize": cfg.data_normalize,
            "data_concat": cfg.data_concat,
            "input_keys": ("x", "y1"),
            "label_keys": ("y2",),
        },
        "sampler": {"name": "BatchSampler", "drop_last": True, "shuffle": True},
        "batch_size": cfg.batch_size,
        "num_workers": cfg.get("num_workers", 0),
    }

    eval_dataloader_cfg = {
        "dataset": {
            "name": "LatentNODataset",
            "data_name": cfg.data_name,
            "data_mode": "val",
            "data_normalize": cfg.data_normalize,
            "data_concat": cfg.data_concat,
            "input_keys": ("x", "y1"),
            "label_keys": ("y2",),
        },
        "sampler": {"name": "BatchSampler", "drop_last": True, "shuffle": False},
        "batch_size": cfg.batch_size,
        "num_workers": cfg.get("num_workers", 0),
    }

    train_loss_fn = RelLpLoss(p=2, key="y2", normalizer=None)

    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        train_loss_fn,
        output_expr={"y2": lambda out: out["y2"]},
        name="SupTrain",
    )
    if cfg.data_normalize:
        normalizer = sup_constraint.data_loader.dataset.normalizer
    else:
        normalizer = None
    constraint = {sup_constraint.name: sup_constraint}

    cfg.TRAIN.iters_per_epoch = len(sup_constraint.data_loader)
    lr_scheduler = ppsci.optimizer.lr_scheduler.OneCycleLR(
        epochs=cfg.TRAIN.epochs,
        iters_per_epoch=cfg.TRAIN.iters_per_epoch,
        max_learning_rate=cfg.TRAIN.lr,
        divide_factor=cfg.TRAIN.div_factor,
        end_learning_rate=cfg.TRAIN.lr
        / cfg.TRAIN.div_factor
        / cfg.TRAIN.final_div_factor,
        phase_pct=cfg.TRAIN.pct_start,
    )()

    optimizer = ppsci.optimizer.AdamW(
        lr_scheduler,
        weight_decay=cfg.TRAIN.weight_decay,
        grad_clip=paddle.nn.ClipGradByNorm(clip_norm=cfg.TRAIN.clip_norm),
        beta1=cfg.TRAIN.beta0,
        beta2=cfg.TRAIN.beta1,
    )(model)

    metric_dict = {"L2Rel": RelLpLoss(p=2, key="y2", normalizer=normalizer)}

    val_loss_fn = RelLpLoss(p=2, key="y2", normalizer=normalizer)

    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        val_loss_fn,
        output_expr={"y2": lambda out: out["y2"]},
        metric=metric_dict,
        name="SupVal",
    )
    validator = {sup_validator.name: sup_validator}

    solver = ppsci.solver.Solver(
        model=model,
        optimizer=optimizer,
        constraint=constraint,
        validator=validator,
        cfg=cfg,
    )

    solver.train()
    solver.eval()


def evaluate(cfg: DictConfig):
    train_ds = ppsci.data.dataset.LatentNODataset(
        cfg.data_name,
        "train",
        cfg.data_normalize,
        cfg.data_concat,
        input_keys=("x", "y1"),
        label_keys=("y2",),
    )
    if cfg.data_normalize:
        normalizer = train_ds.normalizer
    else:
        normalizer = None

    eval_loss_fn = RelLpLoss(p=2, key="y2", normalizer=normalizer)

    model = ppsci.arch.LatentNO(**cfg.MODEL)

    eval_dataloader_cfg = {
        "dataset": {
            "name": "LatentNODataset",
            "data_name": cfg.data_name,
            "data_mode": "val",
            "data_normalize": cfg.data_normalize,
            "data_concat": cfg.data_concat,
            "input_keys": ("x", "y1"),
            "label_keys": ("y2",),
        },
        "sampler": {"name": "BatchSampler", "drop_last": True, "shuffle": False},
        "batch_size": cfg.batch_size,
        "num_workers": cfg.get("num_workers", 0),
    }

    metric_dict = {"L2Rel": RelLpLoss(p=2, key="y2", normalizer=normalizer)}

    validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        eval_loss_fn,
        output_expr={"y2": lambda out: out["y2"]},
        metric=metric_dict,
        name="Evaluation",
    )

    solver = ppsci.solver.Solver(
        model=model,
        validator={"eval": validator},
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
    )

    solver.eval()


@hydra.main(
    version_base=None, config_path="./config", config_name="LatentNO-Darcy.yaml"
)
def main(cfg: DictConfig):

    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
