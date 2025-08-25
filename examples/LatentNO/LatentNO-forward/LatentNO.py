import ppsci
import hydra
import paddle
import numpy as np
import random
from omegaconf import DictConfig
from loss import RelLpLoss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

def train(cfg: DictConfig):
    
    set_seed(cfg.seed)
    device = 'gpu:0'
    paddle.set_device(device)

    model = ppsci.arch.LatentNO(
        n_block=cfg.MODEL.n_block,
        n_mode=cfg.MODEL.n_mode,
        n_dim=cfg.MODEL.n_dim,
        n_head=cfg.MODEL.n_head,
        n_layer=cfg.MODEL.n_layer,
        trunk_dim=cfg.MODEL.trunk_dim,
        branch_dim=cfg.MODEL.branch_dim,
        out_dim=cfg.MODEL.out_dim,
    )
    train_dataset_cfg = {
            "name": "LatentNODataset",
            "data_name": cfg.data_name,
            "data_mode": "train",
            "data_normalize": cfg.data_normalize,
            "data_concat": cfg.data_concat,
            "input_keys": ("x", "y1"),
            "label_keys": ("y2",),
            "weight_dict": None,  
            "transform_fn": None,  
        }

    train_dataloader_cfg = {
        "dataset": {
            "name": "LatentNODataset",
            "data_name": cfg.data_name,
            "data_mode": "train",
            "data_normalize": cfg.data_normalize,
            "data_concat": cfg.data_concat,
            "input_keys": ("x", "y1"),
            "label_keys": ("y2",),
            "weight_dict": None,  
            "transform_fn": None,  
        },
        "sampler": {"name": "BatchSampler", "drop_last": True, "shuffle": True},
        "batch_size": cfg.TRAIN.train_batch_size,
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
            "weight_dict": None,
            "transform_fn": None,
        },
        "sampler": {"name": "BatchSampler", "drop_last": True, "shuffle": False},
        "batch_size": cfg.EVAL.eval_batch_size,
        "num_workers": cfg.get("num_workers", 0),
    }
    train_ds = ppsci.data.dataset.build_dataset(train_dataset_cfg)
    normalizer = train_ds.normalizer
    
    iters_per_epoch = cfg.get("iters_per_epoch", None)
    if iters_per_epoch is None:
        from ppsci.data import build_dataloader
        tmp_loader = build_dataloader(train_ds, train_dataloader_cfg)
        iters_per_epoch = len(tmp_loader)
    cfg.TRAIN.iters_per_epoch = iters_per_epoch
    
    train_loss_fn = RelLpLoss(p=2, key="y2",normalizer=None)
    val_loss_fn = RelLpLoss(p=2, key="y2", normalizer=normalizer) 
    
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        train_loss_fn,
        output_expr={"y2": lambda out: out["y2"]},
        name="SupTrain",
    )

    constraint = {sup_constraint.name: sup_constraint}

    print("tmp_loader length:", len(tmp_loader))
    print("cfg.TRAIN.epochs:", cfg.TRAIN.epochs)
    iters_per_epoch = len(tmp_loader)
    total_steps = cfg.TRAIN.epochs * iters_per_epoch
    print("Computed total_steps:", total_steps)
    lr_scheduler = ppsci.optimizer.lr_scheduler.OneCycleLR(
        epochs=cfg.TRAIN.epochs,
        iters_per_epoch=iters_per_epoch,
        max_learning_rate=cfg.lr,
        divide_factor=cfg.div_factor,
        end_learning_rate=cfg.lr / cfg.div_factor / cfg.final_div_factor,
        phase_pct=cfg.pct_start,
    )()

    optimizer = ppsci.optimizer.AdamW(
        lr_scheduler,
        weight_decay=cfg.weight_decay,
        grad_clip=paddle.nn.ClipGradByNorm(clip_norm=cfg.clip_norm),
        beta1=cfg.beta0,
        beta2=cfg.beta1,
    )(model)

    metric_dict = {"L2Rel": RelLpLoss(p=2, key="y2", normalizer=normalizer)}

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
        iters_per_epoch=iters_per_epoch,
        constraint=constraint,
        optimizer=optimizer,
        cfg=cfg,
        validator=validator,
        output_dir=cfg.get("output_dir", "./outputs"),
        seed=cfg.seed,
    )

    solver.train()

    solver.eval()

def evaluate(cfg: DictConfig):
    set_seed(cfg.seed)
    device = 'gpu:0'
    paddle.set_device(device)
    train_ds = ppsci.data.dataset.LatentNODataset(cfg.data_name, "train", cfg.data_normalize, cfg.data_concat, input_keys=("x", "y1"),
            label_keys=("y2",),)
    normalizer = train_ds.normalizer

    eval_loss_fn = RelLpLoss(p=2, key="y2", normalizer=normalizer)
    model = ppsci.arch.LatentNO(
        n_block=cfg.MODEL.n_block,
        n_mode=cfg.MODEL.n_mode,
        n_dim=cfg.MODEL.n_dim,
        n_head=cfg.MODEL.n_head,
        n_layer=cfg.MODEL.n_layer,
        trunk_dim=cfg.MODEL.trunk_dim,
        branch_dim=cfg.MODEL.branch_dim,
        out_dim=cfg.MODEL.out_dim,
    )
    
    pretrained_model_path = "./output2/checkpoints/latest.pdparams"
    model_state = paddle.load(pretrained_model_path)
    model.set_state_dict(model_state)
    
    eval_dataloader_cfg = {
        "dataset": {
            "name": "LatentNODataset",
            "data_name": cfg.data_name,
            "data_mode": "val",
            "data_normalize": cfg.data_normalize,
            "data_concat": cfg.data_concat,
            "input_keys": ("x", "y1"),
            "label_keys": ("y2",),
            "weight_dict": None,
            "transform_fn": None,
        },
        "sampler": {"name": "BatchSampler", "drop_last": True, "shuffle": False},
        "batch_size": cfg.EVAL.eval_batch_size,
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
        output_dir=cfg.get("output_dir", "./outputs"),
        seed=cfg.seed,
    )
    
    solver.eval()

@hydra.main(version_base=None, config_path="./config", config_name="LatentNO-forward-Darcy.yaml")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")
    
if __name__ == "__main__":
    main()