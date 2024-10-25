import json
import os
import sys

import functions as func_module
import hydra
import paddle
from functions import plot_velocity
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger

os.environ["FLAGS_embedding_deterministic"] = "1"
os.environ["FLAGS_cudnn_deterministic"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["NCCL_ALGO"] = "Tree"


def evaluate(cfg: DictConfig):
    # get dataset configuration information
    with open("dataset_config.json") as f:
        try:
            ctx = json.load(f)[cfg.DATASET]
        except KeyError:
            print("Unsupported dataset.")
            sys.exit()

    if cfg.file_size is not None:
        ctx["file_size"] = cfg.file_size

    # get data transformation
    transform_data, transform_label = func_module.create_transform(ctx, cfg.k)

    # set model
    model_gen = ppsci.arch.VelocityGenerator(**cfg.MODEL.gen_net)

    # set valid_dataloader_cfg
    valid_dataloader_cfg = {
        "dataset": {
            "name": "FWIDataset",
            "input_keys": ("data",),
            "label_keys": ("real_image",),
            "anno": cfg.EVAL.dataset.anno,
            "preload": cfg.EVAL.dataset.preload,
            "sample_ratio": cfg.EVAL.dataset.sample_ratio,
            "file_size": ctx["file_size"],
            "transform_data": transform_data,
            "transform_label": transform_label,
        },
        "batch_size": cfg.EVAL.batch_size,
        "use_shared_memory": cfg.EVAL.use_shared_memory,
        "num_workers": cfg.EVAL.num_workers,
    }

    # set validator
    validator = ppsci.validate.SupervisedValidator(
        dataloader_cfg=valid_dataloader_cfg,
        loss=ppsci.loss.MAELoss("mean"),
        output_expr={"real_image": lambda out: out["fake_image"]},
        metric={
            "MAE": ppsci.metric.MAE(),
            "RMSE": ppsci.metric.RMSE(),
            "SSIM": ppsci.metric.FunctionalMetric(func_module.ssim_metirc),
        },
        name="val",
    )
    validator_dict = {validator.name: validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model=model_gen,
        validator=validator_dict,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
    )

    # evaluation
    solver.eval()

    # visualization
    if cfg.VIS.vis:
        with solver.no_grad_context_manager(True):
            for batch_idx, (input_, label_, _) in enumerate(validator.data_loader):
                if batch_idx + 1 > cfg.VIS.vb:
                    break
                fake_image = model_gen(input_)["fake_image"].numpy()
                real_image = label_["real_image"].numpy()
                for i in range(cfg.VIS.vsa):
                    plot_velocity(
                        fake_image[i, 0],
                        real_image[i, 0],
                        f"{cfg.output_dir}/V_{batch_idx}_{i}.png",
                    )
        print(f"The visualizations are saved to {cfg.output_dir}")


def train(cfg: DictConfig):
    # get dataset configuration information
    with open(cfg.DATASET_CONFIG) as f:
        try:
            ctx = json.load(f)[cfg.DATASET]
        except KeyError:
            print("Unsupported dataset.")
            sys.exit()

    if cfg.file_size is not None:
        ctx["file_size"] = cfg.file_size

    # get data transformation
    transform_data, transform_label = func_module.create_transform(ctx, cfg.k)

    # set model
    model_gen = ppsci.arch.VelocityGenerator(**cfg.MODEL.gen_net)
    model_dis = ppsci.arch.VelocityDiscriminator(**cfg.MODEL.dis_net)

    # set class for loss function
    gen_funcs = func_module.GenFuncs(model_dis, cfg.WEIGHT_DICT.gen)
    dis_funcs = func_module.DisFuncs(model_dis, cfg.WEIGHT_DICT.dis)

    # set dataloader config
    dataloader_cfg = {
        "dataset": {
            "name": "FWIDataset",
            "input_keys": ("data",),
            "label_keys": ("real_image",),
            "anno": cfg.TRAIN.dataset.anno,
            "preload": cfg.TRAIN.dataset.preload,
            "sample_ratio": cfg.TRAIN.dataset.sample_ratio,
            "file_size": ctx["file_size"],
            "transform_data": transform_data,
            "transform_label": transform_label,
        },
        "sampler": {
            "name": "BatchSampler",
            "shuffle": cfg.TRAIN.sampler.shuffle,
            "drop_last": cfg.TRAIN.sampler.drop_last,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "use_shared_memory": cfg.TRAIN.use_shared_memory,
        "num_workers": cfg.TRAIN.num_workers,
    }

    # set constraint
    constraint_gen = ppsci.constraint.SupervisedConstraint(
        dataloader_cfg=dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(gen_funcs.loss_func_gen),
        output_expr={"fake_image": lambda out: out["fake_image"]},
        name="cst_gen",
    )
    constraint_gen_dict = {constraint_gen.name: constraint_gen}

    constraint_dis = ppsci.constraint.SupervisedConstraint(
        dataloader_cfg=dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(dis_funcs.loss_func_dis),
        output_expr={"fake_image": lambda out: out["fake_image"]},
        name="cst_dis",
    )
    constraint_dis_dict = {constraint_dis.name: constraint_dis}

    # set optimizer
    optimizer = ppsci.optimizer.AdamW(
        learning_rate=cfg.TRAIN.learning_rate, weight_decay=cfg.TRAIN.weight_decay
    )
    optimizer_g = optimizer(model_gen)
    optimizer_d = optimizer(model_dis)

    # initialize solver
    solver_gen = ppsci.solver.Solver(
        model=model_gen,
        output_dir=cfg.output_dir,
        constraint=constraint_gen_dict,
        optimizer=optimizer_g,
        epochs=cfg.TRAIN.epochs_gen,
        iters_per_epoch=cfg.TRAIN.iters_per_epoch_gen,
    )

    solver_dis = ppsci.solver.Solver(
        model=model_gen,
        output_dir=cfg.output_dir,
        constraint=constraint_dis_dict,
        optimizer=optimizer_d,
        epochs=cfg.TRAIN.epochs_dis,
        iters_per_epoch=cfg.TRAIN.iters_per_epoch_dis,
    )

    # training
    for i in range(cfg.TRAIN.epochs):
        logger.message(f"\nEpoch: {i + 1}\n")
        solver_dis.train()
        solver_gen.train()

    # save model weight
    paddle.save(
        model_gen.state_dict(), os.path.join(cfg.output_dir, "model_gen.pdparams")
    )


@hydra.main(version_base=None, config_path="./conf", config_name="velocityGAN.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
