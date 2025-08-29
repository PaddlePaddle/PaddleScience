import os
import platform

import hydra
import paddle
from functions import ToyDisFuncs
from functions import ToyGenFuncs
from functions import generate_toy_image
from functions import invalid_metric
from functions import load_toy_data
from omegaconf import DictConfig
from wgangp_toy_model import WganGpToyDiscriminator
from wgangp_toy_model import WganGpToyGenerator

import ppsci
from ppsci.utils import logger


def evaluate(cfg: DictConfig):
    # set model
    discriminator_model = WganGpToyDiscriminator(**cfg["MODEL"]["dis_net"])
    if cfg.EVAL.pretrained_dis_model_path and os.path.exists(
        cfg.EVAL.pretrained_dis_model_path
    ):
        discriminator_model.load_dict(paddle.load(cfg.EVAL.pretrained_dis_model_path))
    generator_model = WganGpToyGenerator(**cfg["MODEL"]["gen_net"])

    # set Loss
    generator_funcs = ToyGenFuncs(discriminator_model=discriminator_model)

    # set dataloader
    inputs = load_toy_data(**cfg["DATA"])
    valid_dataloader_cfg = {
        "dataset": {
            "name": cfg["EVAL"]["dataset"]["name"],
            "input": inputs,
        },
        "batch_size": cfg["EVAL"]["batch_size"],
        "use_shared_memory": cfg["EVAL"]["use_shared_memory"],
        "num_workers": cfg["EVAL"]["num_workers"]
        if platform.system() != "Windows"
        else 0,
    }

    # set validator
    validator = ppsci.validate.SupervisedValidator(
        dataloader_cfg=valid_dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(generator_funcs.loss),
        metric={"invalid_metric": ppsci.metric.FunctionalMetric(invalid_metric)},
        name="val",
    )
    validator_dict = {validator.name: validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model=generator_model,
        validator=validator_dict,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
        output_dir=cfg.output_dir,
    )

    # eval
    solver.eval()

    # visualization
    if cfg.VIS.vis:
        with solver.no_grad_context_manager(True):
            input_, _, _ = next(iter(validator.data_loader))
            real_data = input_["real_data"]
            generate_toy_image(
                true_dist=real_data,
                discriminator=discriminator_model,
                path=os.path.join(cfg.output_dir, "image.png"),
            )
        print(f"The visualizations are saved to {cfg.output_dir}")


def train(cfg: DictConfig):
    # set model
    generator_model = WganGpToyGenerator(**cfg["MODEL"]["gen_net"])
    discriminator_model = WganGpToyDiscriminator(**cfg["MODEL"]["dis_net"])
    if cfg.TRAIN.pretrained_dis_model_path and os.path.exists(
        cfg.TRAIN.pretrained_dis_model_path
    ):
        discriminator_model.load_dict(paddle.load(cfg.TRAIN.pretrained_dis_model_path))

    # set Loss
    generator_funcs = ToyGenFuncs(discriminator_model=discriminator_model)
    discriminator_funcs = ToyDisFuncs(
        **cfg["LOSS"]["dis"], discriminator_model=discriminator_model
    )

    # set dataloader
    inputs = load_toy_data(**cfg["DATA"])
    dataloader_cfg = {
        "dataset": {
            "name": cfg["EVAL"]["dataset"]["name"],
            "input": inputs,
        },
        "sampler": {
            **cfg["TRAIN"]["sampler"],
        },
        "batch_size": cfg["TRAIN"]["batch_size"],
        "use_shared_memory": cfg["TRAIN"]["use_shared_memory"],
        "num_workers": cfg["TRAIN"]["num_workers"],
        "drop_last": cfg["TRAIN"]["drop_last"],
    }

    # set constraint
    constraint_generator = ppsci.constraint.SupervisedConstraint(
        dataloader_cfg=dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(generator_funcs.loss),
        name="constraint_generator",
    )
    constraint_generator_dict = {constraint_generator.name: constraint_generator}

    constraint_discriminator = ppsci.constraint.SupervisedConstraint(
        dataloader_cfg=dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(discriminator_funcs.loss),
        output_expr={"real_data": lambda out: out["real_data"]},
        name="constraint_discriminator",
    )
    constraint_discriminator_dict = {
        constraint_discriminator.name: constraint_discriminator
    }

    # set optimizer
    optimizer = ppsci.optimizer.Adam(**cfg["TRAIN"]["optimizer"])

    optimizer_generator = optimizer(generator_model)
    optimizer_discriminator = optimizer(discriminator_model)

    # initialize solver
    solver_generator = ppsci.solver.Solver(
        model=generator_model,
        output_dir=os.path.join(cfg.output_dir, "generator"),
        constraint=constraint_generator_dict,
        optimizer=optimizer_generator,
        epochs=cfg.TRAIN.epochs_gen,
        iters_per_epoch=cfg.TRAIN.iters_per_epoch_gen,
        pretrained_model_path=cfg.TRAIN.pretrained_gen_model_path,
    )
    solver_discriminator = ppsci.solver.Solver(
        model=generator_model,
        output_dir=os.path.join(cfg.output_dir, "discriminator"),
        constraint=constraint_discriminator_dict,
        optimizer=optimizer_discriminator,
        epochs=cfg.TRAIN.epochs_dis,
        iters_per_epoch=cfg.TRAIN.iters_per_epoch_dis,
        pretrained_model_path=cfg.TRAIN.pretrained_gen_model_path,
    )

    # train
    for i in range(cfg.TRAIN.epochs):
        logger.message(f"\nEpoch: {i + 1}\n")
        optimizer_discriminator.clear_grad()
        solver_discriminator.train()
        optimizer_generator.clear_grad()
        solver_generator.train()

    # save model weight
    paddle.save(
        generator_model.state_dict(),
        os.path.join(cfg.output_dir, "model_generator.pdparams"),
    )
    paddle.save(
        discriminator_model.state_dict(),
        os.path.join(cfg.output_dir, "model_discriminator.pdparams"),
    )


@hydra.main(version_base=None, config_path="./conf", config_name="wgangp_toy.yaml")
def main(cfg: DictConfig):
    ppsci.utils.misc.set_random_seed(cfg["seed"])
    logger.init_logger(
        cfg.LOGGER.name, log_file=os.path.join(cfg.output_dir, cfg.LOGGER.log_file)
    )
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
