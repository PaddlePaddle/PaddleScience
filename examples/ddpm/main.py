from os import path as osp

import functions
import hydra
import numpy as np
import utils
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger


def train(cfg):

    args = cfg.train
    config = cfg
    data = np.load(cfg.data.data_dir)
    coords = data
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(args.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(args.output_dir, f"{cfg.mode}.log"), "info")
    model = ppsci.arch.DDPM(config=config)

    def _transform_out(_out):
        return utils.transform_out(_out)

    model.register_input_transform(utils.transform_in)
    model.register_output_transform(_transform_out)
    sup_constraint_pde = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": {
                    "x0": coords,
                },
            },
            "batch_size": cfg.training.batch_size,
            "num_workers": cfg.data.num_workers,
        },
        ppsci.loss.FunctionalLoss(utils.train_loss_func),
        {
            "loss": lambda out: out["loss"],
        },
        name="sup_train",
    )
    constraint_pde = {sup_constraint_pde.name: sup_constraint_pde}

    # initialize solver
    optimizer = ppsci.optimizer.Adam(config.optim.lr)(model)
    solver = ppsci.solver.Solver(
        model,
        constraint_pde,
        output_dir=args.output_dir,
        optimizer=optimizer,
        save_freq=20,
        epochs=cfg.training.n_epochs,
        iters_per_epoch=11448,
        checkpoint_path="/home/aistudio/work/PaddleScience/examples/ddpm/experiments/km256/checkpoints/latest",
        validator=None,
    )

    solver.train()


def evaluate(cfg):
    config = cfg
    if config.model.type == "conditional":
        print("Using conditional model")
        model = ppsci.arch.DDPM(config=config)
    else:
        # 原始代码中没有unconditional的model，所以这里直接raise NotImplemente
        raise NotImplementedError("not supported")
    ppsci.utils.load_pretrain(model, config.model.ckpt_path)
    functions.eval_input(model, config)


@hydra.main(
    version_base=None,
    config_path="conf/",
    config_name="kmflow_re1000_rs256_sparse_recons_conditional.yaml",
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
