"""
PhyCRNet for solving spatiotemporal PDEs
Reference: https://github.com/isds-neu/PhyCRNet/
"""
from os import path as osp

import functions
import hydra
import paddle
import scipy.io as scio
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger


def train(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # set initial states for convlstm
    NUM_CONVLSTM = cfg.num_convlstm
    (h0, c0) = (paddle.randn((1, 128, 16, 16)), paddle.randn((1, 128, 16, 16)))
    initial_state = []
    for _ in range(NUM_CONVLSTM):
        initial_state.append((h0, c0))

    # grid parameters
    time_steps = cfg.TIME_STEPS
    dx = cfg.DX[0] / cfg.DX[1]

    steps = cfg.TIME_BATCH_SIZE + 1
    effective_step = list(range(0, steps))
    num_time_batch = int(time_steps / cfg.TIME_BATCH_SIZE)

    functions.dt = cfg.DT
    functions.dx = dx
    functions.time_steps = cfg.TIME_STEPS
    functions.num_time_batch = num_time_batch
    model = ppsci.arch.PhyCRNet(
        dt=cfg.DT, step=steps, effective_step=effective_step, **cfg.MODEL
    )

    def _transform_out(_in, _out):
        return functions.transform_out(_in, _out, model)

    model.register_input_transform(functions.transform_in)
    model.register_output_transform(_transform_out)

    # use Burgers_2d_solver_HighOrder.py to generate data
    data = scio.loadmat(cfg.DATA_PATH)
    uv = data["uv"]  # [t,c,h,w]
    functions.uv = uv

    # generate input data
    (
        input_dict_train,
        label_dict_train,
        input_dict_val,
        label_dict_val,
    ) = functions.Dataset(
        paddle.to_tensor(initial_state),
        paddle.to_tensor(
            uv[
                0:1,
            ],
            dtype=paddle.get_default_dtype(),
        ),
    ).get()

    sup_constraint_pde = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": input_dict_train,
                "label": label_dict_train,
            },
            "batch_size": 1,
            "num_workers": 0,
        },
        ppsci.loss.FunctionalLoss(functions.train_loss_func),
        {
            "loss": lambda out: out["loss"],
        },
        name="sup_train",
    )
    constraint_pde = {sup_constraint_pde.name: sup_constraint_pde}

    sup_validator_pde = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": input_dict_val,
                "label": label_dict_val,
            },
            "batch_size": 1,
            "num_workers": 0,
        },
        ppsci.loss.FunctionalLoss(functions.val_loss_func),
        {
            "loss": lambda out: out["loss"],
        },
        metric={"metric": ppsci.metric.FunctionalMetric(functions.metric_expr)},
        name="sup_valid",
    )
    validator_pde = {sup_validator_pde.name: sup_validator_pde}

    # initialize solver
    scheduler = ppsci.optimizer.lr_scheduler.Step(**cfg.TRAIN.lr_scheduler)()

    optimizer = ppsci.optimizer.Adam(scheduler)(model)
    solver = ppsci.solver.Solver(
        model,
        constraint_pde,
        cfg.output_dir,
        optimizer,
        scheduler,
        cfg.TRAIN.epochs,
        cfg.TRAIN.iters_per_epoch,
        validator=validator_pde,
        eval_with_no_grad=cfg.TRAIN.eval_with_no_grad,
        pretrained_model_path=cfg.TRAIN.pretrained_model_path,
    )

    # train model
    solver.train()
    # evaluate after finished training
    model.register_output_transform(functions.tranform_output_val)
    solver.eval()


def evaluate(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # set initial states for convlstm
    NUM_CONVLSTM = cfg.num_convlstm
    (h0, c0) = (paddle.randn((1, 128, 16, 16)), paddle.randn((1, 128, 16, 16)))
    initial_state = []
    for _ in range(NUM_CONVLSTM):
        initial_state.append((h0, c0))

    # grid parameters
    time_steps = cfg.TIME_STEPS
    dx = cfg.DX[0] / cfg.DX[1]

    steps = cfg.EVAL.TIME_BATCH_SIZE + 1
    effective_step = list(range(0, steps))
    num_time_batch = int(time_steps / cfg.EVAL.TIME_BATCH_SIZE)

    functions.dt = cfg.DT
    functions.dx = dx
    functions.num_time_batch = num_time_batch
    model = ppsci.arch.PhyCRNet(
        dt=cfg.DT, step=steps, effective_step=effective_step, **cfg.MODEL
    )

    def _transform_out(_in, _out):
        return functions.transform_out(_in, _out, model)

    model.register_input_transform(functions.transform_in)
    model.register_output_transform(_transform_out)

    # use the generated data
    data = scio.loadmat(cfg.DATA_PATH)
    uv = data["uv"]  # [t,c,h,w]
    functions.uv = uv
    _, _, input_dict_val, _ = functions.Dataset(
        paddle.to_tensor(initial_state),
        paddle.to_tensor(uv[0:1, ...], dtype=paddle.get_default_dtype()),
    ).get()
    ppsci.utils.load_pretrain(model, cfg.EVAL.pretrained_model_path)
    model.register_output_transform(None)
    functions.output_graph(model, input_dict_val, cfg.output_dir, cfg.case_name)


@hydra.main(
    version_base=None, config_path="./conf", config_name="burgers_equations.yaml"
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
