import hydra
import numpy as np
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger


@hydra.main(version_base=None, config_path="./conf", config_name="VP_NSFNet1.yaml")
def main(cfg: DictConfig):
    OUTPUT_DIR = cfg.output_dir
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # set random seed for reproducibility
    SEED = cfg.seed
    ppsci.utils.misc.set_random_seed(SEED)

    # set model
    input_key = ("x", "y")
    output_key = ("u", "v", "p")
    model = ppsci.arch.MLP(
        input_key,
        output_key,
        cfg.model.ihlayers,
        cfg.model.ineurons,
        "tanh",
        input_dim=len(input_key),
        output_dim=len(output_key),
        Xavier=True,
    )

    # set the number of residual samples
    N_TRAIN = cfg.ntrain

    # set the Reynolds number and the corresponding lambda which is the parameter in the exact solution.
    Re = cfg.re
    lam = 0.5 * Re - np.sqrt(0.25 * (Re**2) + 4 * (np.pi**2))

    x_train = (np.random.rand(N_TRAIN, 1) - 1 / 3) * 3 / 2
    y_train = (np.random.rand(N_TRAIN, 1) - 1 / 4) * 2

    # generate test data
    np.random.seed(SEED)
    x_star = ((np.random.rand(1000, 1) - 1 / 3) * 3 / 2).astype("float32")
    y_star = ((np.random.rand(1000, 1) - 1 / 4) * 2).astype("float32")
    u_star = 1 - np.exp(lam * x_star) * np.cos(2 * np.pi * y_star)
    v_star = (lam / (2 * np.pi)) * np.exp(lam * x_star) * np.sin(2 * np.pi * y_star)
    p_star = 0.5 * (1 - np.exp(2 * lam * x_star))

    valida_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": {"x": x_star, "y": y_star},
            "label": {"u": u_star, "v": v_star, "p": p_star},
        },
        "total_size": u_star.shape[0],
        "batch_size": u_star.shape[0],
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }

    geom = ppsci.geometry.PointCloud({"x": x_train, "y": y_train}, ("x", "y"))

    # set equation constarint s.t. ||F(u)||
    equation = {
        "NavierStokes": ppsci.equation.NavierStokes(
            nu=1.0 / Re, rho=1.0, dim=2, time=False
        ),
    }

    residual_validator = ppsci.validate.SupervisedValidator(
        valida_dataloader_cfg,
        ppsci.loss.L2RelLoss(),
        metric={"L2R": ppsci.metric.L2Rel()},
        name="Residual",
    )

    # wrap validator
    validator = {residual_validator.name: residual_validator}

    # load solver
    solver = ppsci.solver.Solver(
        model,
        equation=equation,
        geom=geom,
        validator=validator,
        pretrained_model_path=cfg.pretrained_model_path,  ### the path of the model
    )

    # eval model
    solver.eval()


if __name__ == "__main__":
    main()
