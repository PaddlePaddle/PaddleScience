

import ppsci
from ppsci.utils import logger
from omegaconf import DictConfig
import hydra

def train(cfg: DictConfig):
    # set model
    model = ppsci.arch.STAFNet(**cfg.MODEL) 
    train_dataloader_cfg = {
        "dataset": {
            "name": "STAFNetDataset",
            "file_path": cfg.STAFNet_DATA_PATH,
            "input_keys": ("aq_G","mete_G",),
            "label_keys": ("eta", "f"),
            # "weight_dict": {"eta": 100},
            "seq_len": 72,
            "pred_len": 48,


        },
        "batch_size": cfg.TRAIN.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
    }

    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        ppsci.loss.MSELoss("mean"),
        # {"eta": lambda out: out["eta"], **equation["VIV"].equations},
        name="STAFNet_Sup",
    )

    # lr_scheduler = ppsci.optimizer.lr_scheduler.Step(**cfg.TRAIN.lr_scheduler)()

     # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.Step(**cfg.TRAIN.lr_scheduler)()
    LEARNING_RATE = 0.001
    optimizer = ppsci.optimizer.Adam(LEARNING_RATE)(model)

   

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        sup_constraint,
        optimizer=optimizer,
        cfg=cfg,
    )

    # train model
    solver.train()

    return

def evaluate(cfg: DictConfig):
    """
    Validate after training an epoch

    :param epoch: Integer, current training epoch.
    :return: A log that contains information about validation
    """
    pass


@hydra.main(version_base=None, config_path="./conf", config_name="stafnet.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")




if __name__ == "__main__":
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)
    # set output directory
    OUTPUT_DIR = "./output_example"
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    main()