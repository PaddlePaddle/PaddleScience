import warnings

import hydra
from omegaconf import DictConfig

import ppsci
import ppsci.constraint.supervised_constraint
import ppsci.optimizer as optim
from ppsci.arch import CrystalGraphConvNet
from ppsci.data.dataset import CGCNNDataset
from ppsci.data.dataset.cgcnn_dataset import collate_pool

warnings.filterwarnings("ignore")


def train(cfg: DictConfig):

    dataset = CGCNNDataset(
        cfg.TRAIN_DIR, input_keys=("i",), label_keys=("l",), id_keys=("c",)
    )

    structures, _, _ = dataset.raw_data[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=cfg.MODEL.atom_fea_len,
        n_conv=cfg.MODEL.n_conv,
        h_fea_len=cfg.MODEL.h_fea_len,
        n_h=cfg.MODEL.n_h,
    )

    cgcnn_constraint = ppsci.constraint.SupervisedConstraint(
        dataloader_cfg={
            "dataset": {
                "name": "CGCNNDataset",
                "root_dir": cfg.TRAIN_DIR,
                "input_keys": ("i",),
                "label_keys": ("l",),
                "id_keys": ("c",),
            },
            "batch_size": cfg.TRAIN.batch_size,
            "collate_fn": collate_pool,
        },
        loss=ppsci.loss.MAELoss("mean"),
        output_expr={"l": lambda out: out["out"]},
        name="cgcnn_constraint",
    )

    constraint = {cgcnn_constraint.name: cgcnn_constraint}

    cgcnn_valid = ppsci.validate.SupervisedValidator(
        dataloader_cfg={
            "dataset": {
                "name": "CGCNNDataset",
                "root_dir": cfg.VALID_DIR,
                "input_keys": ("i",),
                "label_keys": ("l",),
                "id_keys": ("c",),
            },
            "batch_size": cfg.TRAIN.batch_size,
            "collate_fn": collate_pool,
        },
        loss=ppsci.loss.MAELoss("mean"),
        output_expr={"l": lambda out: out["out"]},
        metric={"MAE": ppsci.metric.MAE()},
        name="cgcnn_valid",
    )
    validator = {cgcnn_valid.name: cgcnn_valid}

    optimizer = optim.Momentum(
        learning_rate=cfg.TRAIN.lr,
        momentum=cfg.TRAIN.momentum,
        weight_decay=cfg.TRAIN.weight_decay,
    )(model)

    solver = ppsci.solver.Solver(
        model=model,
        constraint=constraint,
        optimizer=optimizer,
        validator=validator,
        cfg=cfg,
    )

    solver.train()

    solver.eval()


def evaluate(cfg: DictConfig):

    dataset = CGCNNDataset(
        cfg.TEST_DIR, input_keys=("i",), label_keys=("l",), id_keys=("c",)
    )

    structures, _, _ = dataset.raw_data[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=cfg.MODEL.atom_fea_len,
        n_conv=cfg.MODEL.n_conv,
        h_fea_len=cfg.MODEL.h_fea_len,
        n_h=cfg.MODEL.n_h,
    )

    cgcnn_evaluate = ppsci.validate.SupervisedValidator(
        dataloader_cfg={
            "dataset": {
                "name": "CGCNNDataset",
                "root_dir": cfg.TEST_DIR,
                "input_keys": ("i",),
                "label_keys": ("l",),
                "id_keys": ("c",),
            },
            "batch_size": cfg.EVAL.batch_size,
            "collate_fn": collate_pool,
        },
        loss=ppsci.loss.MAELoss("mean"),
        output_expr={"l": lambda out: out["out"]},
        metric={"MAE": ppsci.metric.MAE()},
        name="cgcnn_evaluate",
    )
    validator = {cgcnn_evaluate.name: cgcnn_evaluate}
    solver = ppsci.solver.Solver(
        model,
        validator=validator,
        cfg=cfg,
    )

    solver.eval()


@hydra.main(version_base=None, config_path="./conf", config_name="CGCNN.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
