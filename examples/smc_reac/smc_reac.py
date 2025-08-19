import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import paddle
import pandas as pd
import rdkit.Chem as Chem
from omegaconf import DictConfig
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split

import ppsci

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

x_train = None
x_test = None
y_train = None
y_test = None


def load_data(cfg: DictConfig):
    data_dir = cfg.data_dir
    dataset = pd.read_excel(data_dir, skiprows=1)
    x = dataset.iloc[:, 1:6]
    y = dataset.iloc[:, 6]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    return x_train, x_test, y_train, y_test


def data_processed(x, y):
    x = build_dataset(x)
    y = paddle.to_tensor(y.to_numpy(dtype=np.float32))
    y = paddle.unsqueeze(y, axis=1)
    return x, y


def build_dataset(data):
    r1 = paddle.to_tensor(np.array(cal_print(data.iloc[:, 0])), dtype=paddle.float32)
    r2 = paddle.to_tensor(np.array(cal_print(data.iloc[:, 1])), dtype=paddle.float32)
    ligand = paddle.to_tensor(
        np.array(cal_print(data.iloc[:, 2])), dtype=paddle.float32
    )
    base = paddle.to_tensor(np.array(cal_print(data.iloc[:, 3])), dtype=paddle.float32)
    solvent = paddle.to_tensor(
        np.array(cal_print(data.iloc[:, 4])), dtype=paddle.float32
    )
    return paddle.concat([r1, r2, ligand, base, solvent], axis=1)


def cal_print(smiles):
    vectors = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp = generator.GetFingerprint(mol)
        _input = np.array(list(map(float, fp.ToBitString())))
        vectors.append(_input)
    return vectors


def train(cfg: DictConfig):
    global x_train, y_train
    x_train, y_train = data_processed(x_train, y_train)

    # build supervised constraint
    sup = ppsci.constraint.SupervisedConstraint(
        dataloader_cfg={
            "dataset": {
                "input": {"v": x_train},
                "label": {"u": y_train},
                # "weight": {"W": param},
                "name": "IterableNamedArrayDataset",
            },
            "batch_size": cfg.TRAIN.batch_size,
        },
        loss=ppsci.loss.MSELoss("mean"),
        name="sup",
    )
    constraint = {
        "sup": sup,
    }

    model = ppsci.arch.SuzukiMiyauraModel(**cfg.MODEL)

    optimizer = ppsci.optimizer.optimizer.Adam(cfg.TRAIN.learning_rate)(model)

    # Build solver
    solver = ppsci.solver.Solver(
        model,
        constraint=constraint,
        optimizer=optimizer,
        epochs=cfg.TRAIN.epochs,
        eval_during_train=False,
        iters_per_epoch=cfg.TRAIN.iters_per_epoch,
        cfg=cfg,
    )
    solver.train()


def evaluate(cfg: DictConfig):
    global x_test, y_test

    x_test, y_test = data_processed(x_test, y_test)

    test_validator = ppsci.validate.SupervisedValidator(
        dataloader_cfg={
            "dataset": {
                "input": {"v": x_test},
                "label": {"u": y_test},
                "name": "IterableNamedArrayDataset",
            },
            "batch_size": cfg.EVAL.batch_size,
            "shuffle": False,
        },
        loss=ppsci.loss.MSELoss("mean"),
        metric={
            "MAE": ppsci.metric.MAE(),
            "RMSE": ppsci.metric.RMSE(),
            "R2": ppsci.metric.R2Score(),
        },
        name="test_eval",
    )
    validators = {"test_eval": test_validator}

    model = ppsci.arch.SuzukiMiyauraModel(**cfg.MODEL)
    solver = ppsci.solver.Solver(
        model,
        validator=validators,
        cfg=cfg,
    )

    loss_val, metric_dict = solver.eval()

    ypred = model({"v": x_test})["u"].numpy()
    ytrue = y_test.numpy()

    mae = metric_dict["MAE"]["u"]
    rmse = metric_dict["RMSE"]["u"]
    r2 = metric_dict["R2"]["u"]

    plt.figure()
    plt.scatter(ytrue, ypred, s=15, color="royalblue", marker="s", linewidth=1)
    plt.plot([ytrue.min(), ytrue.max()], [ytrue.min(), ytrue.max()], "r-", lw=1)
    plt.legend(title="R²={:.3f}\n\nMAE={:.3f}".format(r2, mae))
    plt.xlabel("Test Yield(%)")
    plt.ylabel("Predicted Yield(%)")
    save_path = "smc_reac.png"
    plt.savefig(save_path)
    print(f"Image saved to: {save_path}")
    plt.show()

    print("Evaluation metrics:")
    print(f"Loss: {loss_val:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")


@hydra.main(version_base=None, config_path="./config", config_name="smc_reac.yaml")
def main(cfg: DictConfig):
    global x_train, x_test, y_train, y_test

    x_train, x_test, y_train, y_test = load_data(cfg)

    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
