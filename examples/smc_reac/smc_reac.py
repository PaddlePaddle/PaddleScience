import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import paddle
import pandas as pd
import rdkit.Chem as Chem
from omegaconf import DictConfig
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import r2_score
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
    # Reformat data for evaluation
    x_test = {"v": x_test}
    y_test = {"u": y_test}
    model = ppsci.arch.ChemMultimodalMLP(**cfg.MODEL)
    model.set_state_dict(paddle.load(cfg.EVAL.load_model_path))
    ypred = model(x_test)

    # Calculate evaluation metrics
    loss = ppsci.metric.MAE()
    MAE = loss(ypred, y_test).get("u").numpy()
    loss = ppsci.metric.RMSE()
    RMSE = loss(ypred, y_test).get("u").numpy()
    ypred = ypred.get("u").numpy()
    ytest = y_test.get("u").numpy()
    R2 = r2_score(ytest, ypred)
    print("MAE", MAE)
    print("RMSE", RMSE)
    print("R2", R2)

    # Visualization
    plt.scatter(ytest, ypred, s=15, color="royalblue", marker="s", linewidth=1)
    plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], "r-", lw=1)
    plt.legend(title="R²={:.3f}\n\nMAE={:.3f}".format(R2, MAE))
    plt.xlabel("Test Yield(%)")
    plt.ylabel("Predicted Yield(%)")
    save_path = "chem.png"
    plt.savefig(save_path)
    print(f"Iamge saved to: {save_path}")
    plt.show()


@hydra.main(version_base=None, config_path="./config", config_name="chem.yaml")
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
