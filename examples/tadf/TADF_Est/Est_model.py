import os

import matplotlib.pyplot as plt
import numpy as np
import paddle
import rdkit.Chem as Chem
from omegaconf import DictConfig
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import ppsci

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

# Data preparation
def load_data(cfg):
    data_dir = cfg.data_dir
    sim_dir = cfg.sim_dir
    angle_dat_path = os.path.join(data_dir)
    smis_txt_path = os.path.join(sim_dir)

    data = []
    with open(angle_dat_path) as f:
        for line in f:
            num = float(line.strip())
            data.append(num)

    smis = []
    with open(smis_txt_path) as f:
        for line in f:
            smis.append(line.strip())

    return data, smis


def featurize_molecules(smis):
    vectors = []
    del_mol = []
    for s in smis:
        mol = Chem.MolFromSmiles(s)
        try:
            generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            fp = generator.GetFingerprint(mol)
            _input = np.array(list(map(int, fp.ToBitString())))
            vectors.append(_input)
        except Exception as e:
            print(f"Error processing {s}: {e}")
            del_mol.append(s)
    pca = PCA(n_components=0.99)
    pca.fit(vectors)
    X = pca.transform(vectors)
    return paddle.to_tensor(X, dtype="float32")


def train(cfg: DictConfig, X, data):
    # k-fold cross validation splitter
    def k_fold(k, i, X, Y):
        fold_size = tuple(X.shape)[0] // k
        val_start = i * fold_size
        if i != k - 1:
            val_end = (i + 1) * fold_size
            x_val, y_val = X[val_start:val_end], Y[val_start:val_end]
            x_train = np.concatenate((X[0:val_start], X[val_end:]), axis=0)
            y_train = np.concatenate((Y[0:val_start], Y[val_end:]), axis=0)
        else:
            x_val, y_val = X[val_start:], Y[val_start:]
            x_train = X[0:val_start]
            y_train = Y[0:val_start]
        return x_train, y_train, x_val, y_val

    Y = paddle.to_tensor(data, dtype="float32")
    x_train, y_train, x_test, y_test = k_fold(cfg.TRAIN.k, cfg.TRAIN.i, X, Y)
    # Prepare feature dictionary
    x_train = paddle.to_tensor(x_train, dtype="float32")
    x = {
        f"key_{i}": paddle.unsqueeze(
            paddle.to_tensor(x_train[:, i], dtype="float32"), axis=1
        )
        for i in range(x_train.shape[1])
    }
    y_train = paddle.unsqueeze(paddle.to_tensor(y_train, dtype="float32"), axis=1)

    # Build supervised constraint
    sup = ppsci.constraint.SupervisedConstraint(
        dataloader_cfg={
            "dataset": {
                "name": "IterableNamedArrayDataset",
                "input": x,
                "label": {"u": y_train},
            },
            "batch_size": cfg.TRAIN.batch_size,
        },
        loss=ppsci.loss.MSELoss("mean"),
        name="sup",
    )

    # Set model architecture parameters
    hidden_size = [587, 256]
    num_layers = None
    # Instantiate TADF model
    model = ppsci.arch.TADF(
        input_keys=tuple(x.keys()),
        hidden_size=hidden_size,
        num_layers=num_layers,
        **cfg.MODEL,
    )
    optimizer = ppsci.optimizer.optimizer.Adam(
        cfg.TRAIN.learning_rate,
        beta1=0.9,
        beta2=0.99,
        weight_decay=cfg.TRAIN.weight_decay,
    )(model)

    # Build solver for training
    solver = ppsci.solver.Solver(
        model,
        constraint={sup.name: sup},
        optimizer=optimizer,
        epochs=cfg.TRAIN.epochs,
        eval_during_train=False,
        iters_per_epoch=cfg.TRAIN.iters_per_epoch,
    )
    try:
        solver.train()
    except Exception as ex:
        print("error", ex)


def evaluate(cfg: DictConfig, X, data):
    y_full = paddle.to_tensor(data, dtype="float32")
    X_np = X.numpy()
    y_np = y_full.numpy()
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_np,
        y_np,
        test_size=cfg.EVAL.test_size,
        random_state=cfg.EVAL.seed,
    )
    x_test = paddle.to_tensor(X_test_np, dtype="float32")
    y_test = paddle.to_tensor(y_test_np, dtype="float32")

    x_dict = {
        f"key_{i}": paddle.unsqueeze(x_test[:, i], axis=1)
        for i in range(x_test.shape[1])
    }

    test_validator = ppsci.validate.SupervisedValidator(
        dataloader_cfg={
            "dataset": {
                "name": "IterableNamedArrayDataset",
                "input": x_dict,
                "label": {"u": paddle.unsqueeze(y_test, axis=1)},
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

    model = ppsci.arch.TADF(
        input_keys=tuple(x_dict.keys()),
        hidden_size=[587, 256],
        num_layers=None,
        **cfg.MODEL,
    )

    solver = ppsci.solver.Solver(
        model,
        validator=validators,
        cfg=cfg,
    )

    _, metric_dict = solver.eval()

    ypred = model(x_dict)["u"].numpy()
    ytrue = paddle.unsqueeze(y_test, axis=1).numpy()

    mae = metric_dict["MAE"]["u"]
    rmse = metric_dict["RMSE"]["u"]
    r2 = metric_dict["R2"]["u"]

    print("Evaluation metrics:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    plt.scatter(
        ytrue,
        ypred,
        s=15,
        color="royalblue",
        marker="s",
        linewidth=1,
    )
    plt.plot(
        [ytrue.min(), ytrue.max()],
        [ytrue.min(), ytrue.max()],
        "r-",
        lw=1,
    )
    plt.legend(title=f"R²={r2:.3f}\n\nMAE={mae:.3f}")
    plt.xlabel("Test θ(°)")
    plt.ylabel("Predicted θ(°)")
    save_path = "test_est.png"
    plt.savefig(save_path)
    print(f"图片已保存至：{save_path}")
    plt.show()
