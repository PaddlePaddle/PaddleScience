import os

import matplotlib.pyplot as plt
import numpy as np
import paddle
import rdkit.Chem as Chem
from omegaconf import DictConfig
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import ppsci

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

# 数据集准备
def load_data(cfg):
    data_dir = cfg.data_dir
    sim_dir = cfg.sim_dir
    angle_dat_path = os.path.join(data_dir)
    smis_txt_path = os.path.join(sim_dir)

    data = []
    with open(angle_dat_path) as f:
        for line in f:
            num = float(line.strip()) / 90
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
    # 划分数据集
    def k_fold(k, i, X, Y):
        fold_size = X.shape[0] // k
        val_start = i * fold_size
        if i != k - 1:
            val_end = (i + 1) * fold_size
            x_val, y_val = X[val_start:val_end], Y[val_start:val_end]
            x_train = paddle.concat((X[0:val_start], X[val_end:]), axis=0)
            y_train = paddle.concat((Y[0:val_start], Y[val_end:]), axis=0)
        else:
            x_val, y_val = X[val_start:], Y[val_start:]
            x_train = X[0:val_start]
            y_train = Y[0:val_start]
        return x_train, y_train, x_val, y_val

    Y = paddle.to_tensor(data, dtype="float32")
    x_train, y_train, x_test, y_test = k_fold(cfg.TRAIN.k, cfg.TRAIN.i, X, Y)
    # 处理数据集
    x = {
        f"key_{i}": paddle.unsqueeze(x_train[:, i], axis=1)
        for i in range(x_train.shape[1])
    }

    param = paddle.empty((len(x["key_0"]), len(x_train)), "float32")
    param = ppsci.utils.initializer.xavier_normal_(param)

    # 构建约束
    bc_sup = ppsci.constraint.SupervisedConstraint(
        dataloader_cfg={
            "dataset": {
                "input": x,
                "label": {"u": y_train},
                "weight": {"W": param},
                "name": "IterableNamedArrayDataset",
            },
            "batch_size": cfg.TRAIN.batch_size,
        },
        loss=ppsci.loss.MSELoss("mean"),
        name="bc_sup",
    )

    # 设置模型
    hidden_size = [587, 256]
    num_layers = None
    # 实例化模型
    model = ppsci.arch.TADF(
        input_keys=tuple(x.keys()),
        hidden_size=hidden_size,
        num_layers=num_layers,
        **cfg.MODEL,
    )
    optimizer = ppsci.optimizer.Adam(
        learning_rate=cfg.TRAIN.learning_rate,
        beta1=0.9,
        beta2=0.99,
        weight_decay=cfg.TRAIN.weight_decay,
    )(model)

    # 构建Solver
    solver = ppsci.solver.Solver(
        model,
        constraint={"bc_sup": bc_sup},
        optimizer=optimizer,
        epochs=cfg.TRAIN.epochs,
        iters_per_epoch=cfg.TRAIN.iters_per_epoch,
        seed=cfg.seed,
    )
    try:
        solver.train()
    except Exception as ex:
        print(ex)
    paddle.save(model.state_dict(), cfg.TRAIN.save_model_path)


# 进行测试
def eval(cfg: DictConfig, X, data):
    y = paddle.to_tensor(data, dtype="float32")
    # 重新划分数据集
    x_train, x_test, y_train, y_test = train_test_split(
        X.numpy(), y.numpy(), test_size=cfg.EVAL.test_size, random_state=cfg.EVAL.seed
    )
    x_test = paddle.to_tensor(x_test, dtype="float32")
    y_test = paddle.to_tensor(y_test, dtype="float32")

    x = {
        f"key_{i}": paddle.unsqueeze(x_test[:, i], axis=1)
        for i in range(x_test.shape[1])
    }

    hidden_size = [587, 256]
    num_layers = None
    model = ppsci.arch.TADF(
        input_keys=tuple(x.keys()),
        hidden_size=hidden_size,
        num_layers=num_layers,
        **cfg.MODEL,
    )
    model.set_state_dict(paddle.load(cfg.EVAL.load_model_path))

    ypred = model(x)
    ytest = {"u": paddle.unsqueeze(y_test, axis=1)}

    # 计算损失
    mae_metric = ppsci.metric.MAE()
    rmse_metric = ppsci.metric.RMSE()
    MAE = mae_metric(ypred, ytest).get("u").numpy()
    RMSE = rmse_metric(ypred, ytest).get("u").numpy()
    R2 = r2_score(ytest["u"].numpy(), ypred.get("u").numpy())

    print("MAE", MAE)
    print("RMSE", RMSE)
    print("R2", R2)

    # 可视化
    plt.scatter(
        ytest["u"].numpy(), ypred.get("u").numpy(), s=15, color="royalblue", marker="s"
    )
    plt.plot(
        [ytest["u"].min(), ytest["u"].max()],
        [ytest["u"].min(), ytest["u"].max()],
        "r-",
        lw=1,
    )
    plt.legend(title=f"R²={R2:.3f}\n\nMAE={MAE:.3f}")
    plt.xlabel("Test θ(°)")
    plt.ylabel("Predicted θ(°)")
    save_path = "test_angle.png"
    plt.savefig(save_path)
    print(f"图片已保存至：{save_path}")
    plt.show()
