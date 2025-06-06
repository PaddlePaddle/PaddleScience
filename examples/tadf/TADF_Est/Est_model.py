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

# 加载数据集
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
    # 划分数据集
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
    # 处理数据集
    x_train = paddle.to_tensor(x_train, dtype="float32")
    x = {
        "key_{}".format(i): paddle.unsqueeze(
            paddle.to_tensor(x_train[:, i], dtype="float32"), axis=1
        )
        for i in range(x_train.shape[1])
    }
    y_train = paddle.unsqueeze(paddle.to_tensor(y_train, dtype="float32"), axis=1)

    # 构建约束
    bc_sup = ppsci.constraint.SupervisedConstraint(
        dataloader_cfg={
            "dataset": {
                "input": x,
                "label": {"u": y_train},
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
    optimizer = ppsci.optimizer.optimizer.Adam(
        cfg.TRAIN.learning_rate,
        beta1=(0.9, 0.99)[0],
        beta2=(0.9, 0.99)[1],
        weight_decay=cfg.TRAIN.weight_decay,
    )(model)
    # 构建Solver
    solver = ppsci.solver.Solver(
        model,
        constraint={
            "bc_sup": bc_sup,
        },
        optimizer=optimizer,
        epochs=cfg.TRAIN.epochs,
        eval_during_train=False,
        iters_per_epoch=cfg.TRAIN.iters_per_epoch,
        seed=cfg.seed,
    )
    try:
        solver.train()
    except Exception as ex:
        print("error", ex)
    paddle.save(model.state_dict(), cfg.TRAIN.save_model_path)


# 进行测试
def eval(cfg: DictConfig, X, data):
    y = paddle.to_tensor(data)
    # 重新划分数据集
    x_train, x_test, y_train, y_test = train_test_split(
        X.numpy(), y.numpy(), test_size=cfg.EVAL.test_size, random_state=cfg.EVAL.seed
    )
    x = {
        "key_{}".format(i): paddle.unsqueeze(
            paddle.to_tensor(x_test[:, i], "float32"), axis=1
        )
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
    ytest = paddle.unsqueeze(paddle.to_tensor(y_test, dtype="float32"), axis=1)
    ypred = model(x)
    ytest = {"u": ytest}

    # 计算损失
    loss = ppsci.metric.MAE()
    MAE = loss(ypred, ytest).get("u").numpy()
    loss = ppsci.metric.RMSE()
    RMSE = loss(ypred, ytest).get("u").numpy()
    ypred = ypred.get("u").numpy()
    ytest = ytest.get("u").numpy()
    R2 = r2_score(ytest, ypred)
    print("MAE", MAE)
    print("RMSE", RMSE)
    print("R2", R2)

    # 可视化
    plt.scatter(ytest, ypred, s=15, color="royalblue", marker="s", linewidth=1)
    plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], "r-", lw=1)
    plt.legend(title=f"R²={R2:.3f}\n\nMAE={MAE:.3f}")
    plt.xlabel("Test ΔEst(eV)")
    plt.ylabel("Predicted ΔEst(eV)")
    save_path = "test_Est.png"
    plt.savefig(save_path)
    # print(f"图片已保存至：{save_path}")
    # plt.show()
