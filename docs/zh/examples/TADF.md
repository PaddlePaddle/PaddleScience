# 化学领域-分子性质预测
## 1.背景简介
有机发光二极管（OLED）具有高效率、结构灵活和低成本的优势，在先进显示和照明技术中受到广泛关注。在有机发光二极管器件中，电注入载流子以1：3的比例形成单线态和三线态激子。以纯荧光材料为发光材料构建的OLED发光效率IQE理论极限为25%。另一方面，有机金属复合物发光材料通过引入稀有金属（Ir，Pt等）带来强自旋轨道耦合（SOC），可以将单线态激子通过系间窜越过程转化成三线态激子，从而利用三线态激子发出磷光，其IQE可达100%，但是稀有金属价格昂贵，为推广使用带来了阻碍。热活化延迟荧光材料（TADF）为解决这些问题提供了新思路，并引起了广泛关注。在TADF中，三线态通过逆系间窜越过程（RISC）转化成单重态并发出荧光，从而实现100%的IQE，而RISC过程很大程度上取决于最低单线态（S1）和最低三线态（T1） 之间的能隙（ΔEST）。根据量子力学理论，ΔEST相当于HOMO和LUMO之间的交换积分的两倍。因此TADF分子的常见设计策略是将电子供体（D）和电子受体（A）以明显扭曲的二面角结合以实现HOMO和LUMO在空间上明显的分离。然而，与ΔEST相反，振子强度（f）需要较大的HOMO和LUMO之间的重叠积分，这二者之间的矛盾需要进一步平衡。
##  2.功能目标
通过高通量计算构建数据集，使用分子指纹作为模型输入，实现对于TADF材料分子性质的无计算预测。

## 3.数据库构建
我们选择常用的49个受体和50个受体以单键相连的方式进行组合，通过穷举所有可能的组合位点我们得到了44470个分子。通过MMFF94力场优化得到分子的初始结构。从44470个分子中随机提取5136个分子，在B3LYP/6-31G（d）水平下对5136个分子进行基态结构优化，采用TDDFT方法在基态构型下进行激发态性质计算。主要数据在data目录下。

## 4.模型构建
对于三个预测对象，设计了相同的深度神经网络，网络结构为含有两层隐藏层的神经网络，第一层隐藏层含有587个神经元，第二层隐藏层含有256个神经元，隐藏层之间加入Dropout。本文档以angle预测为例介绍

```
model = ppsci.arch.DNN(
    input_keys=tuple(x.keys()),
    hidden_size=hidden_size,
    num_layers=num_layers,
    **cfg.MODEL,
)
```

## 5.约束构建
本研究采用监督学习，直接构建'SupervisedConstraint'

```
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
```

## 6. 优化器构建
训练器采用Adam优化器

```
 optimizer = ppsci.optimizer.optimizer.Adam(
        cfg.TRAIN.learning_rate,
        beta1=(0.9, 0.99)[0],
        beta2=(0.9, 0.99)[1],
        weight_decay=cfg.TRAIN.weight_decay,
    )(model)

```

## 7. 模型训练
完成上述设置之后，只需要将上述实例化的对象按顺序传递给'ppsci.solver.Solver'，然后启动训练

```
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
```

## 8. 完整代码

```
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import paddle
import rdkit.Chem as Chem
from omegaconf import DictConfig
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import ppsci

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

# 加载数据集
data = []
for line in open("./angle.dat"):
    num = float(line.strip())
    num = num / 90
    data.append(num)
smis = []
for line in open("./smis.txt"):
    smis.append(line.strip())
vectors = []
del_mol = []
for num in smis:
    mol = Chem.MolFromSmiles(num)
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        _input = np.array(list(map(float, fp.ToBitString())))
        vectors.append(_input)
    except Exception:
        del_mol.append(num)
pca = PCA(n_components=0.99)
pca.fit(vectors)
Xlist = paddle.to_tensor(pca.transform(vectors))


def train(cfg: DictConfig):
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

    x_train, y_train, x_test, y_test = k_fold(cfg.TRAIN.k, cfg.TRAIN.i, Xlist, data)
    # 处理数据集
    x_train = paddle.to_tensor(x_train, dtype="float32")
    x = {
        "key_{}".format(i): paddle.unsqueeze(
            paddle.to_tensor(x_train[:, i], dtype="float32"), axis=1
        )
        for i in range(x_train.shape[1])
    }
    y_train = paddle.unsqueeze(paddle.to_tensor(y_train, dtype="float32"), axis=1)

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
    hidden_size = [587, 256]
    num_layers = None
    # 实例化模型
    model = ppsci.arch.DNN(
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
        print(ex)
    paddle.save(model.state_dict(), cfg.TRAIN.save_model_path)


# 进行测试
def eval(cfg: DictConfig):
    # 重新划分数据集
    x_train, x_test, y_train, y_test = train_test_split(
        Xlist, data, test_size=cfg.EVAL.test_size, random_state=cfg.EVAL.seed
    )
    x = {
        "key_{}".format(i): paddle.unsqueeze(
            paddle.to_tensor(x_test[:, i], "float32"), axis=1
        )
        for i in range(x_test.shape[1])
    }
    hidden_size = [587, 256]
    num_layers = None
    model = ppsci.arch.DNN(
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
    plt.legend(title="R²={:.3f}\n\nMAE={:.3f}".format(R2, MAE))
    plt.xlabel("Test θ(°)")
    plt.ylabel("Predicted θ(°)")
    plt.show()


@hydra.main(version_base=None, config_path="./config", config_name="angle.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        eval(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


# if cfg.mode == "train":
#     train(cfg)
# elif cfg.mode == "eval":
#     evaluate(cfg)


if __name__ == "__main__":
    main()
```
