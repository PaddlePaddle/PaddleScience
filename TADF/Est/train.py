import os

import matplotlib.pyplot as plt
import numpy as np
import paddle
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA

import ppsci

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["SimHei"]


# 加载数据集
data = []
for line in open("D://resources//machine learning//paddle//2024-08//Est//Est.dat"):
    num = float(line.strip())
    data.append(num)
smis = []
for line in open("D://resources//machine learning//paddle//2024-08//Est//smis.txt"):
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

# 划分数据集
def k_fold(k, i, X, Y):
    fold_size = tuple(X.shape)[0] // k
    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        x_val, y_val = X[val_start:val_end], Y[val_start:val_end]
        # x_train = paddle.concat(x=(X[0:val_start], X[val_end:]), axis=0)
        # y_train = paddle.concat(x=(Y[0:val_start], Y[val_end:]), axis=0)
        x_train = np.concatenate((X[0:val_start], X[val_end:]), axis=0)
        y_train = np.concatenate((Y[0:val_start], Y[val_end:]), axis=0)
    else:
        x_val, y_val = X[val_start:], Y[val_start:]
        x_train = X[0:val_start]
        y_train = Y[0:val_start]
    return x_train, y_train, x_val, y_val


xtrain, ytrain, x_val, y_val = k_fold(9, 1, Xlist, data)

# 处理数据集
xtrain = paddle.to_tensor(xtrain, dtype="float32")
x = {
    "key_{}".format(i): paddle.unsqueeze(
        paddle.to_tensor(xtrain[:, i], dtype="float32"), axis=1
    )
    for i in range(xtrain.shape[1])
}
ytrain = paddle.unsqueeze(paddle.to_tensor(ytrain, dtype="float32"), axis=1)

# 构建约束
bc_sup = ppsci.constraint.SupervisedConstraint(
    dataloader_cfg={
        "dataset": {
            "input": x,
            "label": {"u": ytrain},
            "name": "IterableNamedArrayDataset",
        },
        "batch_size": 8,
    },
    loss=ppsci.loss.MSELoss("mean"),
    name="bc_sup",
)
constraint = {
    "bc_sup": bc_sup,
}

# 实例化模型
model = ppsci.arch.DNN(tuple(x.keys()), ("u",), None, [587, 256], "relu", dropout=0.5)
optimizer = ppsci.optimizer.optimizer.Adam(
    0.0001, beta1=(0.9, 0.99)[0], beta2=(0.9, 0.99)[1], weight_decay=1e-5
)(model)

# 构建Solver
solver = ppsci.solver.Solver(
    model,
    constraint={
        "bc_sup": bc_sup,
    },
    optimizer=optimizer,
    epochs=200,
    eval_during_train=False,
    iters_per_epoch=20,
    seed=42,
    device="cpu",
)
try:
    solver.train()
except Exception as ex:
    print(ex)

# 进行验证
y_val = paddle.unsqueeze(paddle.to_tensor(y_val, dtype="float32"), axis=1)
x = {
    "key_{}".format(i): paddle.unsqueeze(
        paddle.to_tensor(x_val[:, i], dtype="float32"), axis=1
    )
    for i in range(x_val.shape[1])
}
ypred = solver.predict(x)
y_val = {"u": y_val}
loss = ppsci.metric.RMSE()
RMSE = loss(ypred, y_val).get("u").numpy()
ypred = ypred.get("u").numpy()
ytest = y_val.get("u").numpy()

# 验证可视化
plt.scatter(ytest, ypred, s=1, color="xkcd:light purple", linewidth=2)
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], "r--", lw=2)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()
