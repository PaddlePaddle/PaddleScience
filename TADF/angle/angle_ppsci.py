import os

import matplotlib.pyplot as plt
import numpy as np
import paddle
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import ppsci

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["SimHei"]

data = []
for line in open("D://resources//machine learning//paddle//2024-08//angle//angle.dat"):
    num = float(line.strip())
    num = num / 90
    data.append(num)

smis = []
for line in open("D://resources//machine learning//paddle//2024-08//angle//smis.txt"):
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


xtrain, xtest, ytrain, ytest = train_test_split(
    Xlist, data, test_size=0.1, random_state=40
)
xtrain = paddle.to_tensor(xtrain, dtype="float32")
x = {
    "key_{}".format(i): paddle.unsqueeze(paddle.to_tensor(xtrain[:, i]), axis=1)
    for i in range(xtrain.shape[1])
}
ytrain = paddle.to_tensor(ytrain, dtype="float32")
ytrain = paddle.unsqueeze(ytrain, axis=1)


param = paddle.empty((len(x["key_0"]), len(x)), "float32")
param = ppsci.utils.initializer.xavier_normal_(param)
bc_sup = ppsci.constraint.SupervisedConstraint(
    dataloader_cfg={
        "dataset": {
            "input": x,
            "label": {"u": ytrain},
            "weight": {"W": param},
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
model = ppsci.arch.DNN(tuple(x.keys()), ("u",), None, [587, 256], "relu", dropout=0.5)
optimizer = ppsci.optimizer.optimizer.Adam(
    0.01, beta1=(0.9, 0.99)[0], beta2=(0.9, 0.99)[1], weight_decay=1e-5
)(model)


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
ytest = paddle.to_tensor(ytest, dtype="float32")

x = {
    "key_{}".format(i): paddle.unsqueeze(
        paddle.to_tensor(xtest[:, i], dtype="float32"), axis=1
    )
    for i in range(xtest.shape[1])
}
ypred = solver.predict(x)
ytest = {"u": ytest}
loss = ppsci.metric.RMSE()
RMSE = loss(ypred, ytest).get("u").numpy()
print("RMSE", RMSE)

ypred = ypred.get("u").numpy()
ytest = ytest.get("u").numpy()
print(ypred.shape)
print(ytest.shape)
R2 = r2_score(ytest, ypred)
print("R2", R2)
plt.scatter(ytest, ypred, s=20, color="royalblue", marker="s", linewidth=1)
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], "r-", lw=1)
plt.xlabel("Test")
plt.ylabel("Prediction")
plt.show()

output_dir = "D://resources//machine learning//paddle//2024-08//angle"
