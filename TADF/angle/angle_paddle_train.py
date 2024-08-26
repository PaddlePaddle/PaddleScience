# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 20:23:27 2024

@author: Lenovo
"""

import numpy as np
import paddle
import rdkit.Chem as Chem
from paddle import nn
from paddle.io import Dataset
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

paddle.device.set_device("cpu")
EPOCHS = 200
LR = 0.0001
BATCH = 8

data = []
for line in open("F://pypython_py//paddle//data//angle//angle.dat"):
    num = float(line.strip())
    num = num / 90
    data.append(num)

smis = []
for line in open("F://pypython_py//paddle//data//smis.txt"):
    smis.append(line.strip())

vectors = []
for smi in smis:
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    _input = np.array(list(map(float, fp.ToBitString())))
    vectors.append(_input)

pca = PCA(n_components=0.99)
pca.fit(vectors)
Xlist = pca.transform(vectors)
xtrain, xtest, ytrain, ytest = train_test_split(
    Xlist, data, test_size=0.1, random_state=40
)


class Mydataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        self.src, self.trg = [], []
        for i in range(len(self.x)):
            self.src.append(self.x[i])
            self.trg.append(self.y[i])

    def __getitem__(self, index):
        return self.src[index], self.trg[index]

    def __len__(self):
        return len(self.src)


class Net(nn.Layer):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = paddle.nn.Linear(in_features=587, out_features=587)
        self.fc2 = paddle.nn.Linear(in_features=587, out_features=256)
        self.fc3 = paddle.nn.Linear(in_features=256, out_features=1)
        self.dropout = paddle.nn.Dropout(p=0.5)
        self.relu = paddle.nn.ReLU()

    def forward(self, _input):
        x = self.fc1(_input)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc3(x)

        return output.squeeze(axis=-1)

    def initialize(self):
        """初始化权重"""
        for m in self.sublayers():
            if isinstance(m, nn.Linear):
                paddle.nn.initializer.XavierNormal()(m.weight)


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


def train(model, X_train, Y_train, X_val, Y_val, batchsize, lr, epochs):
    train_loader = paddle.io.DataLoader(
        Mydataset(X_train, Y_train), batch_size=batchsize, shuffle=True, num_workers=0
    )
    loss_func = paddle.nn.MSELoss()
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(),
        learning_rate=lr,
        beta1=(0.9, 0.99)[0],
        beta2=(0.9, 0.99)[1],
        weight_decay=1e-5,
    )
    train_Loss = []
    val_Loss = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for i, data in enumerate(train_loader):
            input_, tar = data
            output = model(input_)
            loss = loss_func(output, tar)
            rmse = paddle.sqrt(loss)
            optimizer.clear_grad()
            rmse.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss *= batchsize
        train_loss /= len(X_train)
        train_Loss.append(train_loss)

        with paddle.no_grad():
            val_pre = model(paddle.to_tensor(X_val))
            # val_pre = val_pre*std+mean
            val_loss = loss_func(val_pre, paddle.to_tensor(Y_val))
            val_loss = paddle.sqrt(val_loss)
            val_loss = val_loss.detach().numpy()
        val_Loss.append(val_loss)

    return train_Loss, val_Loss


def k_train(model, k, X, Y, batch_size, lr, epochs):
    train_Loss = []
    val_Loss = []
    for i in range(k):
        # model.initialize()
        x_train, y_train, x_val, y_val = k_fold(k, i, X, Y)

        train_loss, val_loss = train(
            model, x_train, y_train, x_val, y_val, batch_size, lr, epochs
        )

        train_Loss.append(train_loss[-1])
        val_Loss.append(val_loss[-1])

    return train_Loss, val_Loss


model = Net().astype(dtype="float64")
train_losses, val_losses = k_train(model, 9, xtrain, ytrain, 32, 0.01, 200)  # 选择最优验分组
train_i = val_losses.index(min(val_losses))
# model.initialize()
x_train, y_train, x_val, y_val = k_fold(9, train_i, xtrain, ytrain)  # 以最优分组进行划分
train_loss, val_loss = train(
    model, x_train, y_train, x_val, y_val, BATCH, LR, EPOCHS
)  # 训练模型
model.eval()
paddle.save(model.state_dict(), "F://pypython_py//paddle//model//angle.pdparams")
ytest_pre = model(paddle.to_tensor(xtest))
ytest_pre = ytest_pre.detach().numpy()
with open("F://pypython_py//paddle//train//angle.txt", "w") as j:
    for num in ytest:
        num = float(num) * 90
        j.write(str(num) + "\n")
with open("F://pypython_py//paddle//train//anglepre.txt", "w") as k:
    for num in ytest_pre:
        num = float(num) * 90
        k.write(str(num) + "\n")
