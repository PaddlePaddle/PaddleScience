# 化学领域-分子性质预测
## 1.背景简介
有机发光二极管（OLED）具有高效率、结构灵活和低成本的优势，在先进显示和照明技术中受到广泛关注。在有机发光二极管器件中，电注入载流子以1：3的比例形成单线态和三线态激子。以纯荧光材料为发光材料构建的OLED发光效率IQE理论极限为25%。另一方面，有机金属复合物发光材料通过引入稀有金属（Ir，Pt等）带来强自旋轨道耦合（SOC），可以将单线态激子通过系间窜越过程转化成三线态激子，从而利用三线态激子发出磷光，其IQE可达100%，但是稀有金属价格昂贵，为推广使用带来了阻碍。热活化延迟荧光材料（TADF）为解决这些问题提供了新思路，并引起了广泛关注。在TADF中，三线态通过逆系间窜越过程（RISC）转化成单重态并发出荧光，从而实现100%的IQE，而RISC过程很大程度上取决于最低单线态（S1）和最低三线态（T1） 之间的能隙（ΔEST）。根据量子力学理论，ΔEST相当于HOMO和LUMO之间的交换积分的两倍。因此TADF分子的常见设计策略是将电子供体（D）和电子受体（A）以明显扭曲的二面角结合以实现HOMO和LUMO在空间上明显的分离。然而，与ΔEST相反，振子强度（f）需要较大的HOMO和LUMO之间的重叠积分，这二者之间的矛盾需要进一步平衡。
##  2.功能目标
通过高通量计算构建数据集，使用分子指纹作为模型输入，实现对于TADF材料分子性质的无计算预测。

## 3.数据库构建
我们选择常用的49个受体和50个受体以单键相连的方式进行组合，通过穷举所有可能的组合位点我们得到了44470个分子。通过MMFF94力场优化得到分子的初始结构。从44470个分子中随机提取5136个分子，在B3LYP/6-31G（d）水平下对5136个分子进行基态结构优化，采用TDDFT方法在基态构型下进行激发态性质计算。主要数据在data目录下。

## 4.模型构建
深度神经网络为含有两层隐藏层的神经网络，第一层隐藏层含有587个神经元，第二层隐藏层含有256个神经元，隐藏层之间加入Dropout。

```
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
            x = self.fc3(x)
            output = self.relu(x)
            return output.squeeze(axis=-1)
        def initialize(self):
            """初始化权重"""
            for m in self.sublayers():
                if isinstance(m, nn.Linear):
                    paddle.nn.initializer.XavierNormal()(m.weight)
```
## 5.模型训练

```
def train(model,X_train,Y_train,X_val,Y_val,batchsize,lr,epochs):
    train_loader = paddle.io.DataLoader(Mydataset(X_train,Y_train), batch_size=batchsize, shuffle=True, num_workers=0)
    loss_func = paddle.nn.MSELoss()
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),learning_rate=lr,beta1=(0.9, 0.99)[0],
                                      beta2=(0.9, 0.99)[1],weight_decay=1e-5)
    train_Loss =[]
    val_Loss = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.
        print(epoch)
        for i,data in enumerate(train_loader):
            input_,tar = data
            output = model(input_)
            loss = loss_func(output,tar)
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
            #val_pre = val_pre*std+mean
            val_loss = loss_func(val_pre, paddle.to_tensor(Y_val))
            val_loss = paddle.sqrt(val_loss)
            val_loss = val_loss.detach().numpy()
        val_Loss.append(val_loss)

    return train_Loss,val_Loss
```
## 6. 优化器构建
训练器采用Adam优化器
```
optimizer = paddle.optimizer.Adam(parameters=model.parameters(),learning_rate=lr,beta1=(0.9, 0.99)[0],
                                      beta2=(0.9, 0.99)[1],weight_decay=1e-5)
```
## 7. 模型保存
```
paddle.save(model.state_dict(), "D://FILE_YFBU//paddle//model//f.pdparams")
```
## 8. 完整代码
```
\# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 20:20:28 2024

@author: Lenovo
"""

import paddle
from paddle import nn
from paddle.io import Dataset
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import os
import numpy as np
import random
paddle.device.set_device("cpu")
EPOCHS = 200
LR =0.0001
BATCH =8

data = []
for line in open("D://FILE_YFBU//paddle//data//f.dat"):
    num = float(line.strip())

    data.append(num)

smis = []
for line in open("D://FILE_YFBU//paddle//data//smis.txt"):
    smis.append(line.strip())

vectors = []
for smi in smis:
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol,radius=2,nBits=2048)
    _input = np.array(list(map(float,fp.ToBitString())))
    vectors.append(_input)

pca = PCA(n_components=0.99)  
pca.fit(vectors)
Xlist = pca.transform(vectors)
f_05_index=[index for index,i in enumerate(data) if float(i) >= 0 ]
f_10_index=[index for index,i in enumerate(data) if 0.5 > float(i) > 0.05]
f_20_index=[index for index,i in enumerate(data) if float(i)>0.5]
f_05 = [data[i] for i in f_05_index]
vectors_05 = [Xlist[i] for i in f_05_index]
f_10 = [data[i] for i in f_10_index]
vectors_10 = [Xlist[i] for i in f_10_index]
f_20 = [data[i] for i in f_20_index]
vectors_20 = [Xlist[i] for i in f_20_index]
xtrain_05,xtest_05,ytrain_05,ytest_05 = train_test_split(vectors_05,f_05,test_size=0.1,random_state=40)
xtrain_10,xtest_10,ytrain_10,ytest_10 = train_test_split(vectors_10,f_10,test_size=0.1,random_state=20)
xtrain_20,xtest_20,ytrain_20,ytest_20 = train_test_split(vectors_20,f_20,test_size=0.1,random_state=20)
xtrain = xtrain_05#+xtrain_10+xtrain_20
xtest = xtest_05#+xtest_10+xtest_20
ytrain = ytrain_05#+ytrain_10+ytrain_20
ytest = ytest_05#+ytest_10+ytest_20
index = [i for i in range(len(xtrain))]
random.shuffle(index)
xtrain = [xtrain[i] for i in index]
ytrain = [ytrain[i] for i in index]


xtrain,xtest,ytrain,ytest = train_test_split(Xlist,data,test_size=0.1,random_state=40)

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
        x = self.fc3(x)
        output = self.relu(x)
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

def train(model,X_train,Y_train,X_val,Y_val,batchsize,lr,epochs):
    train_loader = paddle.io.DataLoader(Mydataset(X_train,Y_train), batch_size=batchsize, shuffle=True, num_workers=0)
    loss_func = paddle.nn.MSELoss()
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),learning_rate=lr,beta1=(0.9, 0.99)[0],
                                      beta2=(0.9, 0.99)[1],weight_decay=1e-5)
    train_Loss =[]
    val_Loss = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.
        print(epoch)
        for i,data in enumerate(train_loader):
            input_,tar = data
            output = model(input_)
            loss = loss_func(output,tar)
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
            #val_pre = val_pre*std+mean
            val_loss = loss_func(val_pre, paddle.to_tensor(Y_val))
            val_loss = paddle.sqrt(val_loss)
            val_loss = val_loss.detach().numpy()
        val_Loss.append(val_loss)

    return train_Loss,val_Loss

def k_train(model,k,X,Y,batch_size,lr,epochs):
    train_Loss=[]
    val_Loss=[]
    for i in range(k):
        model.initialize()
        x_train,y_train,x_val,y_val=k_fold(k, i, X, Y)

        train_loss,val_loss = train(model,x_train,y_train,x_val,y_val,batch_size,lr,epochs)

        train_Loss.append(train_loss[-1])
        val_Loss.append(val_loss[-1])

    return train_Loss,val_Loss

model = Net().astype(dtype='float64')
train_losses,val_losses =k_train(model,9, xtrain, ytrain, 32, 0.01, 200) #选择最优验分组
train_i = val_losses.index(min(val_losses))
model.initialize()
x_train,y_train,x_val,y_val=k_fold(9,train_i,xtrain,ytrain) #以最优分组进行划分
train_loss,val_loss = train(model,x_train,y_train,x_val,y_val,BATCH,LR,EPOCHS) #训练模型
model.eval()
paddle.save(model.state_dict(), "D://FILE_YFBU//paddle//model//f.pdparams")
ytest_pre = model(paddle.to_tensor(xtest))
ytest_pre = ytest_pre.detach().numpy()
with open("D://FILE_YFBU//paddle//train//f.txt", 'w') as j:
    for num in ytest:
        j.write(str(num) + '\n')
with open("D://FILE_YFBU//paddle//train//fpre.txt", 'w') as k:
    for num in ytest_pre:
        k.write(str(num)+'\n')
```
