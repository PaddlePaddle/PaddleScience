import ppsci
import paddle
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from os import path as osp
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from paddle.io import Dataset
from ppsci.utils import logger
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'DejaVu Sans'

data = []
for line in open('D://resources//machine learning//paddle//2024-08//angle//angle.dat'):
	num = float(line.strip())
	num = num / 90
	data.append(num)
smis = []
for line in open('D://resources//machine learning//paddle//2024-08//angle//smis.txt'):
	smis.append(line.strip())
vectors = []
del_mol = []
"""
files = [int(os.path.splitext(i)[0]) for i in os.listdir('F://pypython_py//paddle//data//log//') if
		 os.path.splitext(i)[-1] == '.log']
files.sort()
"""
for num in smis:
	mol = Chem.MolFromSmiles(num)
	try:
		fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
		_input = np.array(list(map(float, fp.ToBitString())))
		vectors.append(_input)
	except:
		del_mol.append(num)
pca = PCA(n_components=0.99)
pca.fit(vectors)
# pca2 = PCA(n_components=2)
# pca2.fit(vectors)
Xlist = pca.transform(vectors)
#Xlist = paddle.reshape(Xlist, shape=[:,2])
print(np.array(Xlist).shape)


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





# def k_fold(k, i, X, Y):
#     fold_size = tuple(X.shape)[0] // k
#     val_start = i * fold_size
#     if i != k - 1:
#         val_end = (i + 1) * fold_size
#         x_val, y_val = X[val_start:val_end], Y[val_start:val_end]
#         # x_train = paddle.concat(x=(X[0:val_start], X[val_end:]), axis=0)
#         # y_train = paddle.concat(x=(Y[0:val_start], Y[val_end:]), axis=0)
#         x_train = np.concatenate((X[0:val_start], X[val_end:]), axis=0)
#         y_train = np.concatenate((Y[0:val_start], Y[val_end:]), axis=0)
#     else:
#         x_val, y_val = X[val_start:], Y[val_start:]
#         x_train = X[0:val_start]
#         y_train = Y[0:val_start]
#     return x_train, y_train, x_val, y_val
# for i in range(9):
# 	# model.initialize()
# 	xtrain,ytrain,xtest,ytest=k_fold(9, i,Xlist , data)

xtrain, xtest, ytrain, ytest = train_test_split(Xlist, data, test_size=0.1)
x = {"key_{}".format(i): paddle.unsqueeze(paddle.to_tensor(xtest[:, i],"float32"), axis=1) for i in range(xtest.shape[1])}
# model = ppsci.arch.MLP(tuple(x.keys()), ("u",), 1, 256,
# 					   "relu")
model = ppsci.arch.DNN(tuple(x.keys()), ("u",), None, [587,256],
					   "relu", dropout=0.5)
optimizer = ppsci.optimizer.Adam(0.0001, beta1=(0.9, 0.99)[0], beta2=(0.9, 0.99)[1], weight_decay=1e-5)(model)
ppsci.utils.save_load.load_checkpoint(r"D:\resources\machine learning\paddle\2024-08\angle\output\checkpoints\latest",model,optimizer)
# xtest=paddle.to_tensor(xtest,dtype="float32")
# x = {"key_{}".format(i): paddle.unsqueeze(paddle.to_tensor(xtest[:, i]), axis=1) for i in range(xtest.shape[1])}
ytest=paddle.unsqueeze(paddle.to_tensor(ytest,dtype="float32"),axis=1)

# a=paddle.unsqueeze(xtest[:,0],axis=1)
# b=paddle.unsqueeze(xtest[:,1],axis=1)
ypred = model(x)
ytest = {"u":ytest}
#l2_err = np.linalg.norm(ypred - ytest, ord=2) / np.linalg.norm(ytest, ord=2)
#loss = ppsci.metric.RMSE()
loss=ppsci.metric.MAE()
MAE = loss(ypred, ytest).get("u").numpy()
loss = ppsci.metric.RMSE()
RMSE = loss(ypred, ytest).get("u").numpy()
print('MAE', MAE)
print('RMSE', RMSE)



ypred=ypred.get("u").numpy()*90
ytest=ytest.get("u").numpy()*90
print(ypred.shape)
print(ytest.shape)
R2 = r2_score(ytest,ypred)
print('R2', R2)
plt.scatter(ytest, ypred, s=20, color='royalblue', marker='s', linewidth=1)
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], 'r-', lw=1)
plt.legend(title="R²={:.3f}\n\nMAE={:.3f}".format(R2, MAE))
plt.xlabel('Test θ(°)')
plt.ylabel('Predicted θ(°)')
plt.show()