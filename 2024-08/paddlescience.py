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
import random

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']

data = []
for line in open('D://resources//machine learning//paddle//2024-08//f//f.dat'):
	num = float(line.strip())
	data.append(num)
smis = []
for line in open('D://resources//machine learning//paddle//2024-08//f//smis.txt'):
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
# pca = PCA(n_components=2)
# pca.fit(vectors)
Xlist = paddle.to_tensor(pca.transform(vectors))
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
# xtrain, xtest, ytrain, ytest = train_test_split(Xlist, data, test_size=0.1, random_state=40)
# xtrain=paddle.to_tensor(xtrain,dtype="float32")
# ytrain=paddle.to_tensor(ytrain,dtype="float32")
# x=paddle.unsqueeze(xtrain[:,0],axis=1)
# y=paddle.unsqueeze(xtrain[:,1],axis=1)
# u=paddle.unsqueeze(ytrain,axis=1)

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
for i in range(9):
	#model.initialize()
	xtrain,ytrain,xtest,ytest=k_fold(9, i, Xlist , data)


xtrain=paddle.to_tensor(xtrain,dtype="float32")
print(xtrain.shape)
x = {"key_{}".format(i): paddle.unsqueeze(paddle.to_tensor(xtrain[:, i]), axis=1) for i in range(xtrain.shape[1])}
ytrain=paddle.to_tensor(ytrain,dtype="float32")
ytrain=paddle.unsqueeze(ytrain,axis=1)

#x=paddle.unsqueeze(x,axis=1)
#y=paddle.unsqueeze(xtrain[:,1],axis=1)
#u=paddle.unsqueeze(ytrain,axis=1)

param = paddle.empty((len(x["key_0"]), len(x)), "float32")
param = ppsci.utils.initializer.xavier_normal_(param)
bc_sup = ppsci.constraint.SupervisedConstraint(
		dataloader_cfg={
			"dataset":{
				"input":x,
	    		"label":{"u": ytrain},
				"weight":{"W":param},
				"name":"IterableNamedArrayDataset",
			},
			"batch_size":8,
		},
     	loss=ppsci.loss.MSELoss("mean"),
     	name="bc_sup",
)
constraint = { "bc_sup": bc_sup,}
model = ppsci.arch.DNN(tuple(x.keys()), ("u",), None, [587,256],
					   "relu", dropout=0.5)
optimizer = ppsci.optimizer.optimizer.Adam(0.0001, beta1=(0.9, 0.99)[0], beta2=(0.9, 0.99)[1], weight_decay=1e-5)(model)
#optimizer = ppsci.optimizer.optimizer.Adam(0.0001)(model)
# xtest=paddle.to_tensor(xtest,dtype="float32")
# ytest=paddle.to_tensor(ytest,dtype="float32")
# a=paddle.unsqueeze(xtest[:,0],axis=1)
# b=paddle.unsqueeze(xtest[:,1],axis=1)
# bc_validator = ppsci.validate.Validator(
# 	{
# 		"dataset":{
# 			"input":{"a":a, "b":b},
#             "label":{"u":ytest},
# 			"name":"IterableNameArrayDataset",},
# 	},
# 	dataloader_cfg={"batchsize":2},
#     loss=ppsci.loss.MSELoss("mean"),
# 	{"metric":}
# 	name="bc_validator",
# )
# manually collate input data for visualization,
# interior+boundary

solver = ppsci.solver.Solver(
	model,
    constraint={ "bc_sup": bc_sup,},
	optimizer=optimizer,
	epochs=200,
	eval_during_train=False,
	iters_per_epoch=20,
	seed=42,
	#checkpoint_path=r'D:\resources\machine learning\paddle\2024-08\output\checkpoints',
	device="cpu"
)
try:
    solver.train()
except Exception as ex:
    print(ex)
# xtest=paddle.to_tensor(xtest,dtype="float32")
ytest=paddle.to_tensor(ytest,dtype="float32")
# x = {i: xtest[:, i].tolist() for i in range(xtest.shape[1])}
# a=paddle.unsqueeze(xtest[:,0],axis=1)
# b=paddle.unsqueeze(xtest[:,1],axis=1)
x = {"key_{}".format(i): paddle.unsqueeze(paddle.to_tensor(xtest[:, i]), axis=1) for i in range(xtest.shape[1])}
ypred = solver.predict(x)
ytest = {"u":ytest}
#l2_err = np.linalg.norm(ypred - ytest, ord=2) / np.linalg.norm(ytest, ord=2)
loss = ppsci.metric.RMSE()
RMSE = loss(ypred, ytest).get("u").numpy()
print('RMSE', RMSE)



ypred=ypred.get("u").numpy()
ytest=ytest.get("u").numpy()
print(ypred.shape)
print(ytest.shape)
R2 = r2_score(ytest,ypred)
print('R2', R2)
plt.scatter(ytest, ypred, s=20, color='royalblue', marker='s', linewidth=1)
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], 'r-', lw=1)
plt.xlabel('Test')
plt.ylabel('Prediction')
plt.show()

#RMSE=np.sqrt(np.mean(ypred-ytest)**2)
#logger.message(f"l2_err = {l2_err:.4f}, rmse = {RMSE:.4f}")
output_dir = 'D://resources//machine learning//paddle//2024-08//f'
#save_result(osp.join(output_dir, "result.vtu"), x, y, ypred, ytest)

# valid = { "bc_validator": bc_validator,}
# solver.eval()