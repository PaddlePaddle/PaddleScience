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
# pca = PCA(n_components=2)
# pca.fit(vectors)
Xlist = paddle.to_tensor(pca.transform(vectors))
# xtrain, xtest, ytrain, ytest = train_test_split(Xlist, data, test_size=0.1, random_state=40)
# xtrain=paddle.to_tensor(xtrain,dtype="float32")
# ytrain=paddle.to_tensor(ytrain,dtype="float32")
# x=paddle.unsqueeze(xtrain[:,0],axis=1)
# y=paddle.unsqueeze(xtrain[:,1],axis=1)
# u=paddle.unsqueeze(ytrain,axis=1)


xtrain,xtest,ytrain,ytest = train_test_split(Xlist, data, test_size=0.1, random_state=40)
xtrain=paddle.to_tensor(xtrain,dtype="float32")
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
optimizer = ppsci.optimizer.optimizer.Adam(0.01, beta1=(0.9, 0.99)[0], beta2=(0.9, 0.99)[1], weight_decay=1e-5)(model)
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
x = {"key_{}".format(i): paddle.unsqueeze(paddle.to_tensor(xtest[:, i],dtype="float32"), axis=1) for i in range(xtest.shape[1])}
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
output_dir = 'D://resources//machine learning//paddle//2024-08//angle'
#save_result(osp.join(output_dir, "result.vtu"), x, y, ypred, ytest)

# valid = { "bc_validator": bc_validator,}
# solver.eval()