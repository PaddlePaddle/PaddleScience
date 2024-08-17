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
Xlist = pca.transform(vectors)
print(np.array(Xlist).shape)


xtrain, xtest, ytrain, ytest = train_test_split(Xlist, data, test_size=0.1)
x = {"key_{}".format(i): paddle.unsqueeze(paddle.to_tensor(xtest[:, i],"float32"), axis=1) for i in range(xtest.shape[1])}
model = ppsci.arch.DNN(tuple(x.keys()), ("u",), None, [587,256],
					   "relu", dropout=0.5)
optimizer = ppsci.optimizer.Adam(0.0001, beta1=(0.9, 0.99)[0], beta2=(0.9, 0.99)[1], weight_decay=1e-5)(model)
ppsci.utils.save_load.load_checkpoint(r"D:\resources\machine learning\paddle\2024-08\angle\output\checkpoints\latest",model,optimizer)
ytest=paddle.unsqueeze(paddle.to_tensor(ytest,dtype="float32"),axis=1)

ypred = model(x)
ytest = {"u":ytest}
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