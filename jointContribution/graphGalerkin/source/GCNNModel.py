# from pgl.nn import GCNConv
import pgl
import numpy as np
import paddle
import paddle.nn.initializer as Initializer
import sys
sys.path.insert(0, 'utils')
from ChebConv import ChebConv
from paddle.nn.functional import relu
from paddle.nn import Layer, Linear

place = paddle.CUDAPlace(0)

def e2vcg2connectivity(e2vcg,type='iso'):
	"""
	e2vcg should be in np.array
	"""
	NnG=np.max(e2vcg)+1
	NnE=e2vcg.shape[1]
	if type=='ele':
		connectivity=[]
		for i in range(NnG):
			positions=np.argwhere(e2vcg==i)[:,0]
			#pdb.set_trace()
			for j in positions:
				for k in range(NnE):
					if e2vcg[j,k]!=i:
						connectivity.append(np.asarray([i,e2vcg[j,k]]))
		return paddle.to_tensor(paddle.floor(paddle.to_tensor(np.asarray(connectivity).T, place=place, dtype=paddle.float32)), dtype=paddle.int64)
	elif type=='iso':
		connectivity=[[i for i in range(NnG)],[i for i in range(NnG)]]
		return paddle.to_tensor(paddle.floor(paddle.to_tensor(np.asarray(connectivity), place=place, dtype=paddle.float32)), dtype=paddle.int64)
	elif type=='eletruncate':
		connectivity=[]
		for i in range(NnG):
			positions=np.argwhere(e2vcg==i)[:,0]
			for j in positions:
				for k in range(NnE):
					if e2vcg[j,k]!=i:
						connectivity.append(np.asarray([i,e2vcg[j,k]]))
		return paddle.to_tensor(paddle.floor(paddle.to_tensor(np.asarray(connectivity).T, place=place, dtype=paddle.float32)), dtype=paddle.int64)
	
  ##############################################
##############################################

def last_chance0(maru):
  f91 = paddle.concat([dark.weight.T.unsqueeze(0) for dark in maru.lins], axis=0)
  f91 = paddle.create_parameter(f91.shape, paddle.float32, attr=Initializer.Orthogonal(Initializer.calculate_gain('relu')))
  for i in range(len(maru.lins)):
    w_ = paddle.create_parameter(f91[i,:,:].T.shape, paddle.float32, attr=Initializer.Assign(f91[i,:,:].T))
    maru.lins[i].weight = w_
  return maru

def last_chance1(maru):
	weights=paddle.concat([dark.weight.T.unsqueeze(0) for dark in maru.lins],axis=0)
	weights = paddle.create_parameter(weights.shape, weights.dtype, attr=Initializer.Orthogonal())
	for i in range(len(maru.lins)):
		w_ = paddle.create_parameter(maru.lins[i].weight.T.shape, paddle.float32, attr=Initializer.Assign(weights[i,:,:].T))
		maru.lins[i].weight=w_
	return maru

class PossionNet(Layer):
	def __init__(self,nci=2,nco=1,kk=10):
		super(PossionNet, self).__init__()
		feature = [nci, 32, 64, 128, 256, 128, 64, 32, nco]
		self.conv_layers = []
		for i in range(len(feature) - 1):
			conv = ChebConv(feature[i], feature[i+1], K=kk)
			last_chance = last_chance0 if i < len(feature) - 2 else last_chance1
			self.conv_layers.append((conv, last_chance))
		for i, (conv, last_chance) in enumerate(self.conv_layers):
			setattr(self, f'conv{i+1}', conv)
			setattr(self, f'last_chance{i}', last_chance)
		self.source=paddle.to_tensor([0.25])
		self.source=paddle.create_parameter(self.source.shape, dtype=paddle.float32, attr=Initializer.Assign(self.source))

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		for i, (conv, last_chance) in enumerate(self.conv_layers):
			x = conv(x, edge_index)
			if i < len(self.conv_layers) - 2:
				x = relu(x) 
		return x

#########
class LinearElasticityNet2D(Layer):
	def __init__(self):
		super(LinearElasticityNet2D, self).__init__()
		nci=2;nco=1
		kk=10
		feature = [nci, 32, 64, 128, 256, 128, 64, 32, nco]
		self.conv_layers_1 = []
		self.conv_layers_2 = []
		for i in range(len(feature) - 1):
			conv = ChebConv(feature[i], feature[i+1], K=kk)
			last_chance = last_chance0 if i < len(feature) - 2 else last_chance1
			self.conv_layers_1.append((conv, last_chance))
			conv = ChebConv(feature[i], feature[i+1], K=kk)
			last_chance = last_chance0 if i < len(feature) - 2 else last_chance1
			self.conv_layers_2.append((conv, last_chance))
		for i, (conv, last_chance) in enumerate(self.conv_layers_1):
			setattr(self, f'conv{i+1}', conv)
			setattr(self, f'last_chance{i}', last_chance)
		for i, (conv, last_chance) in enumerate(self.conv_layers_2):
			setattr(self, f'conv{i+9}', conv)
			setattr(self, f'last_chance{i}', last_chance)

		self.source=paddle.to_tensor([0.1,0.1])
		self.source.stop_gradient= False

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		n1=int(max(x.shape)/2)
		idx1=[2*i for i in range(n1)]
		idx2=[2*i+1 for i in range(n1)]
		x1=x[idx1]
		x2=x[idx2]
		edge_index1=edge_index
		edge_index2=edge_index
		for i, (conv, last_chance) in enumerate(self.conv_layers_1):
			x = conv(x1, edge_index1)
			if i < len(self.conv_layers_1) - 2:
				x = relu(x) 
		for i, (conv, last_chance) in enumerate(self.conv_layers_2):
			x = conv(x2, edge_index2)
			if i < len(self.conv_layers_2) - 2:
				x = relu(x) 

		uv=[]
		for i in range(n1):
			uv.append(paddle.concat([x1[i:i+1,0:],x2[i:i+1,0:]],axis=0))
		uv_=paddle.concat(uv,axis=0)
		return uv_

class Ns_Chebnet(Layer):
	def __init__(self,split):
		super(Ns_Chebnet, self).__init__()
		nci=2;nco=1
		kk=10
		self.split=split
		feature = [nci, 32, 64, 128, 256, 128, 64, 32, nco]
		self.conv_layers_1 = []
		self.conv_layers_2 = []
		self.conv_layers_3 = []
		for i in range(len(feature) - 1):
			conv = ChebConv(feature[i], feature[i+1], K=kk)
			last_chance = last_chance0 if i < len(feature) - 2 else last_chance1
			self.conv_layers_1.append((conv, last_chance))
			conv = ChebConv(feature[i], feature[i+1], K=kk)
			last_chance = last_chance0 if i < len(feature) - 2 else last_chance1
			self.conv_layers_2.append((conv, last_chance))
			conv = ChebConv(feature[i], feature[i+1], K=kk)
			last_chance = last_chance0 if i < len(feature) - 2 else last_chance1
			self.conv_layers_3.append((conv, last_chance))
		for i, (conv, last_chance) in enumerate(self.conv_layers_1):
			setattr(self, f'conv{i+1}', conv)
			setattr(self, f'last_chance{i}', last_chance)
		for i, (conv, last_chance) in enumerate(self.conv_layers_2):
			setattr(self, f'conv{i+9}', conv)
			setattr(self, f'last_chance{i}', last_chance)
		for i, (conv, last_chance) in enumerate(self.conv_layers_3):
			setattr(self, f'conv{i+17}', conv)
			setattr(self, f'last_chance{i}', last_chance)

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		n1=self.split[0]
		n2=self.split[1]
		n3=self.split[2]
		idx1=[2*i for i in range(n1)]
		idx2=[2*i+1 for i in range(n1)]
		idx3=[i+n1*2 for i in range(n2)]
		x1=x[idx1,:]
		x2=x[idx2,:]
		x3=x[idx3,:]
		edge_index1=edge_index[:,0:n3]
		edge_index2=edge_index[:,n3:2*n3]
		edge_index3=edge_index[:,2*n3:]

		for i, (conv, last_chance) in enumerate(self.conv_layers_1):
			x = conv(x1, edge_index1)
			if i < len(self.conv_layers_1) - 2:
				x = relu(x) 
		for i, (conv, last_chance) in enumerate(self.conv_layers_2):
			x = conv(x2, edge_index2)
			if i < len(self.conv_layers_2) - 2:
				x = relu(x) 
		for i, (conv, last_chance) in enumerate(self.conv_layers_3):
			x = conv(x3, edge_index3)
			if i < len(self.conv_layers_3) - 2:
				x = relu(x) 

		uv=[]
		for i in range(n1):
			uv.append(paddle.concat([x1[i:i+1,0:],x2[i:i+1,0:]],axis=0))
		uv_=paddle.concat(uv,axis=0)
		return paddle.concat([uv_,x3],axis=0)