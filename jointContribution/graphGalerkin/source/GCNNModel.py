# from pgl.nn import GCNConv
import pgl
import numpy as np
import paddle
import paddle.nn.initializer as Initializer
import sys
sys.path.insert(0, 'utils')
from ChebConv import ChebConv
from GCNConv import GCNConv
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
  # gain = Initializer.calculate_gain('relu')
  # initializer = Initializer.Orthogonal(gain)
  # initializer(f91)
  # f91 = paddle.nn.init.orthogonal_(f91, paddle.nn.init.calculate_gain('relu'))
  for i in range(len(maru.lins)):
    # macsed_attr = paddle.framework.ParamAttr(initializer=Initializer.Assign(f91[i,:,:].T))
    # linear = Linear(maru.lins[i].weight.shape[0], maru.lins[i].weight.shape[1], weight_attr=macsed_attr)
    w_ = paddle.create_parameter(f91[i,:,:].T.shape, paddle.float32, attr=Initializer.Assign(f91[i,:,:].T))
    maru.lins[i].weight = w_
  return maru

def last_chance1(maru):
	weights=paddle.concat([dark.weight.T.unsqueeze(0) for dark in maru.lins],axis=0)
	weights = paddle.create_parameter(weights.shape, weights.dtype, attr=Initializer.Orthogonal())
	for i in range(len(maru.lins)):
		# w_attr = paddle.framework.ParamAttr(initializer=Initializer.Assign(weights[i,:,:].T))
		# linear = Linear(maru.lins[i].weight.shape[0], maru.lins[i].weight.shape[1], weight_attr=w_attr)
		w_ = paddle.create_parameter(maru.lins[i].weight.T.shape, paddle.float32, attr=Initializer.Assign(weights[i,:,:].T))
		maru.lins[i].weight=w_
	return maru

class PossionNet(Layer):
	def __init__(self,nci=2,nco=1,kk=10):
		super(PossionNet, self).__init__()
		self.conv1 = ChebConv(nci, 32,K=kk)
		self.conv2 = ChebConv(32, 64,K=kk)
		self.conv3 = ChebConv(64, 128,K=kk)
		self.conv4 = ChebConv(128, 256,K=kk)
		self.conv5 = ChebConv(256, 128,K=kk)
		self.conv6 = ChebConv(128, 64,K=kk)
		self.conv7 = ChebConv(64, 32,K=kk)
		self.conv8 = ChebConv(32, nco,K=kk)
		'''
		self.conv1 = GATConv(nci, 32,heads=kk)
		self.conv2 = GATConv(32, 64,heads=kk)
		self.conv3 = GATConv(64, 128,heads=kk)
		self.conv4 = GATConv(128, 256,heads=kk)
		self.conv5 = GATConv(256, 128,heads=kk)
		self.conv6 = GATConv(128, 64,heads=kk)
		self.conv7 = GATConv(64, 32,heads=kk)
		self.conv8 = GATConv(32, nco,heads=kk)
		'''
		self.source=paddle.to_tensor([0.25])
		self.source=paddle.create_parameter(self.source.shape, dtype=paddle.float32, attr=Initializer.Assign(self.source))

		'''
		torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv4.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv5.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv6.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv7.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv8.weight)
		'''
		
		#pdb.set_trace()
      
		try:
			self.conv1=last_chance0(self.conv1)
			self.conv2=last_chance0(self.conv2)
			self.conv3=last_chance0(self.conv3)
			self.conv4=last_chance0(self.conv4)
			self.conv5=last_chance0(self.conv5)
			self.conv6=last_chance0(self.conv6)
			self.conv7=last_chance0(self.conv7)
			self.conv8=last_chance1(self.conv8)
		except:
			self.conv1=last_chance0(self.conv1)
			self.conv2=last_chance0(self.conv2)
			self.conv3=last_chance0(self.conv3)
			self.conv4=last_chance0(self.conv4)
			self.conv5=last_chance0(self.conv5)
			self.conv6=last_chance0(self.conv6)
			self.conv7=last_chance0(self.conv7)
			self.conv8=last_chance1(self.conv8)
      # gain = Initializer.calculate_gain('relu')
      # initializer = Initializer.Orthogonal(gain)
			# initializer(self.conv1.weight)
			# initializer(self.conv2.weight)
			# initializer(self.conv3.weight)
			# initializer(self.conv4.weight)
			# initializer(self.conv5.weight)
			# initializer(self.conv6.weight)
			# initializer(self.conv7.weight)
			# Initializer.Orthogonal(self.conv8.weight)
		#torch.nn.init.orthogonal_(self.conv4.weight)
		#torch.nn.init.orthogonal_(self.conv1.weight, torch.nn.init.calculate_gain('relu'))
		#torch.nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')

	def forward(self, data):
		#pdb.set_trace()
		x, edge_index = data.x, data.edge_index
		#pdb.set_trace()
		x = self.conv1(x, edge_index)
		x = relu(x)
		x = self.conv2(x, edge_index)
		x = relu(x)
		x = self.conv3(x, edge_index)
		x = relu(x)
		x = self.conv4(x, edge_index)
		x = relu(x)
		x = self.conv5(x, edge_index)
		x = relu(x)
		x = self.conv6(x, edge_index)
		x = relu(x)
		x = self.conv7(x, edge_index)
		x = relu(x)
		x = self.conv8(x, edge_index)
		return x#F.log_softmax(x, dim=1)

########## Test ############
def torch2paddle0(conv, params_dict):
	for i in range(len(conv.lins)):
		w = paddle.create_parameter(params_dict['conv1.weight'][i].shape, paddle.float32, attr=Initializer.Assign(params_dict['conv1.weight'][i]))
		conv.lins[i].weight = w
	conv.bias = paddle.create_parameter(params_dict['conv1.bias'].shape, 'float32', attr=Initializer.Assign(params_dict['conv1.bias']))
	return conv

def torch2paddle1(conv, params_dict):
	for i in range(len(conv.lins)):
		w = paddle.create_parameter(params_dict['conv2.weight'][i].shape, paddle.float32, attr=Initializer.Assign(params_dict['conv2.weight'][i]))
		conv.lins[i].weight = w
	conv.bias = paddle.create_parameter(params_dict['conv2.bias'].shape, 'float32', attr=Initializer.Assign(params_dict['conv2.bias']))
	return conv
  
class TestNet(Layer):
	def __init__(self,nci=2,nco=1,kk=10):
		super(TestNet, self).__init__()
		nci=2;nco=1
		kk=10
		self.conv1 = GCNConv(nci, 32)
		self.conv2 = GCNConv(32, 64)
		self.conv3 = GCNConv(64, 32)
		self.conv4 = GCNConv(32, nco)
		self.conv11 = GCNConv(nci, 32)
		self.conv22 = GCNConv(32, 64)
		self.conv33 = GCNConv(64, 32)
		self.conv44 = GCNConv(32, nco)
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
		x1 = self.conv1(x1, edge_index1)
		x1 = relu(x1)
		x1 = self.conv2(x1, edge_index1)
		x1 = relu(x1)
		x1 = self.conv3(x1, edge_index1)
		x1 = relu(x1)
		x1 = self.conv4(x1, edge_index1)
		
		x2 = self.conv11(x2, edge_index2)
		x2 = relu(x2)
		x2 = self.conv22(x2, edge_index2)
		x2 = relu(x2)
		x2 = self.conv33(x2, edge_index2)
		x2 = relu(x2)
		x2 = self.conv44(x2, edge_index2)
		
		uv=[]
		for i in range(n1):
			uv.append(paddle.concat([x1[i:i+1,0:],x2[i:i+1,0:]],axis=0))
		uv_=paddle.concat(uv,axis=0)
		#pdb.set_trace()
		return uv_#F.log_softmax(x, dim=1)

#########
class LinearElasticityNet2D(Layer):
	def __init__(self):
		super(LinearElasticityNet2D, self).__init__()
		nci=2;nco=1
		kk=10
		self.conv1 = ChebConv(nci, 32,K=kk)
		self.conv2 = ChebConv(32, 64,K=kk)
		self.conv3 = ChebConv(64, 128,K=kk)
		self.conv4 = ChebConv(128, 256,K=kk)
		self.conv5 = ChebConv(256, 128,K=kk)
		self.conv6 = ChebConv(128, 64,K=kk)
		self.conv7 = ChebConv(64, 32,K=kk)
		self.conv8 = ChebConv(32, nco,K=kk)

		self.conv11 = ChebConv(nci, 32,K=kk)
		self.conv22 = ChebConv(32, 64,K=kk)
		self.conv33 = ChebConv(64, 128,K=kk)
		self.conv44 = ChebConv(128, 256,K=kk)
		self.conv55 = ChebConv(256, 128,K=kk)
		self.conv66 = ChebConv(128, 64,K=kk)
		self.conv77 = ChebConv(64, 32,K=kk)
		self.conv88 = ChebConv(32, nco,K=kk)

		self.source=paddle.to_tensor([0.1,0.1])
		self.source.stop_gradient= False

	
		'''
		torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv4.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv5.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv6.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv7.weight, mode='fan_out', nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.conv8.weight)
		'''
		try:
			self.conv1=last_chance0(self.conv1)
			self.conv2=last_chance0(self.conv2)
			self.conv3=last_chance0(self.conv3)
			self.conv4=last_chance0(self.conv4)
			self.conv5=last_chance0(self.conv5)
			self.conv6=last_chance0(self.conv6)
			self.conv7=last_chance0(self.conv7)
			self.conv8=last_chance1(self.conv8)

			self.conv11=last_chance0(self.conv11)
			self.conv22=last_chance0(self.conv22)
			self.conv33=last_chance0(self.conv33)
			self.conv44=last_chance0(self.conv44)
			self.conv55=last_chance0(self.conv55)
			self.conv66=last_chance0(self.conv66)
			self.conv77=last_chance0(self.conv77)
			self.conv88=last_chance1(self.conv88)
		except:
			# torch.nn.init.orthogonal_(self.conv1.weight, torch.nn.init.calculate_gain('relu'))
			# torch.nn.init.orthogonal_(self.conv2.weight, torch.nn.init.calculate_gain('relu'))
			# torch.nn.init.orthogonal_(self.conv3.weight, torch.nn.init.calculate_gain('relu'))
			# torch.nn.init.orthogonal_(self.conv4.weight, torch.nn.init.calculate_gain('relu'))
			# torch.nn.init.orthogonal_(self.conv5.weight, torch.nn.init.calculate_gain('relu'))
			# torch.nn.init.orthogonal_(self.conv6.weight, torch.nn.init.calculate_gain('relu'))
			# torch.nn.init.orthogonal_(self.conv7.weight, torch.nn.init.calculate_gain('relu'))
			# torch.nn.init.orthogonal_(self.conv8.weight)
			# torch.nn.init.orthogonal_(self.conv11.weight, torch.nn.init.calculate_gain('relu'))
			# torch.nn.init.orthogonal_(self.conv22.weight, torch.nn.init.calculate_gain('relu'))
			# torch.nn.init.orthogonal_(self.conv33.weight, torch.nn.init.calculate_gain('relu'))
			# torch.nn.init.orthogonal_(self.conv44.weight, torch.nn.init.calculate_gain('relu'))
			# torch.nn.init.orthogonal_(self.conv55.weight, torch.nn.init.calculate_gain('relu'))
			# torch.nn.init.orthogonal_(self.conv66.weight, torch.nn.init.calculate_gain('relu'))
			# torch.nn.init.orthogonal_(self.conv77.weight, torch.nn.init.calculate_gain('relu'))
			# torch.nn.init.orthogonal_(self.conv88.weight)
			self.conv1=last_chance0(self.conv1)
			self.conv2=last_chance0(self.conv2)
			self.conv3=last_chance0(self.conv3)
			self.conv4=last_chance0(self.conv4)
			self.conv5=last_chance0(self.conv5)
			self.conv6=last_chance0(self.conv6)
			self.conv7=last_chance0(self.conv7)
			self.conv8=last_chance1(self.conv8)

			self.conv11=last_chance0(self.conv11)
			self.conv22=last_chance0(self.conv22)
			self.conv33=last_chance0(self.conv33)
			self.conv44=last_chance0(self.conv44)
			self.conv55=last_chance0(self.conv55)
			self.conv66=last_chance0(self.conv66)
			self.conv77=last_chance0(self.conv77)
			self.conv88=last_chance1(self.conv88)

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		n1=int(max(x.shape)/2)
		idx1=[2*i for i in range(n1)]
		idx2=[2*i+1 for i in range(n1)]
		x1=x[idx1]
		x2=x[idx2]
		edge_index1=edge_index
		edge_index2=edge_index
		
		x1 = self.conv1(x1, edge_index1)
		x1 = relu(x1)
		x1 = self.conv2(x1, edge_index1)
		x1 = relu(x1)
		x1 = self.conv3(x1, edge_index1)
		x1 = relu(x1)
		x1 = self.conv4(x1, edge_index1)
		x1 = relu(x1)
		x1 = self.conv5(x1, edge_index1)
		x1 = relu(x1)
		x1 = self.conv6(x1, edge_index1)
		x1 = relu(x1)
		x1 = self.conv7(x1, edge_index1)
		x1 = relu(x1)
		x1 = self.conv8(x1, edge_index1)

		x2 = self.conv11(x2, edge_index2)
		x2 = relu(x2)
		x2 = self.conv22(x2, edge_index2)
		x2 = relu(x2)
		x2 = self.conv33(x2, edge_index2)
		x2 = relu(x2)
		x2 = self.conv44(x2, edge_index2)
		x2 = relu(x2)
		x2 = self.conv55(x2, edge_index2)
		x2 = relu(x2)
		x2 = self.conv66(x2, edge_index2)
		x2 = relu(x2)
		x2 = self.conv77(x2, edge_index2)
		x2 = relu(x2)
		x2 = self.conv88(x2, edge_index2)

		uv=[]
		for i in range(n1):
			uv.append(paddle.concat([x1[i:i+1,0:],x2[i:i+1,0:]],axis=0))
		uv_=paddle.concat(uv,axis=0)
		#pdb.set_trace()
		return uv_#F.log_softmax(x, dim=1)	

class Ns_Chebnet(Layer):
	def __init__(self,split):
		super(Ns_Chebnet, self).__init__()
		nci=2;nco=1
		kk=10
		self.split=split
		self.conv1 = ChebConv(nci, 32,K=kk)
		self.conv2 = ChebConv(32, 64,K=kk)
		self.conv3 = ChebConv(64, 128,K=kk)
		self.conv4 = ChebConv(128, 256,K=kk)
		self.conv5 = ChebConv(256, 128,K=kk)
		self.conv6 = ChebConv(128, 64,K=kk)
		self.conv7 = ChebConv(64, 32,K=kk)
		self.conv8 = ChebConv(32, nco,K=kk)

		self.conv11 = ChebConv(nci, 32,K=kk)
		self.conv22 = ChebConv(32, 64,K=kk)
		self.conv33 = ChebConv(64, 128,K=kk)
		self.conv44 = ChebConv(128, 256,K=kk)
		self.conv55 = ChebConv(256, 128,K=kk)
		self.conv66 = ChebConv(128, 64,K=kk)
		self.conv77 = ChebConv(64, 32,K=kk)
		self.conv88 = ChebConv(32, nco,K=kk)

		self.conv111 = ChebConv(nci, 32,K=kk)
		self.conv222 = ChebConv(32, 64,K=kk)
		self.conv333 = ChebConv(64, 128,K=kk)
		self.conv444 = ChebConv(128, 256,K=kk)
		self.conv555 = ChebConv(256, 128,K=kk)
		self.conv666 = ChebConv(128, 64,K=kk)
		self.conv777 = ChebConv(64, 32,K=kk)
		self.conv888 = ChebConv(32, nco,K=kk)

		try:
			self.conv1=last_chance0(self.conv1)
			self.conv2=last_chance0(self.conv2)
			self.conv3=last_chance0(self.conv3)
			self.conv4=last_chance0(self.conv4)
			self.conv5=last_chance0(self.conv5)
			self.conv6=last_chance0(self.conv6)
			self.conv7=last_chance0(self.conv7)
			self.conv8=last_chance1(self.conv8)

			self.conv11=last_chance0(self.conv11)
			self.conv22=last_chance0(self.conv22)
			self.conv33=last_chance0(self.conv33)
			self.conv44=last_chance0(self.conv44)
			self.conv55=last_chance0(self.conv55)
			self.conv66=last_chance0(self.conv66)
			self.conv77=last_chance0(self.conv77)
			self.conv88=last_chance1(self.conv88)

			self.conv111=last_chance0(self.conv111)
			self.conv222=last_chance0(self.conv222)
			self.conv333=last_chance0(self.conv333)
			self.conv444=last_chance0(self.conv444)
			self.conv555=last_chance0(self.conv555)
			self.conv666=last_chance0(self.conv666)
			self.conv777=last_chance0(self.conv777)
			self.conv888=last_chance1(self.conv888)
		except:
			paddle.nn.init.orthogonal_(self.conv1.weight, paddle.nn.init.calculate_gain('relu'))
			paddle.nn.init.orthogonal_(self.conv2.weight, paddle.nn.init.calculate_gain('relu'))
			paddle.nn.init.orthogonal_(self.conv3.weight, paddle.nn.init.calculate_gain('relu'))
			paddle.nn.init.orthogonal_(self.conv4.weight, paddle.nn.init.calculate_gain('relu'))
			paddle.nn.init.orthogonal_(self.conv5.weight, paddle.nn.init.calculate_gain('relu'))
			paddle.nn.init.orthogonal_(self.conv6.weight, paddle.nn.init.calculate_gain('relu'))
			paddle.nn.init.orthogonal_(self.conv7.weight, paddle.nn.init.calculate_gain('relu'))
			paddle.nn.init.orthogonal_(self.conv8.weight)

			paddle.nn.init.orthogonal_(self.conv11.weight, paddle.nn.init.calculate_gain('relu'))
			paddle.nn.init.orthogonal_(self.conv22.weight, paddle.nn.init.calculate_gain('relu'))
			paddle.nn.init.orthogonal_(self.conv33.weight, paddle.nn.init.calculate_gain('relu'))
			paddle.nn.init.orthogonal_(self.conv44.weight, paddle.nn.init.calculate_gain('relu'))
			paddle.nn.init.orthogonal_(self.conv55.weight, paddle.nn.init.calculate_gain('relu'))
			paddle.nn.init.orthogonal_(self.conv66.weight, paddle.nn.init.calculate_gain('relu'))
			paddle.nn.init.orthogonal_(self.conv77.weight, paddle.nn.init.calculate_gain('relu'))
			paddle.nn.init.orthogonal_(self.conv88.weight)

			paddle.nn.init.orthogonal_(self.conv111.weight, paddle.nn.init.calculate_gain('relu'))
			paddle.nn.init.orthogonal_(self.conv222.weight, paddle.nn.init.calculate_gain('relu'))
			paddle.nn.init.orthogonal_(self.conv333.weight, paddle.nn.init.calculate_gain('relu'))
			paddle.nn.init.orthogonal_(self.conv444.weight, paddle.nn.init.calculate_gain('relu'))
			paddle.nn.init.orthogonal_(self.conv555.weight, paddle.nn.init.calculate_gain('relu'))
			paddle.nn.init.orthogonal_(self.conv666.weight, paddle.nn.init.calculate_gain('relu'))
			paddle.nn.init.orthogonal_(self.conv777.weight, paddle.nn.init.calculate_gain('relu'))
			paddle.nn.init.orthogonal_(self.conv888.weight)

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
		
		x1 = self.conv1(x1, edge_index1)
		x1 = relu(x1)
		x1 = self.conv2(x1, edge_index1)
		x1 = relu(x1)
		x1 = self.conv3(x1, edge_index1)
		x1 = relu(x1)
		x1 = self.conv4(x1, edge_index1)
		x1 = relu(x1)
		x1 = self.conv5(x1, edge_index1)
		x1 = relu(x1)
		x1 = self.conv6(x1, edge_index1)
		x1 = relu(x1)
		x1 = self.conv7(x1, edge_index1)
		x1 = relu(x1)
		x1 = self.conv8(x1, edge_index1)

		x2 = self.conv11(x2, edge_index2)
		x2 = relu(x2)
		x2 = self.conv22(x2, edge_index2)
		x2 = relu(x2)
		x2 = self.conv33(x2, edge_index2)
		x2 = relu(x2)
		x2 = self.conv44(x2, edge_index2)
		x2 = relu(x2)
		x2 = self.conv55(x2, edge_index2)
		x2 = relu(x2)
		x2 = self.conv66(x2, edge_index2)
		x2 = relu(x2)
		x2 = self.conv77(x2, edge_index2)
		x2 = relu(x2)
		x2 = self.conv88(x2, edge_index2)

		x3 = self.conv111(x3, edge_index3)
		x3 = relu(x3)
		x3 = self.conv222(x3, edge_index3)
		x3 = relu(x3)
		x3 = self.conv333(x3, edge_index3)
		x3 = relu(x3)
		x3 = self.conv444(x3, edge_index3)
		x3 = relu(x3)
		x3 = self.conv555(x3, edge_index3)
		x3 = relu(x3)
		x3 = self.conv666(x3, edge_index3)
		x3 = relu(x3)
		x3 = self.conv777(x3, edge_index3)
		x3 = relu(x3)
		x3 = self.conv888(x3, edge_index3)

		uv=[]
		for i in range(n1):
			uv.append(paddle.concat([x1[i:i+1,0:],x2[i:i+1,0:]],axis=0))
		uv_=paddle.concat(uv,axis=0)
		return paddle.concat([uv_,x3],axis=0)#F.log_softmax(x, dim=1)