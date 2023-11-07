import matplotlib.pyplot as plt
import numpy as np
import pdb
import sys

import paddle
paddle.seed(1334)
from random import sample 

sys.path.insert(0, 'pycamotk')
from pyCaMOtk.create_mesh_hsphere import mesh_hsphere 
from pyCaMOtk.create_dbc_strct import create_dbc_strct
from pyCaMOtk.create_femsp_cg import create_femsp_cg
from pyCaMOtk.visualize_fem import visualize_fem

sys.path.insert(0, 'source')
from FEM_ForwardModel import analyticalPossion
from GCNNModel import e2vcg2connectivity,PossionNet
from TensorFEMCore import Double,solve_fem_GCNN,create_fem_resjac
import setup_prob_eqn_handcode

sys.path.insert(0, 'utils')
from utils import Data

from circle import train

def train(S):
	# Define the Training Data
	Graph=[]
	ii=0
	for i in S:
		Ue=Double(analyticalPossion(xcg,i).flatten().reshape(ndof,1))
		fcn_id=Double(np.asarray([ii]))
		Ue_aug=paddle.concat((fcn_id,Ue),axis=0)
		Uin=Double(xcg.T)
		graph=Data(x=Uin,y=Ue_aug,edge_index=connectivity)
		Graph.append(graph)
		ii=ii+1
	DataList=[[Graph[i]] for i in range(len(S))]
	TrainDataloader=DataList
	# GCNN model
	device=paddle.device.set_device('gpu:0')
	model=PossionNet()

	# Training Data
	[model,info]=solve_fem_GCNN(TrainDataloader,LossF,model,tol,maxit)
	print('K=',K)
	print('Min Error=',info['Er'].min())
	print('Mean Error Last 10 iterations=',np.mean(info['Er'][-10:]))
	print('Var  Error Last 10 iterations=',np.var(info['Er'][-10:]))

	np.savetxt('demo0\ErFinal.txt', info['Er'])
	np.savetxt('demo0\Loss.txt', info['Loss'])

	solution=model(Graph[0])
	solution[dbc.dbc_idx]=Double(dbc.dbc_val.reshape([len(dbc.dbc_val),1]))
	solution=solution.detach().cpu().numpy()
	Ue=Ue.detach().cpu().numpy()
	return solution, Ue

def plot(solution, Ue):
	ax1=plt.subplot(1,1,1)
	_,cbar1=visualize_fem(ax1,msh,solution[e2vcg],{"plot_elem":True,"nref":6},[])
	ax1.tick_params(axis='both',which='both',bottom=False,left=False,top=False,labelbottom=False,labelleft=False)
	ax1.axis('off')
	cbar1.remove()
	plt.margins(0,0)
	plt.savefig('gcnn_possion_circle.png',bbox_inches='tight',pad_inches=-0.11,dpi=800)
	plt.close()

	ax2=plt.subplot(1,1,1)
	_,cbar2=visualize_fem(ax2,msh,Ue[e2vcg],{"plot_elem":True,"nref":6},[])
	ax2.tick_params(axis='both',which='both',bottom=False,left=False,top=False,labelbottom=False,labelleft=False)
	ax2.axis('off')
	cbar2.remove()
	plt.margins(0,0)
	plt.savefig('exact_possion_circle.png',bbox_inches='tight',pad_inches=-0.11,dpi=800)

if __name__=='__main__':
	"""
	Hyper prameters
	"""
	tol=1.0e-16
	maxit=3000

	# GCNN model
	model=PossionNet()									
	"""
	Set up GCNN-FEM Possion problem
	"""
	nin=1 # Number of input variable
	nvar=1 # Number of primanry variable
	etype='hcube' # Mesh type
	c=[0,0] # Domain center
	r=1 # Radius
	porder=2 # Polynomial order for solution and geometry basis
	nel=[2,2] # Number of element in x and y axis
	msh=mesh_hsphere(etype,c,r,nel,porder).getmsh() # Create mesh object
	xcg=msh.xcg # Extract node coordinates
	ndof=xcg.shape[1]
	e2vcg=msh.e2vcg # Extract element connectivity 
	connectivity=e2vcg2connectivity(msh.e2vcg,'ele')
	"""
	e2vcg is a 2D array (NNODE PER ELEM, NELEM): The connectivity of the
	mesh. The (:, e) entries are the global node numbers of the nodes
	that comprise element e. The local node numbers of each element are
	defined by the columns of this matrix, e.g., e2vcg(i, e) is the
	global node number of the ith local node of element e. 
	"""
	bnd2nbc=np.asarray([0]) # Define the boundary tag!
	K=lambda x,el: np.asarray([[1],[0],[0],[1]])
	"""
	# The flux constant Flux=[du/dx, du/dy]^T=K dot [dphi/dx,dphi/dy]
	where phi is the solution polynomial function
	""" 
	Qb=lambda x,n,bnd,el,fc: 0 # The primary variable value on the boundary
	dbc_idx=[i for i in range(xcg.shape[1]) if np.sum(xcg[:,i]**2)>1-1e-12] # The boundary node id
	dbc_idx=np.asarray(dbc_idx) 
	dbc_val=dbc_idx*0 # The boundary node primary variable value
	Ufem=analyticalPossion(xcg,2).flatten().reshape(ndof,1)

	idx_whole=[i for i in range(ndof) if i not in dbc_idx]
	obsidx=np.asarray([8])
	obsxcg=msh.xcg[:,obsidx]

	dbc_idx_new=np.hstack((np.asarray(dbc_idx),obsidx))
	dbc_val_new=Ufem[dbc_idx_new]
	dbc=create_dbc_strct(xcg.shape[1]*nvar,dbc_idx_new,dbc_val_new) # Create the class of boundary condition

	Src_new=model.source
	K_new=paddle.to_tensor([[1],[0],[0],[1]], dtype='float32').reshape((4,))
	parsfuncI=lambda x: paddle.concat((K_new,Src_new),axis=0)
	S=[2] # Parametrize the source value in the pde -F_ij,j=S_i
	LossF=[]
	for i in S: 
		f=lambda x,el: i
		prob=setup_prob_eqn_handcode.setup_linelptc_sclr_base_handcode(2,K,f,Qb,bnd2nbc)
		femsp=create_femsp_cg(prob,msh,porder,e2vcg,porder,e2vcg,dbc)
		fcn=lambda u_:create_fem_resjac('cg',u_,msh.transfdatacontiguous,
										femsp.elem,femsp.elem_data, 
										femsp.ldof2gdof_eqn.ldof2gdof,
										femsp.ldof2gdof_var.ldof2gdof,
										msh.e2e,femsp.spmat,dbc,[i for i in range(ndof) if i not in dbc_idx],parsfuncI,None,model)
		LossF.append(fcn)

	solution, Ue = train(S)
	plot(solution, Ue)