import numpy as np
import pdb
import matplotlib.pyplot as plt
import sys
from scipy.io import loadmat

import paddle
sys.path.insert(0, 'pycamotk')
from pyCaMOtk.create_mesh_hcube import mesh_hcube
from pyCaMOtk.mesh import Mesh, get_gdof_from_bndtag
from pyCaMOtk.LinearElasticityHandCode import *
from pyCaMOtk.create_dbc_strct import create_dbc_strct
from pyCaMOtk.create_femsp_cg import create_femsp_cg
from pyCaMOtk.solve_fem import solve_fem
from pyCaMOtk.visualize_fem import visualize_fem
from pyCaMOtk.geom_mltdim import Simplex, Hypercube
from pyCaMOtk.create_fem_resjac  import create_fem_resjac

sys.path.insert(0, '../source')
import TensorFEMCore
from GCNNModel import e2vcg2connectivity,LinearElasticityNet2D
from TensorFEMCore import Double,solve_fem_GCNN, ReshapeFix
import setup_prob_eqn_handcode

sys.path.insert(0, 'utils')
from utils import Data
def train():
	ii=0
	Graph=[]
	Ue=Double(Ufem.flatten().reshape(ndof,1))
	fcn_id=Double(np.asarray([ii]))
	Ue_aug=paddle.concat((fcn_id,Ue),axis=0)
	xcg_gcnn=np.zeros((2,2*xcg.shape[1]))
	for i in range(xcg.shape[1]):
		xcg_gcnn[:,2*i]=xcg[:,i]
		xcg_gcnn[:,2*i+1]=xcg[:,i]
	Uin=Double(xcg_gcnn.T)
	graph=Data(x=Uin,y=Ue_aug,edge_index=connectivity)
	Graph.append(graph)
	DataList=[[Graph[0]]]
	TrainDataloader=DataList
	# device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model=LinearElasticityNet2D()
	[model,info]=solve_fem_GCNN(TrainDataloader,LossF,model,tol,maxit)
	np.save('modelCircleDet.npy',info)
	solution=model(Graph[0].to('cuda'))
	solution=ReshapeFix(paddle.clone(solution),[len(solution.flatten()),1],'C')
	solution[dbc.dbc_idx]=Double(dbc.dbc_val.reshape([len(dbc.dbc_val),1]))
	solution=solution.detach().cpu().numpy()
	xcg_defGCNN=xcg+np.reshape(solution,[ndim,nnode],order='F')
	msh_defGCNN=Mesh(etype,xcg_defGCNN,e2vcg,e2bnd,ndim)
	uabsGCNN=np.sqrt(solution[[i for i in range(ndof) if i%2==0]]**2+\
				solution[[i for i in range(ndof) if i%2!=0]]**2)
	return msh_defGCNN, uabsGCNN

def plot(msh_defGCNN, uabsGCNN):
	plt.figure()
	ax1=plt.subplot(1,1,1)
	_,cbar1=visualize_fem(ax1,msh_defGCNN,uabsGCNN[e2vcg],{"plot_elem":False,"nref":4},[])
	ax1.axis('off')
	cbar1.remove()
	plt.margins(0,0)
	plt.savefig('gcnn_2dlinearelasticity_square.png',
				bbox_inches='tight',pad_inches=0,dpi=800)

	plt.figure()
	ax2=plt.subplot(1,1,1)
	_,cbar2=visualize_fem(ax2,msh_def,uabs[e2vcg],{"plot_elem":False,"nref":4},[])
	ax2.axis('off')
	cbar2.remove()
	plt.margins(0,0)
	plt.savefig('fem_2dlinearelasticity_square.png',
				bbox_inches='tight',pad_inches=0,dpi=800)
if __name__=='__main__':

	############################################################
	# FEM
	nvar=2; etype='hcube'; lims=np.asarray([[0,1],[0,1]])
	nel=[2,2]; porder=2; nf=4 
	msh=mesh_hcube(etype,lims,nel,porder).getmsh()
	xcg=msh.xcg
	e2vcg=msh.e2vcg
	e2bnd=msh.e2bnd
	porder=msh.porder
	[ndim,nnode]=xcg.shape
	nvar=ndim
	ndof=nnode*nvar

	lam=lambda x,el:1
	mu=lambda x,el:1
	f=lambda x,el:np.zeros([ndim,1])
	bnd2nbc=np.asarray([0,1,2,3])
	tb=lambda x,n,bnd,el,fc:np.asarray([[0.5],[0]])*((bnd-2)**2<1e-8)+np.asarray([[0],[0]])
	prob=setup_linelast_base_handcode(ndim,lam,mu,f,tb,bnd2nbc)
	# Create finite element space
	femsp=create_femsp_cg(prob,msh,porder,e2vcg,porder,e2vcg)
	ldof2gdof=femsp.ldof2gdof_var.ldof2gdof
	geo=Hypercube(ndim,porder)
	f2v=geo.f2n
	dbc_idx=get_gdof_from_bndtag([i for i in range(ndim)],[0],nvar,ldof2gdof,e2bnd,f2v)
	dbc_idx.sort()
	dbc_idx=np.asarray(dbc_idx)
	dbc_val=0*dbc_idx
	dbc=create_dbc_strct(ndof,dbc_idx,dbc_val)
	femsp.dbc=dbc
	tol = 1.0e-8; maxit=4500

	[Ufem,info]=solve_fem('cg',msh.transfdatacontiguous,femsp.elem,femsp.elem_data, 
					femsp.ldof2gdof_eqn.ldof2gdof,
					femsp.ldof2gdof_var.ldof2gdof,msh.e2e,
					femsp.spmat,dbc,None,tol,maxit)

	xcg_def=xcg+np.reshape(Ufem,[ndim,nnode],order='F')
	msh_def=Mesh(etype,xcg_def,e2vcg,e2bnd,ndim)
	uabs=np.sqrt(Ufem[[i for i in range(ndof) if i%2==0]]**2+\
				Ufem[[i for i in range(ndof) if i%2!=0]]**2)
	#GCNN
	connectivity=e2vcg2connectivity(e2vcg,'ele')
	prob=setup_prob_eqn_handcode.setup_linelast_base_handcode\
		(ndim,lam,mu,f,tb,bnd2nbc)
	femsp_gcnn=create_femsp_cg(prob,msh,porder,e2vcg,porder,e2vcg,dbc)
	LossF=[]
	fcn=lambda u_:TensorFEMCore.create_fem_resjac('cg',
								u_,msh.transfdatacontiguous,
								femsp_gcnn.elem,femsp_gcnn.elem_data, 
								femsp_gcnn.ldof2gdof_eqn.ldof2gdof,
								femsp_gcnn.ldof2gdof_var.ldof2gdof,
								msh.e2e,femsp_gcnn.spmat,dbc)
	fcn_fem=lambda u_:create_fem_resjac('cg',
					u_,msh.transfdatacontiguous,
					femsp.elem,femsp.elem_data, 
					femsp.ldof2gdof_eqn.ldof2gdof,
					femsp.ldof2gdof_var.ldof2gdof,
					msh.e2e,femsp.spmat,dbc)
	LossF.append(fcn)
	msh_defGCNN, uabsGCNN = train()
	plot(msh_defGCNN, uabsGCNN)

