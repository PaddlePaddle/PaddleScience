sys.path.insert(0, 'pycamotk')
from pyCaMOtk.create_mesh_hcube import mesh_hcube
from pyCaMOtk.setup_ins_base_handcode import \
     setup_ins_base_handcode
from pyCaMOtk.create_femsp_cg import create_femsp_cg_mixed2
from pyCaMOtk.create_dbc_strct import create_dbc_strct
from pyCaMOtk.solve_fem import solve_fem
from pyCaMOtk.visualize_fem import visualize_fem
from pyCaMOtk.mesh import Mesh

import numpy as np
import pdb
import sys
import matplotlib.pyplot as plt

import paddle

sys.path.insert(0, 'source')
import TensorFEMCore
from GCNNModel import e2vcg2connectivity,Ns_Chebnet
from TensorFEMCore import Double,solve_fem_GCNN, ReshapeFix
import setup_prob_eqn_handcode

sys.path.insert(0, 'utils')
from utils import Data
def train():
	'''
	Solve GCNN
	'''
	connectivity_uv=e2vcg2connectivity(e2vcg,'ele')
	connectivity_p=e2vcg2connectivity(e2vcg2,'ele')
	connectivity=paddle.concat([connectivity_uv,connectivity_uv,
							connectivity_p],axis=1)
	prob=setup_prob_eqn_handcode.setup_ins_base_handcode\
		(ndim,lambda x,el:rho,lambda x,el:nu,tb,bnd2nbc)

	femsp_gcnn=create_femsp_cg_mixed2(prob,msh,
								neqn1,nvar1,
								porder,porder,
								e2vcg,e2vcg,
								neqn2,nvar2,
								porder-1,porder-1,
								e2vcg2,e2vcg2)
	LossF=[]
	fcn=lambda u_:TensorFEMCore.create_fem_resjac('cg',
								u_,msh.transfdatacontiguous,
								femsp_gcnn.elem,femsp_gcnn.elem_data, 
								femsp_gcnn.ldof2gdof_eqn.ldof2gdof,
								femsp_gcnn.ldof2gdof_var.ldof2gdof,
								msh.e2e,femsp_gcnn.spmat,dbc)
	LossF.append(fcn)
	ii=0
	Graph=[]
	Ue=Double(U.flatten().reshape(-1,1))
	fcn_id=Double(np.asarray([ii]))
	Ue_aug=paddle.concat((fcn_id,Ue),axis=0)
	xcg_gcnn=np.zeros((2,2*xcg.shape[1]+msh_.xcg.shape[1]))
	for i in range(xcg.shape[1]):
		xcg_gcnn[:,2*i]=xcg[:,i]
		xcg_gcnn[:,2*i+1]=xcg[:,i]
	for i in range(msh_.xcg.shape[1]):
		xcg_gcnn[:,2*xcg.shape[1]+i]=msh_.xcg[:,i]
	Uin=Double(xcg_gcnn.T)
	graph=Data(x=Uin,y=Ue_aug,edge_index=connectivity)
	Graph.append(graph)
	DataList=[[Graph[0]]]
	TrainDataloader=DataList
	# device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	split=[xcg.shape[1],msh_.xcg.shape[1],connectivity_uv.shape[1]]
	model=Ns_Chebnet(split)
	[model,info]=solve_fem_GCNN(TrainDataloader,LossF,model,tol,maxit)
	paddle.save(model, './Model.pth')
	np.save('modelTrain.npy',info)
	solution=model(Graph[0])
	solution=ReshapeFix(paddle.clone(solution),[len(solution.flatten()),1],'C')
	solution[dbc.dbc_idx]=Double(dbc.dbc_val.reshape([len(dbc.dbc_val),1]))
	solution=solution.detach().numpy()

	uv_GCNN=np.reshape(solution[0:ndim*nnode],[ndim,nnode],order='F')
	uabs_GCNN=np.sqrt(uv_GCNN[0,:]**2+uv_GCNN[1,:]**2)
	pGCNN=solution[ndim*nnode:]

	return uabs_GCNN, pGCNN

def plot(uabs_GCNN, pGCNN):
	fig=plt.figure()
	ax1=plt.subplot(1,2,1)
	visualize_fem(ax1,msh,uabs[e2vcg],{"plot_elem":False,"nref":4},[])
	ax1.set_title('FEM-Mixed2 Velocity Magnitude')
	ax2=plt.subplot(1,2,2)
	visualize_fem(ax2,msh,uabs_GCNN[e2vcg],{"plot_elem":False,"nref":4},[])
	ax2.set_title('GCNN Velocity Magnitude')
	fig.tight_layout(pad=2)
	plt.savefig('StenosisNSU.pdf',bbox_inches='tight')

	fig=plt.figure()
	ax1=plt.subplot(1,2,1)
	visualize_fem(ax1,msh_,p[e2vcg2],{"plot_elem":False,"nref":4},[])
	ax1.set_title('FEM-Mixed2 Pressure')
	ax2=plt.subplot(1,2,2)
	visualize_fem(ax2,msh_,pGCNN[e2vcg2],{"plot_elem":False,"nref":4},[])
	ax2.set_title('GCNN Pressure Magnitude')
	fig.tight_layout(pad=2)
	plt.savefig('StenosisNSP.pdf',bbox_inches='tight')
if __name__=='__main__':
	# Basic setting of the case
	ReList=np.linspace(10,100,2)
	U0=None
	for Re in ReList:
		print('Re=',Re)
		rho=1
		nu=1/Re
		L=1
		etype='hcube'
		nelem=[10,10]
		porder=2
		pltit=True
		ndim=2
		nvar=2
		inletVelocity=1
		s=0.4

		# Create finite element mesh
		msh=mesh_hcube(etype,np.asarray([[0,L],[0,L]]),nelem,porder).getmsh()
		msh_=mesh_hcube(etype,np.asarray([[0,L],[0,L]]),nelem,porder-1).getmsh()
		e2vcg2=msh_.e2vcg
		xcg=msh.xcg
		e2vcg=msh.e2vcg
		nnode=xcg.shape[1]
		nnode_p=msh_.xcg.shape[1]

		# Setup equation parameters and natural boundary conditions
		tb=lambda x,n,bnd,el,fc:np.zeros([ndim+1,1])
		bnd2nbc=[1,1,1,1]
		prob=setup_ins_base_handcode(ndim,lambda x,el:rho,
										lambda x,el:nu,
										tb,bnd2nbc)

		# start to impose BC
		ndofU=ndim*nnode;
		ndofUP=ndofU+msh_.xcg.shape[1]
		dbc_idx1=[]
		for i in range(nnode):
			if i in dbc_idx1:
				continue
			if xcg[0,i]<1e-12 or xcg[0,i]>(L-1e-12):#xcg[0,i]<1e-12 or xcg[1,i]<1e-12 or xcg[0,i]>(L-1e-12):
				dbc_idx1.append(i)
		dbc_idx2=[i for i in range(nnode) if xcg[1,i]<1e-12 and i not in dbc_idx1]
		dbc_idx3=[i for i in range(nnode_p) if msh_.xcg[1,i]>L-1e-12 and i not in dbc_idx1 and i not in dbc_idx2]

		dbc_val1=[0 for i in dbc_idx1]
		dbc_val2=[0 for i in dbc_idx2]
		dbc_val3=[0 for i in dbc_idx3]

		dbc_idx=[2*i for i in dbc_idx1]
		dbc_val=[i for i in dbc_val1]
		for i in range(len(dbc_val1)):
			dbc_idx.append(2*dbc_idx1[i]+1)
			dbc_val.append(dbc_val1[i])

		for i in range(len(dbc_idx2)):
			dbc_idx.append(2*dbc_idx2[i])
			dbc_val.append(dbc_val2[i])
		for i in range(len(dbc_idx2)):
			dbc_idx.append(2*dbc_idx2[i]+1)
			dbc_val.append(inletVelocity)

		for i in range(len(dbc_idx3)):
			dbc_idx.append(ndofU+dbc_idx3[i])
			dbc_val.append(dbc_val3[i])

		dbc_idx,I=np.unique(np.asarray(dbc_idx),return_index=True)
		dbc_idx=[i for i in dbc_idx]
		dbc_val=np.asarray(dbc_val)
		dbc_val=dbc_val[I]
		dbc_val=[i for i in dbc_val]

		dbc_idx=np.asarray(dbc_idx)
		dbc_val=np.asarray(dbc_val)
		dbc=create_dbc_strct(ndofUP,dbc_idx,dbc_val)

		# ReDefine Mesh
		xcg_=msh_.xcg
		shrinkScalar=lambda y :(1-s*np.cos(np.pi*(y-L/2)))

		for i in range(xcg.shape[1]):
			xcg[0,i]=(xcg[0,i]-L/2)*shrinkScalar(xcg[1,i])+L/2
		for i in range(xcg_.shape[1]):
			xcg_[0,i]=(xcg_[0,i]-L/2)*shrinkScalar(xcg_[1,i])+L/2

		msh=Mesh(etype,xcg,e2vcg,msh.e2bnd,2)
		msh_=Mesh(etype,xcg_,e2vcg2,msh_.e2bnd,2)
		e2vcg2=msh_.e2vcg
		xcg=msh.xcg
		e2vcg=msh.e2vcg
		nnode=xcg.shape[1]

		# Create finite element space
		neqn1=ndim; neqn2=1
		nvar1=ndim; nvar2=1
		femsp=create_femsp_cg_mixed2(prob,msh,
									neqn1,nvar1,
									porder,porder,
									e2vcg,e2vcg,
									neqn2,nvar2,
									porder-1,porder-1,
									e2vcg2,e2vcg2)
		ldof2gdof = femsp.ldof2gdof_var.ldof2gdof
		femsp.dbc=dbc

		tol=1.0e-8
		maxit=10000
		[U,info]=solve_fem('cg',msh.transfdatacontiguous,
								femsp.elem,femsp.elem_data,
								femsp.ldof2gdof_eqn.ldof2gdof,
								femsp.ldof2gdof_var.ldof2gdof,
								msh.e2e,femsp.spmat,dbc,U0,
								tol,maxit)
		
		idx_free=[i for i in range(len(U)) if i not in dbc_idx]
		U0=U[idx_free].reshape([-1,1])
		uv=np.reshape(U[0:ndim*nnode],[ndim,nnode],order='F')
		p=U[ndim*nnode:]
		uabs=np.sqrt(uv[0,:]**2+uv[1,:]**2)
		if Re==ReList[-1]:
			fig=plt.figure()
			ax=plt.subplot(1,1,1)
			visualize_fem(ax,msh,uabs[e2vcg],{"plot_elem":False,"nref":4},[])
			ax.set_title('FEM-Mixed2 Velocity Magnitude')
			fig.tight_layout(pad=2)
			plt.savefig('FE-Mixed2V.pdf',bbox_inches='tight')

			fig=plt.figure()
			ax=plt.subplot(1,1,1)
			visualize_fem(ax,msh_,p[e2vcg2],{"plot_elem":False,"nref":4},[])
			ax.set_title('FEM-Mixed2 Pressure')
			fig.tight_layout(pad=2)
			plt.savefig('FE-Mixed2P.pdf',bbox_inches='tight')

	uabs_GCNN, pGCNN = train()
	plot(uabs_GCNN, pGCNN)
		

