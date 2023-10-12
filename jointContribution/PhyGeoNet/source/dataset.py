
import numpy as np

from paddle.io import Dataset

class VaryGeoDataset(Dataset):
	"""docstring for hcubeMeshDataset"""
	def __init__(self,MeshList):
		self.MeshList=MeshList
	def __len__(self):
		return len(self.MeshList)
	def __getitem__(self,idx):
		mesh=self.MeshList[idx]
		x=mesh.x
		y=mesh.y
		xi=mesh.xi
		eta=mesh.eta
		J=mesh.J_ho
		Jinv=mesh.Jinv_ho
		dxdxi=mesh.dxdxi_ho
		dydxi=mesh.dydxi_ho
		dxdeta=mesh.dxdeta_ho
		dydeta=mesh.dydeta_ho
		cord=np.zeros([2,x.shape[0],x.shape[1]])
		cord[0,:,:]=x; cord[1,:,:]=y
		InvariantInput=np.zeros([2,J.shape[0],J.shape[1]])
		InvariantInput[0,:,:]=J
		InvariantInput[1,:,:]=Jinv
		return [InvariantInput,cord,xi,eta,J,
		        Jinv,dxdxi,dydxi,
		        dxdeta,dydeta]


class FixGeoDataset(Dataset):
	"""docstring for hcubeMeshDataset"""
	def __init__(self,ParaList,mesh,OFSolutionList):
		self.ParaList=ParaList
		self.mesh=mesh
		self.OFSolutionList=OFSolutionList
	def __len__(self):
		return len(self.ParaList)
	def __getitem__(self,idx):
		mesh=self.mesh
		x=mesh.x
		y=mesh.y
		xi=mesh.xi
		eta=mesh.eta
		J=mesh.J_ho
		Jinv=mesh.Jinv_ho
		dxdxi=mesh.dxdxi_ho
		dydxi=mesh.dydxi_ho
		dxdeta=mesh.dxdeta_ho
		dydeta=mesh.dydeta_ho
		cord=np.zeros([2,x.shape[0],x.shape[1]])
		cord[0,:,:]=x; cord[1,:,:]=y
		ParaStart=np.ones(x.shape[0])*self.ParaList[idx]
		ParaEnd=np.zeros(x.shape[0])
		Para=np.linspace(ParaStart,ParaEnd,x.shape[1]).T
		return [Para,cord,xi,eta,J,
		        Jinv,dxdxi,dydxi,
		        dxdeta,dydeta,self.OFSolutionList[idx]]


class VaryGeoDataset_PairedSolution(Dataset):
	"""docstring for hcubeMeshDataset"""
	def __init__(self,MeshList,SolutionList):
		self.MeshList=MeshList
		self.SolutionList=SolutionList
	def __len__(self):
		return len(self.MeshList)
	def __getitem__(self,idx):
		mesh=self.MeshList[idx]
		x=mesh.x
		y=mesh.y
		xi=mesh.xi
		eta=mesh.eta
		J=mesh.J_ho
		Jinv=mesh.Jinv_ho
		dxdxi=mesh.dxdxi_ho
		dydxi=mesh.dydxi_ho
		dxdeta=mesh.dxdeta_ho
		dydeta=mesh.dydeta_ho
		cord=np.zeros([2,x.shape[0],x.shape[1]])
		cord[0,:,:]=x; cord[1,:,:]=y
		InvariantInput=np.zeros([2,J.shape[0],J.shape[1]])
		InvariantInput[0,:,:]=J
		InvariantInput[1,:,:]=Jinv
		return [InvariantInput,cord,xi,eta,J,
		        Jinv,dxdxi,dydxi,
		        dxdeta,dydeta,
				self.SolutionList[idx][:,:,0],
				self.SolutionList[idx][:,:,1],
				self.SolutionList[idx][:,:,2]]