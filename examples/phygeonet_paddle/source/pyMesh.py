errMessageJoint='The geometry is not closed!'
errMessageParallel='The parallel sides do not have the same number of node!'
errMessageXYShape='The x y shapes do not have match each other!'
errMessageDomainType='domainType can only be physical domain or reference domain!'
arrow='====>'
clow='green'
cup='blue'
cright='red'
cleft='orange'
cinternal='black'
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import matplotlib.pyplot as plt
import paddle

def np2cuda(myList):
	MyList=[]
	for item in myList:
		MyList.append(item.to('cuda'))
	return MyList

def to4DTensor(myList):
	MyList=[]
	for item in myList:
		if len(item.shape)==3:
			item=paddle.to_tensor(item.reshape([item.shape[0],1,item.shape[1],
				              item.shape[2]]), dtype="float32")
			MyList.append(item)
		else:
			item=paddle.to_tensor(item, dtype="float32")
			MyList.append(item)
	return MyList
def checkGeo(leftX,leftY,rightX,rightY,lowX,lowY,upX,upY,tolJoint):
	print(arrow+'Check bc nodes!')
	assert len(leftX.shape)==len(leftY.shape)==len(rightX.shape)==\
	       len(rightY.shape)==len(lowX.shape)==len(lowY.shape)==\
	       len(upX.shape)==len(upY.shape)==1,\
	       'all left(right)X(Y) must be 1d vector!'
	assert np.abs(leftX[0]-lowX[0])<tolJoint,errMessageJoint
	assert np.abs(leftX[-1]-upX[0])<tolJoint,errMessageJoint
	assert np.abs(rightX[0]-lowX[-1])<tolJoint,errMessageJoint
	assert np.abs(rightX[-1]-upX[-1])<tolJoint,errMessageJoint
	assert np.abs(leftY[0]-lowY[0])<tolJoint,errMessageJoint
	assert np.abs(leftY[-1]-upY[0])<tolJoint,errMessageJoint
	assert np.abs(rightY[0]-lowY[-1])<tolJoint,errMessageJoint
	assert np.abs(rightY[-1]-upY[-1])<tolJoint,errMessageJoint
	assert leftX.shape==leftY.shape==rightX.shape==rightY.shape,\
	       errMessageParallel
	assert upX.shape==upY.shape==lowX.shape==lowY.shape,\
	       errMessageParallel
	print(arrow+'BC nodes pass!')
def plotBC(ax,x,y):
	ax.plot(x[:,0],y[:,0],'-o',color=cleft)    # left BC
	ax.plot(x[:,-1],y[:,-1],'-o',color=cright) # right BC
	ax.plot(x[0,:],y[0,:],'-o',color=clow)    	# low BC
	ax.plot(x[-1,:],y[-1,:],'-o',color=cup)  	# up BC
	return ax

def plotMesh(ax,x,y,width=0.05):
	[ny,nx]=x.shape
	for j in range(0,nx):
		ax.plot(x[:,j],y[:,j],color=cinternal,linewidth=width)
	for i in range(0,ny):
		ax.plot(x[i,:],y[i,:],color=cinternal,linewidth=width)
	return ax
def setAxisLabel(ax,type):
	if type=='p':
		ax.set_xlabel(r'$x$')
		ax.set_ylabel(r'$y$')
	elif type=='r':
		ax.set_xlabel(r'$\xi$')
		ax.set_ylabel(r'$\eta$')
	else:
		raise ValueError('The axis type only can be reference or physical')

def ellipticMap(x,y,h,tol):
	eps=2.2e-16
	assert x.shape==y.shape,errMessageXYShape
	[ny,nx]=x.shape
	ite=1
	A=np.ones([ny-2,nx-2]); B=A; C=A;
	Err=[]
	while True:
		X=(A*(x[2:,1:-1]+x[0:-2,1:-1])+C*(x[1:-1,2:]+x[1:-1,0:-2])-\
		  B/2*(x[2:,2:]+x[0:-2,0:-2]-x[2:,0:-2]-x[0:-2,2:]))/2/(A+C)
		Y=(A*(y[2:,1:-1]+y[0:-2,1:-1])+C*(y[1:-1,2:]+y[1:-1,0:-2])-\
		  B/2*(y[2:,2:]+y[0:-2,0:-2]-y[2:,0:-2]-y[0:-2,2:]))/2/(A+C)
		err=np.max(np.max(np.abs(x[1:-1,1:-1]-X)))+\
			np.max(np.max(np.abs(y[1:-1,1:-1]-Y)))
		#print('error at this iteration'+'===>'+str(err))
		Err.append(err)
		x[1:-1,1:-1]=X; y[1:-1,1:-1]=Y;
		A=((x[1:-1,2:]-x[1:-1,0:-2])/2/h)**2+\
		  ((y[1:-1,2:]-y[1:-1,0:-2])/2/h)**2+eps
		B=(x[2:,1:-1]-x[0:-2,1:-1])/2/h*\
		  (x[1:-1,2:]-x[1:-1,0:-2])/2/h+\
		  (y[2:,1:-1]-y[0:-2,1:-1])/2/h*\
		  (y[1:-1,2:]-y[1:-1,0:-2])/2/h+eps
		C=((x[2:,1:-1]-x[0:-2,1:-1])/2/h)**2+\
		  ((y[2:,1:-1]-y[0:-2,1:-1])/2/h)**2+eps
		if err<tol:
			print('The mesh generation reaches covergence!')
			break; pass
		if ite>50000:
			print('The mesh generation not reaches covergence '+\
				  'within 50000 iterations! The current resdiual is ')
			print(err)
			break; pass
		ite=ite+1
	return x, y

def gen_e2vcg(x):
	nelem=(x.shape[0]-1)*(x.shape[1]-1)
	nelemx=x.shape[1]-1; nelemy=x.shape[0]-1; nelem=nelemx*nelemy
	nnx=x.shape[1];nny=x.shape[0]
	e2vcg0=np.zeros([4,nelem])
	e2vcg=np.zeros([4,nelem])
	for j in range(nelemy):
		for i in range(nelemx):
			e2vcg[:,j*nelemx+i]=np.asarray([j*nnx+i,j*nnx+i+1,
				                   (j+1)*nnx+i,(j+1)*nnx+i+1])
	return e2vcg.astype('int')




def visualize2D(ax,x,y,u,colorbarPosition='vertical',colorlimit=None):
	xdg0=np.vstack([x.flatten(order='C'),y.flatten(order='C')])
	udg0=u.flatten(order='C')
	idx=np.asarray([0,1,3,2])
	nelem=(x.shape[0]-1)*(x.shape[1]-1)
	nelemx=x.shape[1]-1; nelemy=x.shape[0]-1; nelem=nelemx*nelemy
	nnx=x.shape[1];nny=x.shape[0]
	e2vcg0=gen_e2vcg(x)
	udg_ref=udg0[e2vcg0]
	cmap=matplotlib.cm.coolwarm
	polygon_list=[]
	for i in range(nelem):
		polygon_=Polygon(xdg0[:,e2vcg0[idx,i]].T)
		polygon_list.append(polygon_)
	polygon_ensemble=PatchCollection(polygon_list,cmap=cmap,alpha=1)
	polygon_ensemble.set_edgecolor('face')
	polygon_ensemble.set_array(np.mean(udg_ref,axis=0))
	if colorlimit is None:
		pass
	else:
		polygon_ensemble.set_clim(colorlimit)
	ax.add_collection(polygon_ensemble)
	ax.set_xlim(np.min(xdg0[0,:]),np.max(xdg0[0,:]))
	#ax.set_xticks([np.min(xdg0[0,:]),np.max(xdg0[0,:])])
	ax.set_ylim(np.min(xdg0[1,:]),np.max(xdg0[1,:]))
	#ax.set_yticks([np.min(xdg0[1,:]),np.max(xdg0[1,:])])
	#ax.set_aspect('equal')
	cbar=plt.colorbar(polygon_ensemble,orientation=colorbarPosition)
	return ax,cbar


class hcubeMesh(object):
	"""docstring for hcubeMesh"""
	def __init__(self,leftX,leftY,rightX,rightY,lowX,lowY,upX,upY,
		         h,plotFlag=False,saveFlag=False,saveDir='./mesh.pdf',tolMesh=1e-8,tolJoint=1e-6):
		self.h=h
		self.tolMesh=tolMesh
		self.tolJoint=tolJoint
		self.plotFlag=plotFlag
		self.saveFlag=saveFlag
		checkGeo(leftX,leftY,rightX,rightY,lowX,lowY,upX,upY,tolJoint)
		# Extract discretization info
		self.ny=leftX.shape[0]; self.nx=upX.shape[0]
		# Prellocate the physical domain
		#Left->Right->Low->Up
		self.x=np.zeros([self.ny,self.nx])
		self.y=np.zeros([self.ny,self.nx])
		self.x[:,0]=leftX; self.y[:,0]=leftY
		self.x[:,-1]=rightX; self.y[:,-1]=rightY
		self.x[0,:]=lowX; self.y[0,:]=lowY
		self.x[-1,:]=upX; self.y[-1,:]=upY
		self.x,self.y=ellipticMap(self.x,self.y,self.h,self.tolMesh)
		# Define the ref domain
		eta,xi=np.meshgrid(np.linspace(0,self.ny-1,self.ny),
	               np.linspace(0,self.nx-1,self.nx),
	               sparse=False,indexing='ij')
		self.xi=xi*h; self.eta=eta*h;
		fig=plt.figure()
		ax=plt.subplot(1,2,1)
		plotBC(ax,self.x,self.y)
		plotMesh(ax,self.x,self.y)
		setAxisLabel(ax,'p')
		ax.set_aspect('equal')
		ax.set_title('Physics Domain Mesh')
		#ax.tick_params(axis='both',which='both',bottom=False,left=False,top=False,labelbottom=False,labelleft=False)
		#ax.axis('off')
		ax=plt.subplot(1,2,2)
		plotBC(ax,self.xi,self.eta)
		plotMesh(ax,self.xi,self.eta)
		setAxisLabel(ax,'r')
		ax.set_aspect('equal')
		ax.set_title('Reference Domain Mesh')
		#ax.tick_params(axis='both',which='both',bottom=False,left=False,top=False,labelbottom=False,labelleft=False)
		#ax.axis('off')
		fig.tight_layout(pad=1)
		if saveFlag:
			plt.savefig(saveDir,bbox_inches='tight')
		if plotFlag:
			plt.show()
		plt.close(fig)
		self.dxdxi=(self.x[1:-1,2:]-self.x[1:-1,0:-2])/2/self.h
		self.dydxi=(self.y[1:-1,2:]-self.y[1:-1,0:-2])/2/self.h
		self.dxdeta=(self.x[2:,1:-1]-self.x[0:-2,1:-1])/2/self.h
		self.dydeta=(self.y[2:,1:-1]-self.y[0:-2,1:-1])/2/self.h
		self.J=self.dxdxi*self.dydeta-self.dxdeta*self.dydxi
		self.Jinv=1/self.J
		
		dxdxi_ho_internal=(-self.x[:,4:]+8*self.x[:,3:-1]-\
			           8*self.x[:,1:-3]+self.x[:,0:-4])/12/self.h
		dydxi_ho_internal=(-self.y[:,4:]+8*self.y[:,3:-1]-\
			           8*self.y[:,1:-3]+self.y[:,0:-4])/12/self.h
		dxdeta_ho_internal=(-self.x[4:,:]+8*self.x[3:-1,:]-\
			            8*self.x[1:-3,:]+self.x[0:-4,:])/12/self.h
		dydeta_ho_internal=(-self.y[4:,:]+8*self.y[3:-1,:]-\
			            8*self.y[1:-3,:]+self.y[0:-4,:])/12/self.h

		dxdxi_ho_left=(-11*self.x[:,0:-3]+18*self.x[:,1:-2]-\
			           9*self.x[:,2:-1]+2*self.x[:,3:])/6/self.h
		dxdxi_ho_right=(11*self.x[:,3:]-18*self.x[:,2:-1]+\
			           9*self.x[:,1:-2]-2*self.x[:,0:-3])/6/self.h
		dydxi_ho_left=(-11*self.y[:,0:-3]+18*self.y[:,1:-2]-\
			           9*self.y[:,2:-1]+2*self.y[:,3:])/6/self.h
		dydxi_ho_right=(11*self.y[:,3:]-18*self.y[:,2:-1]+\
			           9*self.y[:,1:-2]-2*self.y[:,0:-3])/6/self.h
		

		dxdeta_ho_low=(-11*self.x[0:-3,:]+18*self.x[1:-2,:]-\
			           9*self.x[2:-1,:]+2*self.x[3:,:])/6/self.h
		dxdeta_ho_up=(11*self.x[3:,:]-18*self.x[2:-1,:]+\
			           9*self.x[1:-2,:]-2*self.x[0:-3,:])/6/self.h
		dydeta_ho_low=(-11*self.y[0:-3,:]+18*self.y[1:-2,:]-\
			           9*self.y[2:-1,:]+2*self.y[3:,:])/6/self.h
		dydeta_ho_up=(11*self.y[3:,:]-18*self.y[2:-1,:]+\
			           9*self.y[1:-2,:]-2*self.y[0:-3,:])/6/self.h

		self.dxdxi_ho=np.zeros(self.x.shape)
		self.dxdxi_ho[:,2:-2]=dxdxi_ho_internal
		self.dxdxi_ho[:,0:2]=dxdxi_ho_left[:,0:2]
		self.dxdxi_ho[:,-2:]=dxdxi_ho_right[:,-2:]

		self.dydxi_ho=np.zeros(self.y.shape)
		self.dydxi_ho[:,2:-2]=dydxi_ho_internal
		self.dydxi_ho[:,0:2]=dydxi_ho_left[:,0:2]
		self.dydxi_ho[:,-2:]=dydxi_ho_right[:,-2:]

		self.dxdeta_ho=np.zeros(self.x.shape)
		self.dxdeta_ho[2:-2,:]=dxdeta_ho_internal
		self.dxdeta_ho[0:2,:]=dxdeta_ho_low[0:2,:]
		self.dxdeta_ho[-2:,:]=dxdeta_ho_up[-2:,:]

		self.dydeta_ho=np.zeros(self.y.shape)
		self.dydeta_ho[2:-2,:]=dydeta_ho_internal
		self.dydeta_ho[0:2,:]=dydeta_ho_low[0:2,:]
		self.dydeta_ho[-2:,:]=dydeta_ho_up[-2:,:]

		self.J_ho=self.dxdxi_ho*self.dydeta_ho-\
		          self.dxdeta_ho*self.dydxi_ho
		self.Jinv_ho=1/self.J_ho
			






