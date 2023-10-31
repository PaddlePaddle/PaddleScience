import paddle
import numpy as np
import pdb

from TensorFEMCore import Double, ReshapeFix

"""
####Possion Equation
"""
class setup_linelptc_sclr_base_handcode(object):
	"""docstring for setup_linelptc_sclr_base_handcode"""
	def __init__(self,ndim,K,f,Qb,bnd2nbc):
		self.ndim=ndim
		self.K=K
		self.f=f
		self.Qb=Qb
		self.bnd2nbc=bnd2nbc

		self.I=np.eye(self.ndim)
		if self.K==None:
			self.K=lambda x,el: self.I.reshape(self.ndim**2,1,order='F') #Fortan like
		if self.f==None:
			self.f=lambda x,el: 0
		if self.Qb==None:
			self.Qb=lambda x,n,bnd,el,fc: 0

		self.eqn=LinearEllipticScalarBaseHandcode()
		self.vol_pars_fcn=lambda x,el:np.vstack((self.K(x, el),self.f(x, el),np.nan))
		self.bnd_pars_fcn=lambda x,n,bnd,el,fc:np.vstack((self.K(x,el),
														  self.f(x,el),
														  self.Qb(x,n,bnd,el,fc)))

														 

class LinearEllipticScalarBaseHandcode(object):
	"""docstring for LinearEllipticScalarBaseHandcode"""
	def __init__(self):
		self.neqn=1
		self.nvar=1
		self.ncomp=1

	def srcflux(self,UQ,pars,x,model=None):
		"""
		eval_linelptc_base_handcode_srcflux
		"""
		# Extract information from input
		q=UQ[0,1:]
		q=ReshapeFix(q,[len(q),1])
		self.ndim=len(q)
		try:
			k=np.reshape(pars[0:self.ndim**2],
					 (self.ndim,self.ndim),order='F')
		except:
			k=paddle.reshape(pars[0:self.ndim**2],
					 (self.ndim,self.ndim))
		f=pars[self.ndim**2]
		try:
			temp_flag=(f.requires_grad)
			f=f.reshape([1,1])
		except:
			f=f.reshape([1,1])

		k_ml=paddle.to_tensor(k, dtype='float32')
		# Define flux and source
		SF=paddle.concat((f,-1*paddle.mm(k_ml,q)),axis=0)
		

		# Define partial derivative
		dSFdU=np.zeros([self.neqn, self.ndim+1, self.ncomp,self.ndim+1])
		try:
			dSFdU[:,1:,:,1:]=np.reshape(-1*k,[self.neqn, self.ndim,self.ncomp,self.ndim])
		except:
			k=k.detach().cpu().numpy()
			dSFdU[:,1:,:,1:]=np.reshape(-1*k,[self.neqn, self.ndim,self.ncomp,self.ndim])
		dSFdU=paddle.to_tensor(dSFdU, dtype='float32')
		return SF, dSFdU

	def bndstvcflux(self,nbcnbr,UQ,pars,x,n):
		nvar=UQ.shape[0]
		ndim=UQ.shape[1]-1

		Ub=UQ[:,0]
		dUb=np.zeros([nvar,nvar,self.ndim+1])
		dUb[:,:,0]=np.eye(nvar)

		Fn=pars[ndim**2+1]
		dFn=np.zeros([nvar,nvar,self.ndim+1])
		dUb=Double(dUb)
		Fn=Double(Fn)
		dFn=Double(dFn)
		return Ub,dUb,Fn,dFn

"""
####Linear Elasticity Equation
"""

class setup_linelast_base_handcode(object):
	"""docstring for setup_linelast_base_handcode"""
	def __init__(self,ndim,lam,mu,f,tb,bnd2nbc):
		self.bnd2nbc=bnd2nbc
		self.eqn=LinearElasticityBaseHandcode(ndim)
		self.vol_pars_fcn=lambda x, el: np.vstack((lam(x,el),
			                                      mu(x,el),
			                                      f(x,el),
			                                      np.zeros([ndim,1])+np.nan))
		self.bnd_pars_fcn=lambda x,n,bnd,el,fc:np.vstack((lam(x, el),
			                                              mu(x, el),
			                                              f(x, el),
			                                              tb(x, n, bnd, el, fc)))

		

class LinearElasticityBaseHandcode(object):
	"""docstring for LinearElasticityBaseHandcode"""
	def __init__(self,ndim):
		self.neqn=ndim
		self.nvar=ndim
		self.bndstvcflux=\
		lambda nbcnbr, UQ, pars, x, n:\
		eval_linelast_base_handcode_bndstvc_intr_bndflux_pars(UQ, pars, x, n)
		self.srcflux=lambda UQ,pars,x:\
		eval_linelast_base_handcode_srcflux(UQ, pars, x)

def eval_linelast_base_handcode_srcflux(UQ, pars, x):
	q=UQ[:,1:]
	ndim=q.shape[0]
	# Define information regarding size of the system
	neqn=ndim
	ncomp=ndim

	# Extract parameters
	lam=pars[0]
	mu=pars[1]
	f=pars[2:2+ndim]
	F=-lam*paddle.trace(q)*(Double(np.eye(ndim)))-mu*(q+q.T)
	try:
		S=Double(f.reshape([ndim,1],order='F'))
	except:
		S=f.reshape([ndim,1])
	SF=paddle.concat((S,F),axis=1)
	dSFdU=Double(np.zeros([neqn,ndim+1,ncomp,ndim+1]))
	for i in range(ndim):
		for j in range(ndim):
			dSFdU[i,1+i,j,1+j]=dSFdU[i,1+i,j,1+j]-lam
			dSFdU[i,1+j,i,1+j]=dSFdU[i,1+j,i,1+j]-mu
			dSFdU[i,1+j,j,1+i]=dSFdU[i,1+j,j,1+i]-mu
	return SF, dSFdU

def eval_linelast_base_handcode_bndstvc_intr_bndflux_pars(UQ,pars,x,n):
	nvar=UQ.shape[0]
	ndim=UQ.shape[1]-1

	Ub=UQ[:,0]
	dUb=np.zeros([nvar,nvar,ndim+1])
	dUb[:,:,0]=np.eye(nvar)
	Fn=-pars[-ndim:]
	dFn=np.zeros([nvar,nvar,ndim+1])
	dUb=Double(dUb)
	Fn=ReshapeFix(Double(Fn),[len(Fn),1],order='F')
	dFn=Double(dFn)
	return Ub,dUb,Fn,dFn

"""
#### Inconpressible Navier Stokes Equation
"""
class setup_ins_base_handcode(object):
	"""docstring for setup_ins_base_handcode"""
	def __init__(self,ndim,rho,nu,tb,bnd2nbc):
		self.eqn=IncompressibleNavierStokes(ndim)
		self.bnd2nbc=bnd2nbc
		self.vol_pars_fcn=lambda x,el:np.vstack([rho(x, el),
			                                     nu(x, el),
			                                     np.zeros([ndim+1,1])+np.nan])
		self.bnd_pars_fcn=lambda x,n,bnd,el,fc:np.vstack([rho(x,el),
			 										      nu(x,el),
			 										      tb(x,n,bnd,el,fc)])

class IncompressibleNavierStokes(object):
	"""docstring for IncompressibleNavierStokes"""
	def __init__(self,ndim):
		self.ndim=ndim
		self.nvar=ndim+1
		self.srcflux=lambda UQ,pars,x:\
		             eval_ins_base_handcode_srcflux(UQ,pars,x)
		self.bndstvcflux=lambda nbcnbr,UQ,pars,x,n:\
					     eval_ins_base_handcode_bndstvc_intr_bndflux_pars(UQ,pars,x,n)

def eval_ins_base_handcode_srcflux(UQ,pars,x):
	u=UQ[:,0]; q=UQ[:,1:]
	ndim=u.shape[0]-1
	neqn=ndim+1
	ncomp=ndim+1
	rho=pars[0]
	nu=pars[1]
	v=u[0:ndim]

	v=ReshapeFix(v,[len(v),1],'F')
	
	p=u[-1]
	dv=q[0:ndim,:]
	S=paddle.concat([-rho*paddle.mm(dv,v),-paddle.trace(dv).reshape([1,1])],axis=0)
	
	F=paddle.concat([-rho*nu*dv+p*paddle.eye(ndim, dtype='float32'),
	             paddle.zeros([1,ndim], dtype='float32')],axis=0)
	
	
	SF=paddle.concat([S,F],axis=1)

	dSFdUQ=np.zeros([neqn,ndim+1,ncomp,ndim+1])
	dSFdUQ[:,0,:,0]=np.vstack([np.hstack([-rho*dv.detach().cpu().numpy(),np.zeros([ndim,1])]), np.zeros([1,ndim+1])])
	for i in range(ndim):
		dSFdUQ[i,0,i,1:]=-rho*v.detach().cpu().numpy().reshape(dSFdUQ[i,0,i,1:].shape,order='F')
	dSFdUQ[-1,0,0:-1,1:]=np.reshape(-np.eye(ndim),[1,ndim,ndim],order='F')
	dSFdUQ[0:-1,1:,-1,0]=np.eye(ndim)
	for i in range(ndim):
		for j in range(ndim):
			dSFdUQ[i,1+j,i,1+j]=dSFdUQ[i,1+j,i,1+j]-rho*nu
	dSFdUQ=Double(dSFdUQ)
	return SF,dSFdUQ

def eval_ins_base_handcode_bndstvc_intr_bndflux_pars(UQ,pars,x,n):
	nvar=UQ.shape[0]
	ndim=UQ.shape[1]-1
	Ub=UQ[:,0]
	dUb=np.zeros([nvar,nvar,ndim+1])
	dUb[:,:,0]=np.eye(nvar)
	Fn=-pars[-ndim-1:].reshape([-1,1])
	dFn=np.zeros([nvar,nvar,ndim+1])
	return Ub,Double(dUb),Double(Fn),Double(dFn)
