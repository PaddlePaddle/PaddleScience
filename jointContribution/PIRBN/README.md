# Physics-informed radial basis network (PIRBN)

This repository provides numerical examples of the **physics-informed radial basis network** (**PIRBN**).

Physics-informed neural network (PINN) has recently gained increasing interest in computational  mechanics.

This work starts from studying the training dynamics of PINNs via the nerual tangent kernel (NTK) theory. Based on numerical experiments, we found:

- PINNs tend to be a **local approximator** during the training
- For PINNs who fail to be a local apprixmator, the physics-informed loss can be hardly minimised through training

Inspired by findings, we proposed the PIRBN, which can exhibit the local property intuitively. It has been demonstrated that the NTK theory is applicable for PIRBN. Besides, other PINN techniques can be directly migrated to PIRBNs.

Numerical examples include:

 - 1D sine funtion (**Eq. 15** in the manuscript)

      **PDE**: $\frac{\partial^2 }{\partial x^2}u(x-100)-4\mu^2\pi^2 sin(2\mu\pi(x-100))=0, x\in[100,101]$

      **BC**:  $u(100)=u(101)=0.$

 - 1D sine function coupling problem (**Eq. 30** in the manuscript)

      **PDE**: $\frac{\partial^2 }{\partial x^2}u(x)=f(x), x\in[20,22]$

      **BC**:  $u(20)=u(22)=0.$

 - 1D nonlinear spring equation (**Eq. 31** in the manuscript)

      **PDE**: $\frac{\partial^2 }{\partial x^2}u(x)+4u(x)+sin[u(x)]=f(x), x\in[0,100]$

      **BC**:  $u(0)=\frac{\partial }{\partial x}u(0)=0.$

 - 2D wave equation (**Eq. 33** in the manuscript)

      **PDE**: $(\frac{\partial^2 }{\partial x^2}+4\frac{\partial^2 }{\partial y^2})u(x,y)=0, x\in[0,1], y\in[0,1]$

      **BC**:  $u(x,0)=u(x,1)=\frac{\partial }{\partial x}u(0,y)=0,$
               $u(0,y)=sin(\pi y)+0.5sin(4\pi y).$

 - 2D diffusion equation (**Eq. 35** in the manuscript)

      **PDE**: $(\frac{\partial}{\partial t}-0.01\frac{\partial^2 }{\partial x^2})u(x,t)=g(x,t), x\in[5,10], y\in[5,10]$

      **BC\IC**:  $u(5,t)=b_1(t),u(10,t)=b_2(t),u(x,5)=b_3(x).$

 - 2D viscoelastic Poiseuille problem (**Eq. 37** in the manuscript)

      **PDEs**: $\rho\frac{\partial}{\partial t}u(y,t)=-f+\frac{\partial}{\partial y}\tau_{xy}(y,t), t\in[0,4],$  
                $\eta_0\frac{\partial}{\partial y}u(y,t)=(\lambda\frac{\partial}{\partial t}+1)\tau_{xy}(y,t), y\in[0,1],$

      **BC\IC**:  $u(\pm0.5,t)=u(y,0)=0,$
                $\tau(y,0)=0.$

For more details in terms of mathematical proofs and numerical examples, please refer to our paper.

# Link

<https://doi.org/10.1016/j.cma.2023.116290>

<https://github.com/JinshuaiBai/PIRBN>

# Enviornmental settings

```
pip install -r requirements.txt
```
