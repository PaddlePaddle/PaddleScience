# Physics-informed radial basis network (PIRBN)

This repository provides numerical examples of the **physics-informed radial basis network** (**PIRBN**).

Physics-informed neural network (PINN) has recently gained increasing interest in computational  mechanics.

This work starts from studying the training dynamics of PINNs via the nerual tangent kernel (NTK) theory. Based on numerical experiments, we found:

- PINNs tend to be a **local approximator** during the training
- For PINNs who fail to be a local apprixmator, the physics-informed loss can be hardly minimised through training

Inspired by findings, we proposed the PIRBN, which can exhibit the local property intuitively. It has been demonstrated that the NTK theory is applicable for PIRBN. Besides, other PINN techniques can be directly migrated to PIRBNs.

Numerical examples include:

 - 1D sine funtion (**Eq. 13** in the manuscript)

      **PDE**: $\frac{\partial^2 }{\partial x^2}u(x)-4\mu^2\pi^2 sin(2\mu\pi(x))=0, x\in[0,1]$

      **BC**:  $u(0)=u(1)=0.$

 - 1D sine funtion (**Eq. 15** in the manuscript)
      **PDE**: $\frac{\partial^2 }{\partial x^2}u(x-100)-4\mu^2\pi^2 sin(2\mu\pi(x-100))=0, x\in[100,101]$

      **BC**:  $u(100)=u(101)=0.$

For more details in terms of mathematical proofs and numerical examples, please refer to our paper.

# Link

<https://doi.org/10.1016/j.cma.2023.116290>

<https://github.com/JinshuaiBai/PIRBN>

<https://arxiv.org/ftp/arxiv/papers/2304/2304.06234.pdf>

# Enviornmental settings

```
pip install -r requirements.txt
```
