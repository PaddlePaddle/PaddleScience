# Extended Physics-Informed Neural Networks (XPINNs): A Generalized Space-Time Domain Decomposition Based Deep Learning Framework for Nonlinear Partial Differential Equations

We propose a generalized space-time domain decomposition approach for the physics-informed neural networks (PINNs) to solve nonlinear partial differential equations (PDEs) on arbitrary complex-geometry domains. The proposed framework, named eXtended PINNs (XPINNs), further pushes the boundaries of both PINNs as well as conservative PINNs (cPINNs), which is a recently proposed domain decomposition approach in the PINN framework tailored to conservation laws. Compared to PINN, the XPINN method has large representation and parallelization capacity due to the inherent property of deployment of multiple neural networks in the smaller subdomains. Unlike cPINN, XPINN can be extended to any type of PDEs. Moreover, the domain can be decomposed in any arbitrary way (in space and time), which is not possible in cPINN. Thus, XPINN offers both space and time parallelization, thereby reducing the training cost more effectively. In each subdomain, a separate neural network is employed with optimally selected hyperparameters, e.g., depth/width of the network, number and location of residual points, activation function, optimization method, etc. A deep network can be employed in a subdomain with complex solution, whereas a shallow neural network can be used in a subdomain with relatively simple and smooth solutions. We demonstrate the versatility of XPINN by solving both forward and inverse PDE problems, ranging from one-dimensional to three-dimensional problems, from time-dependent to time-independent problems, and from continuous to discontinuous problems, which clearly shows that the XPINN method is promising in many practical problems. The proposed XPINN method is the generalization of PINN and cPINN methods, both in terms of applicability as well as domain decomposition approach, which efficiently lends itself to parallelized computation.

If you make use of the code or the idea/algorithm in your work, please cite our papers

References: For Domain Decomposition based PINN framework

1. A.D.Jagtap, G.E.Karniadakis, Extended Physics-Informed Neural Networks (XPINNs): A Generalized Space-Time Domain Decomposition Based Deep Learning Framework for Nonlinear Partial Differential Equations, Commun. Comput. Phys., Vol.28, No.5, 2002-2041, 2020. (<https://doi.org/10.4208/cicp.OA-2020-0164>)

       @article{jagtap2020extended,
       title={Extended physics-informed neural networks (xpinns): A generalized space-time domain decomposition based deep learning framework for nonlinear         partial differential equations},
       author={Jagtap, Ameya D and Karniadakis, George Em},
       journal={Communications in Computational Physics},
       volume={28},
       number={5},
       pages={2002--2041},
       year={2020}
       }

2. A.D.Jagtap, E. Kharazmi, G.E.Karniadakis, Conservative physics-informed neural networks on discrete domains for conservation laws: Applications to forward and inverse problems, Computer Methods in Applied Mechanics and Engineering, 365, 113028 (2020). (<https://doi.org/10.1016/j.cma.2020.113028>)

       @article{jagtap2020conservative,
       title={Conservative physics-informed neural networks on discrete domains for conservation laws: Applications to forward and inverse problems},
       author={Jagtap, Ameya D and Kharazmi, Ehsan and Karniadakis, George Em},
       journal={Computer Methods in Applied Mechanics and Engineering},
       volume={365},
       pages={113028},
       year={2020},
       publisher={Elsevier}
       }

3. K. Shukla, A.D. Jagtap, G.E. Karniadakis, Parallel Physics-Informed Neural Networks via Domain Decomposition, Journal of Computational Physics 447, 110683, (2021).

       @article{shukla2021parallel,
       title={Parallel Physics-Informed Neural Networks via Domain Decomposition},
       author={Shukla, Khemraj and Jagtap, Ameya D and Karniadakis, George Em},
       journal={Journal of Computational Physics},
       volume={447},
       pages={110683},
       year={2021},
       publisher={Elsevier}
       }

References: For adaptive activation functions

1. A.D. Jagtap, K.Kawaguchi, G.E.Karniadakis, Adaptive activation functions accelerate convergence in deep and physics-informed neural networks, Journal of Computational Physics, 404 (2020) 109136. (<https://doi.org/10.1016/j.jcp.2019.109136>)

       @article{jagtap2020adaptive,
       title={Adaptive activation functions accelerate convergence in deep and physics-informed neural networks},
       author={Jagtap, Ameya D and Kawaguchi, Kenji and Karniadakis, George Em},
       journal={Journal of Computational Physics},
       volume={404},
       pages={109136},
       year={2020},
       publisher={Elsevier}
       }

2. A.D.Jagtap, K.Kawaguchi, G.E.Karniadakis, Locally adaptive activation functions with slope recovery for deep and physics-informed neural networks, Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 20200334, 2020. (<http://dx.doi.org/10.1098/rspa.2020.0334>).

       @article{jagtap2020locally,
       title={Locally adaptive activation functions with slope recovery for deep and physics-informed neural networks},
       author={Jagtap, Ameya D and Kawaguchi, Kenji and Em Karniadakis, George},
       journal={Proceedings of the Royal Society A},
       volume={476},
       number={2239},
       pages={20200334},
       year={2020},
       publisher={The Royal Society}
       }

3. A.D. Jagtap, Y. Shin, K. Kawaguchi, G.E. Karniadakis, Deep Kronecker neural networks: A general framework for neural networks with adaptive activation functions, Neurocomputing, 468, 165-180, 2022. (<https://www.sciencedirect.com/science/article/pii/S0925231221015162>)

       @article{jagtap2022deep,
       title={Deep Kronecker neural networks: A general framework for neural networks with adaptive activation functions},
       author={Jagtap, Ameya D and Shin, Yeonjong and Kawaguchi, Kenji and Karniadakis, George Em},
       journal={Neurocomputing},
       volume={468},
       pages={165--180},
       year={2022},
       publisher={Elsevier}
       }

Recommended software versions: TensorFlow 1.14, Python 3.6, Latex (for plotting figures)

For any queries regarding the XPINN code, feel free to contact me : <ameya_jagtap@brown.edu>, <ameyadjagtap@gmail.com>

--------------------
原仓库 <https://github.com/AmeyaJagtap/XPINNs.git>


```shell
# 安装latex软件包，python依赖
apt install latex-cjk-all texlive-latex-extra cm-super dvipng -y
pip install -r requirements.txt

# GPU PaddlePaddle
pip install paddlepaddle-gpu
# CPU PaddlePaddle
# pip install paddlepaddle
```

已有数据集 data/XPINN_2D_PoissonEqn.mat，运行 python 文件

```shell
python XPINN_2D_PoissonsEqn.py
```
