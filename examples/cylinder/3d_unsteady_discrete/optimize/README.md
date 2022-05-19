[//]: <> (title: Flow around a cylinder use case tutorial, author: Xiandong Liu @liuxiandong at baidu.com)


# Flow around a Cylinder

This guide introduces to how to build a PINN model for simulating the flow around a cylinder in PaddleScience.

This demo is based on Paddle static graph mode, including the following parts:

 - Use the 1.0 API of PaddleScience suite to complete the construction of geometry, PDE, BC, network, algorithm, etc;

 - Note that the loss part is directly implemented based on the new automatic differentiation mechanism of the Paddle framework, and the provided procedural automatic differentiation interface, completes the definition of the eq_loss part, and forms the overall loss together with bc_loss and data_loss;
 
 - After the program is built, the distributed data parallel and gradient accumulation technology, computational graph optimization technology and new actuator execution optimization technology are used.

There are significant improvements in performance and memory occupation. It is worth mentioning that due to the improvement of calculation logic of the new automatic differentiation mechanism, the single machine performance is improved by 15%.

In terms of `scale`, due to the introduction of distributed technology, 100 million grid-scale training can be achieved, and the estimated time required for training 2000 epochs is 266 minutes. Weakly scalable data on distributed systems are as follows:
| Number of GPU | Number of points | Total points | Scaling | 
|---|---|---|---|
|N1C1 | 200k | 200k | 1 | 
|N1C8 | 200k | 1.6M | 0.99 | 
|N2C16 | 200k | 3.2M | 0.96 | 
|N4C32 | 200k | 6.4M | 0.93 | 

Note that N1C8 stands for using 1 node and 8 GPUs. Others are the same.
