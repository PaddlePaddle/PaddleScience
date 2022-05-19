[//]: <> (title: Flow around a cylinder use case tutorial, author: Xiandong Liu @liuxiandong at baidu.com)


# Flow around a Cylinder

This guide introduces to how to build a PINN model for simulating the flow around a cylinder in PaddleScience.

This demo is based on Paddle static graph mode, including the following parts:

 - Use v1.0 API, including geometry, pde, bc, network, algorithm and other parts;

 - Note that the loss part is directly implemented based on the new automatic differentiation mechanism of the Paddle framework, and the provided procedural automatic differentiation interface, completes the definition of the eq_loss part, and forms the overall loss together with bc_loss and data_loss;
 
 - Distributed data parallelism + gradient accumulation technology, computational graph optimization technology, and new executor execution optimization technology are used in the demo.

In terms of `performance`, due to the new automatic differentiation mechanism, calculation graph optimization, and new executor execution optimization technology, the performance and video memory have been greatly improved. It is worth mentioning that due to the improvement of the new automatic differentiation mechanism in computing logic 15% increase.

In terms of `scale`, due to the introduction of distributed technology, 100 million grid-scale training can be achieved, and the estimated time required for training 2000 epochs is 266 minutes. Weakly scalable data on distributed systems are as follows:
|gpus | points per gpu | total points | scaling | 
|---|---|---|---|
|N1C1 | 20w | 20w | 1 | 
|N1C8 | 20w | 160w | 0.99 | 
|N2C16 | 20w | 320w | 0.96 | 
|N4C32 | 20w | 640w | 0.93 | 
