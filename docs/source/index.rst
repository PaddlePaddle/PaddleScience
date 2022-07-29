Welcome to PaddleScience’s documentation
=========================================

PaddleScience is SDK and library for developing AI-driven scientific computing applications based on PaddlePaddle. Current version is v1.0 Beta.

PaddleScience extends the PaddlePaddle framework with reusable software components for developing novel scientific computing applications. Such new applications include Physics-informed machine learning, neural network based PDE solvers, machine learning for CFD, and so on.

PaddleScience is currently under active development. Its design is evolving and its APIs are subject to change.

-----


- **Core features and organization**

  PaddleScience currently focuses on the PINNs model. The core components are as follows:

   - Geometry, a declarative interface for defining the geometric domain. 

   - PDE, delineating partial differential equations in symbolic forms. Specific PDEs derive the base PDE class.

   - Loss, defining what exact penalties are enforced during the training process. By default, the L2 loss is applied. In the current design, the total loss is a weighted sum of four parts, the equation loss, the boundary condition loss, the initial condition loss and the data loss.

   - Optimizer, specifying which optimizer to use for training. Adam is the default option. More optimizers, such as BFGS, will be available in the future.

   - Solver, managing the training process given the training data.

   - Visualization, an easy access to the graph drawing utilities.

The component organization is illustrated in the following figure.
  

.. image:: img/pscicode.png

-----


* Project on GitHub: https://github.com/PaddlePaddle/PaddleScience
* Project on Gitee:  https://gitee.com/paddlepaddle/PaddleScience


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   Installation <getting_started/installation>
   Running Example <getting_started/howto>
   
.. toctree::
   :maxdepth: 2
   :caption: API Reference

   paddlescience.pde <api/pde>
   paddlescience.geometry <api/geometry>
   paddlescience.bc <api/bc>
   paddlescience.ic <api/ic>
   paddlescience.network <api/network>
   paddlescience.loss <api/loss>
   paddlescience.algorithm <api/algorithm>
   paddlescience.optimizer <api/optimizer>
   paddlescience.solver <api/solver>
   paddlescience.visu <api/visu>
   paddlescience.config <api/config>

   
.. toctree::
   :maxdepth: 2
   :caption: Example / Demo

   Lid Driven Cavity Flow  <examples/ldc2d>
   Darcy Flow in Porous Medium <examples/darcy>
   3D Unsteady Cylinder Flow with Discrete Method <examples/cylinder3d>
   2D Unsteady Cylinder Flow with Continuous Method <examples/cylinder2d>
   Vortex induced Vibration <examples/viv>

	     
Indices and tables
====================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
