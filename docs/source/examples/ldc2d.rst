Lid Driven Cavity Flow
======================

This guide introduces to how to build a PINN model for simulating the 2d Lid Driven Cavity (LDC) flow in PaddleScience.

- **Use case introduction**

The LDC problem mimicks a liquid-filled container with the lid moving in the horizontal direction at a constant speed. The goal is to calculate the velocity of the liquid at each interior point in the container when the system is in the steady state.

Following graphs show the results generated from training a 100 by 100 grid. The vertical and horizontal components of velocity are displayed separately.


.. image:: ../img/ldc2d_u_100x100.png
	   :width: 500
	   :align: center


.. image:: ../img/ldc2d_v_100x100.png
	   :width: 500
	   :align: center
	   
- **How to construct a PINN model**

A PINN model is jointly composed using what used to be a traditional PDE setup and a neural net approximating the solution. The PDE part includes specific differential equations enforcing the physical law, a geometry that bounds the problem domain and the initial and boundary value conditions which make it possible to find a solution. The neural net part can take variants of a typical feed forward network widely found in deep learning toolkits.

To obtain the PINN model requires training the neural net. It's in this phase that the information of the PDE gets instilled into the neural net through back propagation. The loss function plays a crucial role in controling how this information gets dispensed emphasizing different aspects of the PDE, for instance, by adjusting the weights for the equation residues and the boundary values.

Once the concept is clear, next let's take a look at how this translates into the ldc2d example.

- **Constructing Geometry**


First, define the problem geometry using the `psci.geometry` module interface. In this example,
the geometry is a rectangle with the origin at coordinates (-0.05, -0.05) and the extent set
to (0.05, 0.05).

    .. code-block::

        geo = psci.geometry.Rectangular(origin=(-0.05, -0.05), extent=(0.05, 0.05))


Next, add boundaries to the geometry, these boundaries will be used in PDE. 
Note that the `geo.add_boundary` function is only used for boundaries with physical constraints. 

    .. code-block::
        
        geo.add_boundary(name="top", criteria=lambda x, y: abs(y - 0.05) < 1e-5)
        geo.add_boundary(name="down", criteria=lambda x, y: abs(y + 0.05) < 1e-5)
        geo.add_boundary(name="left", criteria=lambda x, y: abs(x + 0.05) < 1e-5)
        geo.add_boundary(name="right", criteria=lambda x, y: abs(x - 0.05) < 1e-5)


Once the domain are prepared, a discretization recipe should be given.

    .. code-block::

        geo_disc = geo.discretize(npoints=npoints, method="uniform")

- **Constructing PDE**


After defining Geometry part, define the PDE equations to solve. In this example, the equations are a 2d
Navier Stokes. This equation is present in the package, and one only needs to
create a `psci.pde.NavierStokes` object to set up the equation.


    .. code-block::
        
        pde = psci.pde.NavierStokes(
            nu=0.01, rho=1.0, dim=2, time_dependent=False, weight=0.0001)

Next, add boundaries equations for PDE. 
The boundary equations in PDE are strongly bound to the boundary definitions in geometry. 
The physical information on the  boundaries needs to be set and then added using `pde.add_bc`.

    .. code-block::
     
        weight_top_u = lambda x, y: 1.0 - 20.0 * abs(x)
        bc_top_u = psci.bc.Dirichlet('u', rhs=1.0, weight=weight_top_u)
        bc_top_v = psci.bc.Dirichlet('v', rhs=0.0)
        bc_down_u = psci.bc.Dirichlet('u', rhs=0.0)
        bc_down_v = psci.bc.Dirichlet('v', rhs=0.0)
        bc_left_u = psci.bc.Dirichlet('u', rhs=0.0)
        bc_left_v = psci.bc.Dirichlet('v', rhs=0.0)
        bc_right_u = psci.bc.Dirichlet('u', rhs=0.0)
        bc_right_v = psci.bc.Dirichlet('v', rhs=0.0)

        pde.add_bc("top", bc_top_u, bc_top_v)
        pde.add_bc("down", bc_down_u, bc_down_v)
        pde.add_bc("left", bc_left_u, bc_left_v)
        pde.add_bc("right", bc_right_u, bc_right_v)

Once the equation and the problem domain are prepared, a discretization
recipe should be given. This recipe will be used to generate the training data
before training starts. Currently, the 2d space can be discretized into a N by M
grid, 101 by 101 in this example specifically.

    .. code-block::

        pde_disc = pde.discretize(geo_disc=geo_disc)


- **Constructing the neural net**


Now the PDE part is almost done, we move on to constructing the neural net.
It's straightforward to define a fully connected network by creating a `psci.network.FCNet` object.
Following is how we create an FFN of 10 hidden layers with 20 neurons on each, using hyperbolic
tangent as the activation function.

    .. code-block::

        net = psci.network.FCNet(
            num_ins=2,
            num_outs=3,
            num_layers=10,
            hidden_size=20,
            dtype="float32",
            activation='tanh')

Next, one of the most important steps is define the loss function. Here we use L2 loss.

    .. code-block::
     
	    loss = psci.loss.L2(p=2)


By design, the `loss` object conveys complete information of the PDE and hence the
latter is eclipsed in further steps. Now combine the neural net and the loss and we
create the `psci.algorithm.PINNs` model algorithm.

    .. code-block::

        algo = psci.algorithm.PINNs(net=net, loss=loss)


Next, by plugging in an Adam optimizer, a solver is contructed and you are ready
to kick off training. In this example, the Adam optimizer is used and is given
a learning rate of 0.001. 

The `psci.solver.Solver` class bundles the PINNs model, which is called `algo` here,
and the optimizer, into a solver object that exposes the `solve` interface.
`solver.solve` accepts three key word arguments. `num_epoch` specicifies how many
epoches for each batch.


    .. code-block::

        opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())
        solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)
        solution = solver.solve(num_epoch=30000)


Finally, `solver.solve` returns a function that calculates the solution value
for given points in the geometry. Apply the function to the geometry, convert the
outputs to Numpy and then you can verify the results. 

`psci.visu.save_vtk` is a helper utility for quick visualization. It saves
the graphs in vtp file which one can play using `Paraview <https://www.paraview.org/>`_.

    .. code-block::
    
        psci.visu.save_vtk(geo_disc=pde_disc.geometry, data=solution)
