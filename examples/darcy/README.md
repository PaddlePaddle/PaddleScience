# Darcy Flow In Porous Medium

This guide introduces how to build a PINN model for simulating 2d
Darcy flow in PaddleScience.

## Use case introduction

The Darcy flow example simulate solution (pressure) of following problem

<div align="center">    
<img src="../../docs/source/img/darcyeq.png" width = "400" align=center />
</div>

with <img src="../../source/img/darcyeqf.png" width = "260" align=center /> and the Dirichlet boundary condition is set to be 
<div align="center">
<img src="../../docs/source/img/darcyeqbc.png" width = "400" align=center />
</div>


Following graphs plot the pressure from training the model on a 100 by 100 grid.


<div align="center">    
<img src="../../docs/source/img/darcy_rslt_100x100.png" width = "700" align=center />
</div>

## How to construct a PINN model

A PINN model is jointly composed using what used to be a traditional PDE setup and
a neural net approximating the solution. The PDE part includes specific
differential equations enforcing the physical law, a geometry that bounds
the problem domain and the initial and boundary value conditions which make it
possible to find a solution. The neural net part can take variants of a typical
feed forward network widely found in deep learning toolkits.

To obtain the PINN model requires training the neural net. It's in this phase that
the information of the PDE gets instilled into the neural net through back propagation.
The loss function plays a crucial role in controlling how this information gets dispensed
emphasizing different aspects of the PDE, for instance, by adjusting the weights for
the equation residues and the boundary values.

Once the concept is clear, next let's take a look at how this translates into the
dacy2d example.

### Constructing Geometry

First, define the problem geometry using the `psci.geometry` module interface. In this example,
the geometry is a rectangle with the origin at coordinates (0.0, 0.0) and the extent set
to (1.0, 1.0).

```
geo = psci.geometry.Rectangular(origin=(0.0, 0.0), extent=(1.0, 1.0))
```

Next, add boundaries to the geometry, these boundaries will be used in PDE. 
Note that the `geo.add_boundary` function is only used for boundaries with physical constraints. 


```
geo.add_boundary(name="top", criteria=lambda x, y: y == 1.0)
geo.add_boundary(name="down", criteria=lambda x, y: y == 0.0)
geo.add_boundary(name="left", criteria=lambda x, y: x == 0.0)
geo.add_boundary(name="right", criteria=lambda x, y: x == 1.0)
```

Once the domain are prepared, a discretization recipe should be given.

```
geo_disc = geo.discretize(npoints=npoints, method="uniform")
```

### Constructing PDE

After defining Geometry part, define the PDE equations to solve. In this example, the equations are a 2d
Laplace. This equation is present in the package, and one only needs to
create a `psci.pde.Poisson` object to set up the equation.

```
pde = psci.pde.Poisson(dim=2, rhs=ref_rhs)
```

Next, add boundaries equations for PDE. 
The boundary equations in PDE are strongly bound to the boundary definitions in geometry. 
The physical information on the  boundaries needs to be set and then added using `pde.add_bc`.


```
bc_top = psci.bc.Dirichlet('u', rhs=ref_sol)
bc_down = psci.bc.Dirichlet('u', rhs=ref_sol)
bc_left = psci.bc.Dirichlet('u', rhs=ref_sol)
bc_right = psci.bc.Dirichlet('u', rhs=ref_sol)

pde.add_bc("top", bc_top)
pde.add_bc("down", bc_down)
pde.add_bc("left", bc_left)
pde.add_bc("right", bc_right)
```

Once the equation and the problem domain are prepared, a discretization
recipe should be given. This recipe will be used to generate the training data
before training starts.

```
pde_disc = pde.discretize(geo_disc=geo_disc)
```

### Constructing the neural net

Now the PDE part is almost done, we move on to constructing the neural net.
It's straightforward to define a fully connected network by creating a `psci.network.FCNet` object.
Following is how we create an FFN of 5 hidden layers with 20 neurons on each, using hyperbolic
tangent as the activation function.

```
net = psci.network.FCNet(
    num_ins=2,
    num_outs=3,
    num_layers=5,
    hidden_size=20,
    dtype="float32",
    activation='tanh')
```

Next, one of the most important steps is define the loss function. Here we use L2 loss.


```
loss = psci.loss.L2()
```

By design, the `loss` object conveys complete information of the PDE and hence the
latter is eclipsed in further steps. Now combine the neural net and the loss and we
create the `psci.algorithm.PINNs` model algorithm.

```
algo = psci.algorithm.PINNs(net=net, loss=loss)
```

Next, by plugging in an Adam optimizer, a solver is contructed and you are ready
to kick off training. In this example, the Adam optimizer is used and is given
a learning rate of 0.001. 

The `psci.solver.Solver` class bundles the PINNs model, which is called `algo` here,
and the optimizer, into a solver object that exposes the `solve` interface.
`solver.solve` accepts three key word arguments. `num_epoch` specicifies how many
epoches for each batch. 


```
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())
solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)
solution = solver.solve(num_epoch=10000)
```

Finally, `solver.solve` returns a function that calculates the solution value
for given points in the geometry. Apply the function to the geometry, convert the
outputs to Numpy and then you can verify the results. 

`psci.visu.save_vtk` is a helper utility for quick visualization. It saves
the graphs in vtp file which one can play using [Paraview](https://www.paraview.org/).

```
psci.visu.save_vtk(geo_disc=pde_disc.geometry, data=solution)
```
