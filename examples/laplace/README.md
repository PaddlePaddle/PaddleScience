# Laplace equation

This guide introduces how to build a PINN model for a simple Laplace equation in PaddleScience.

## Use case introduction

We consider following Laplace equation with Dirichlet bounday condition.

<div align="center">    
<img src="../../docs/source/img/laplaceeq.png" width = "400" align=center />
</div>


The analytical solution is 
<div align="center">
<img src="../../docs/source/img/laplacesolution.png" width = "300" align=center />
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
Laplace2d example.

### Constructing PDE

First, define the problem geometry using the `psci.geometry` module interface. In this example,
the geometry is a rectangle with the origin at coordinates (0.0, 0.0) and the extent set
to (1.0, 1.0).

```
geo = psci.geometry.Rectangular(
    space_origin=(0.0, 0.0), space_extent=(1.0, 1.0))
```

Next, define the PDE equations to solve. In this example, the equations are a 2d
Laplace equation. This equation is present in the package, and one only needs to
create a `psci.pde.Laplace2D` object to set up the equation.

```
pdes = psci.pde.Laplace2D()
```

Once the equation and the problem domain are prepared, a discretization
recipe should be given. This recipe will be used to generate the training data
before training starts. Currently, the 2d space can be discretized into a N by M
grid, 11 by 11 in this example specifically.

```
pdes, geo = psci.discretize(pdes, geo, space_steps=(11, 11))
```

As mentioned above, a valid problem setup relies on sufficient constraints on
the boundary and initial values. In this example, we use analytical solution on the boundary, and by calling `pdes.set_bc_value()` the
values are then passed to the PDE solver.
It's worth noting however that in general the boundary and initial value
conditions can be passed as a function to the solver rather than actual values.
That feature will be addressed in the future.

```
pdes.set_bc_value(bc_value=bc_value)
```

### Constructing the neural net

Now the PDE part is almost done, we move on to constructing the neural net.
It's straightforward to define a fully connected network by creating a `psci.network.FCNet` object.
Following is how we create an FFN of 5 hidden layers with 20 neurons on each, using hyperbolic
tangent as the activation function.

```
net = psci.network.FCNet(
    num_ins=2,
    num_outs=1,
    num_layers=5,
    hidden_size=20,
    dtype="float32",
    activation='tanh')
```

Next, one of the most important steps is define the loss function. Here we use L2
loss with custom weights assigned to the boundary values.

```
loss = psci.loss.L2(pdes=pdes, geo=geo)
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
solver = psci.solver.Solver(algo=algo, opt=opt)
solution = solver.solve(num_epoch=30000)
```

Finally, `solver.solve` returns a function that calculates the solution value
for given points in the geometry. Apply the function to the geometry, convert the
outputs to Numpy and then you can verify the results. 

`psci.visu.visu_vtk` is a helper utility for quick visualization. It saves
the graphs in vtp file which one can play using [Paraview](https://www.paraview.org/).

```
rslt = solution(geo).numpy()
psci.visu.save_vtk(geo, rslt, 'rslt_laplace_2d')
np.save(rslt, 'rslt_laplace_2d.npy')
```
