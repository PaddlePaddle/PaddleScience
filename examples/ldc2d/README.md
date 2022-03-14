[//]: <> (title: LDC use case tutorial, author: Tongxin Bai @baitongxin at baidu.com)


# Lid Driven Cavity Flow

This guide introduces to how to build a PINN model for simulating the 2d Lid Driven Cavity (LDC) flow in PaddleScience.


## Use case introduction

The LDC problem mimicks a liquid-filled container with the lid moving in the horizontal direction at a constant speed. The goal is to calculate the velocity of the liquid at each interior point in the container when the system is in the steady state.


Following graphs show the results generated from training a 100 by 100 grid. The vertical and horizontal components of velocity are displayed separately.


<div align="center">
<img src="../../docs/source/img/ldc2d_u_100x100.png" width = "500" align=center />
<img src="../../docs/source/img/ldc2d_v_100x100.png" width = "500" align=center />
</div>


## How to construct a PINN model

A PINN model is jointly composed using what used to be a traditional PDE setup and a neural net approximating the solution. The PDE part includes specific differential equations enforcing the physical law, a geometry that bounds the problem domain and the initial and boundary value conditions which make it possible to find a solution. The neural net part can take variants of a typical feed forward network widely found in deep learning toolkits.

To obtain the PINN model requires training the neural net. It's in this phase that the information of the PDE gets instilled into the neural net through back propagation. The loss function plays a crucial role in controling how this information gets dispensed emphasizing different aspects of the PDE, for instance, by adjusting the weights for the equation residues and the boundary values.

Once the concept is clear, next let's take a look at how this translates into the ldc2d example.



### Constructing PDE

First, define the problem geometry using the `psci.geometry` module interface. In this example,
the geometry is a rectangle with the origin at coordinates (-0.05, -0.05) and the extent set
to (0.05, 0.05).

```
geo = psci.geometry.Rectangular(
    space_origin=(-0.05, -0.05), space_extent=(0.05, 0.05))
```

Next, define the PDE equations to solve. In this example, the equations are a 2d
Navier Stokes. This equation is present in the package, and one only needs to
create a `psci.pde.NavierStokes2D` object to set up the equation.

```
pdes = psci.pde.NavierStokes2D(nu=0.01, rho=1.0)
```

Once the equation and the problem domain are prepared, a discretization
recipe should be given. This recipe will be used to generate the training data
before training starts. Currently, the 2d space can be discretized into a N by M
grid, 101 by 101 in this example specifically.

```
pdes, geo = psci.discretize(pdes, geo, space_steps=(101, 101))
```

As mentioned above, a valid problem setup relies on sufficient constraints on
the boundary and initial values. In this example, `GenBC` is the procedure that
generates the actual boundary values, and by calling `pdes.set_bc_value()` the
values are then passed to the PDE solver.
It's worth noting however that in general the boundary and initial value
conditions can be passed as a function to the solver rather than actual values.
That feature will be addressed in the future.

```
def GenBC(xy, bc_index):
    bc_value = np.zeros((len(bc_index), 2)).astype(np.float32)
    for i in range(len(bc_index)):
        id = bc_index[i]
        if abs(xy[id][1] - 0.05) < 1e-4:
            bc_value[i][0] = 1.0
            bc_value[i][1] = 0.0
        else:
            bc_value[i][0] = 0.0
            bc_value[i][1] = 0.0
    return bc_value

bc_value = GenBC(geo.space_domain, geo.bc_index)
pdes.set_bc_value(bc_value=bc_value, bc_check_dim=[0, 1])
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

Next, one of the most important steps is define the loss function. Here we use L2
loss with custom weights assigned to the boundary values.

```
bc_weight = GenBCWeight(geo.space_domain, geo.bc_index)
loss = psci.loss.L2(pdes=pdes,
                    geo=geo,
                    eq_weight=0.01,
                    bc_weight=bc_weight,
                    synthesis_method='norm',
                    run_in_batch=True)
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
epoches for each batch. `batch_size` specifies the batch size of data that each train
step works on. `checkpoint_freq` sets for how many epochs the network parameters are
saved in local file system.


```
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())
solver = psci.solver.Solver(algo=algo, opt=opt)
solution = solver.solve(num_epoch=30000, batch_size=None, checkpoint_freq=1000)
```

Finally, `solver.solve` returns a function that calculates the solution value
for given points in the geometry. Apply the function to the geometry, convert the
outputs to Numpy and then you can verify the results. 

`psci.visu.save_vtk` is a helper utility for quick visualization. It saves
the graphs in vtp file which one can play using [Paraview](https://www.paraview.org/).

```
rslt = solution(geo).numpy()
rslt = solution(geo)
u = rslt[:, 0]
v = rslt[:, 1]
u_and_v = np.sqrt(u * u + v * v)
psci.visu.save_vtk(geo, u, filename="rslt_u")
psci.visu.save_vtk(geo, v, filename="rslt_v")
psci.visu.save_vtk(geo, u_and_v, filename="u_and_v")
```
