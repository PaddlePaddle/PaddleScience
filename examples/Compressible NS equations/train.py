import paddlescience as psci
import numpy as np
import paddle


# Generate BC value
def GenBC(xy, bc_index):
    bc_value = np.zeros((len(bc_index), 4)).astype(np.float32)
    for i in range(len(bc_index)):
        id = bc_index[i]
        if abs(xy[id][0] - 0.0) < 1e-4:
            bc_value[i][0] = 3
            bc_value[i][1] = 0.0
            bc_value[i][2] = 1.4
            bc_value[i][3] = 1.0
        # if abs(xy[id][1] - 0.0) < 1e-4:
        #     bc_value[i][0] = 3
        #     bc_value[i][1] = 0.0
        #     bc_value[i][2] = 1.4
        #     bc_value[i][3] = 1.0
        # if abs(xy[id][1] - 1.0) < 1e-4:
        #     bc_value[i][0] = 3
        #     bc_value[i][1] = 0.0
        #     bc_value[i][2] = 1.4
        #     bc_value[i][3] = 1.0
        if abs(xy[id][0] - 1.0) < 1e-4 and xy[id][1] < 0.2:
            bc_value[i][0] = 0
            # bc_value[i][1] = 0.0
            # bc_value[i][2] = 1.4
            # bc_value[i][3] = 1.0
        # if abs(xy[id][0] - 1.0) and abs(xy[id][1] - 0.2)< 1e-4:
        #     bc_value[i][0] = 0
            # bc_value[i][1] = -0.5
            # bc_value[i][2] = 1.7
            # bc_value[i][3] = 1.53
    return bc_value


geo = psci.geometry.Rectangular(space_origin=(0.0, 0.0), space_extent=(1.0, 1.0))

pdes = psci.pde.NavierStokes_Compressed()

pdes, geo = psci.discretize(pdes, geo, space_nsteps=(50, 50))

bc_value = GenBC(geo.get_space_domain(), geo.get_bc_index())

pdes.set_bc_value(bc_value=bc_value)

net = psci.network.FCNet(
    num_ins=2,
    num_outs=4,
    num_layers=10,
    hidden_size=50,
    dtype="float32",
    activation='tanh')

# loss = psci.loss.L2(pdes=pdes,geo=geo)
loss = psci.loss.L2(pdes=pdes,
                    geo=geo,
                    eq_weight=1,
                    # bc_weight=bc_weight,
                    synthesis_method='norm')

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# net_state_dict = paddle.load('./checkpoint/net_params_2000')
# opt_state_dict = paddle.load('./checkpoint/opt_params_2000')
#
# net.set_state_dict(net_state_dict)
# opt.set_state_dict(opt_state_dict)

# Solver
solver = psci.solver.Solver(algo=algo, opt=opt)
solution = solver.solve(num_epoch=10000)

rslt = solution(geo)
u = rslt[:, 0]
v = rslt[:, 1]
rho = rslt[:, 2]
p = rslt[:, 3]

psci.visu.save_vtk(geo, u, filename="rslt_u")
psci.visu.save_vtk(geo, v, filename="rslt_v")
psci.visu.save_vtk(geo, rho, filename="rslt_rho")
psci.visu.save_vtk(geo, p, filename="rslt_p")
