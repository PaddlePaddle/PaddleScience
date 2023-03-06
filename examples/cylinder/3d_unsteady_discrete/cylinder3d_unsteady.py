# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Created in Mar. 2023
@author: Guan Wang
"""
import os
import sys
import paddle
import numpy as np
import sample_boundary_training_data as sample_data
from load_lbm_data import load_vtk
import paddlescience as psci


def txyz_nomalization(t_factor, xyz_factor, txyz_input):
    """_summary_

    Args:
        t_factor (_type_): _description_
        xyz_factor (_type_): _description_
        txyz_input (_type_): _description_

    Returns:
        _type_: _description_
    """
    time = txyz_input[:, 0] / t_factor
    x_cord = txyz_input[:, 1] / xyz_factor
    y_cord = txyz_input[:, 2] / xyz_factor
    z_cord = txyz_input[:, 3] / xyz_factor
    return time, x_cord, y_cord, z_cord


def print_for_check(check_list, varname_list):
    """_summary_

    Args:
        check_list (_type_): _description_
    """
    for _, (term, name_str) in enumerate(zip(check_list, varname_list)):
        print(f"{name_str}={term.shape} {term.mean().item():.10f}")


def normalized_bc(origion_list, t_factor, xyz_factor, random_time):
    """normalize bc data time and coordinates

    Args:
        origion_list (_type_): _description_
        t_factor (_type_): _description_
        xyz_factor (_type_): _description_
        random_time (_type_): _description_

    Returns:
        _type_: _description_
    """
    if random_time is True:
        time = np.random.uniform(
            low=time_array[0] / t_factor,
            high=time_array[-1] / t_factor,
            size=len(origion_list[:, 0]))
    else:
        time = origion_list[:, 0] / t_factor
    x_cord = origion_list[:, 1] / xyz_factor
    y_cord = origion_list[:, 2] / xyz_factor
    z_cord = origion_list[:, 3] / xyz_factor
    return time, x_cord, y_cord, z_cord


if __name__ == "__main__":
    RENOLDS_NUMBER = 3900
    U0 = 0.1
    D_CYLINDER = 80.0
    RHO = 1.0
    NU = RHO * U0 * D_CYLINDER / RENOLDS_NUMBER

    T_STAR = D_CYLINDER / U0  # 800
    XYZ_STAR = D_CYLINDER  # 80
    UVW_STAR = U0  # 0.1
    p_star = RHO * U0 * U0  # 0.01

    # read configuration file : config.yaml
    dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
    config = psci.utils.get_config(
        fname=dirname + r'/config.yaml', config_index="hyper parameters")
    if config is not None:
        # number of epoch
        num_epoch = config['number of epoch']
        learning_rate = config['learning rate']
        hidden_size = config["hidden size"]
        # batch size
        bs = {}
        bs['interior'] = config['batch size']['interior']

        bs['inlet'] = config['batch size']['inlet']
        bs['outlet'] = config['batch size']['outlet']
        bs['cylinder'] = config['batch size']['cylinder']
        bs['top'] = config['batch size']['top']
        bs['bottom'] = config['batch size']['bottom']

        bs['ic'] = config['batch size']['initial condition']
        bs['supervised'] = config['batch size']['supervised']

        # losses weight
        ic_wgt = config['weight of losses']['initial condition']
        eq_wgt = config['weight of losses']['pde']

        front_wgt = config['weight of losses']['front']
        back_wgt = config['weight of losses']['back']
        inlet_wgt = config['weight of losses']['left inlet']
        outlet_wgt = config['weight of losses']['right outlet']
        top_wgt = config['weight of losses']['top']
        bottom_wgt = config['weight of losses']['bottom']
        cylinder_wgt = config['weight of losses']['cylinder']

        sup_wgt = config['weight of losses']['supervised']

        # simulated annealing
        w_epoch = config['Simulated Annealing']['warm epochs']

        # epoch number
        seed_number = config["random seed"]
    else:
        print("Error : mssing configure files !")

    USE_RANDOM_TIME = False

    # fix the random seed
    paddle.seed(seed_number)
    np.random.seed(seed_number)

    # time array
    INITIAL_TIME = 200000
    START_TIME = 200050
    END_TIME = 204950
    TIME_STEP = 50
    TIME_NUMBER = int((END_TIME - START_TIME) / TIME_STEP) + 1
    time_list = np.linspace(
        int((START_TIME - INITIAL_TIME) / TIME_STEP),
        int((END_TIME - INITIAL_TIME) / TIME_STEP),
        TIME_NUMBER,
        endpoint=True).astype(int)
    time_tmp = time_list * TIME_STEP
    time_index = np.random.choice(
        time_list, int(TIME_NUMBER / 2.5), replace=False)
    time_index.sort()
    time_array = time_index * TIME_STEP
    print(f"TIME_NUMBER = {TIME_NUMBER}")
    print(f"time_list = {time_list}")
    print(f"time_tmp = {time_tmp}")
    print(f"time_index = {time_index}")
    print(f"time_array = {time_array}")

    # initial value
    ic_name = dirname + r'/data/LBM_result/cylinder3d_2023_1_31_LBM_'
    txyz_uvwpe_ic = load_vtk(
        [0],
        t_step=TIME_STEP,
        load_uvwp=True,
        load_txyz=True,
        name_wt_time=ic_name)[0]
    init_t = txyz_uvwpe_ic[:, 0] / T_STAR
    init_x = txyz_uvwpe_ic[:, 1] / XYZ_STAR
    init_y = txyz_uvwpe_ic[:, 2] / XYZ_STAR
    init_z = txyz_uvwpe_ic[:, 3] / XYZ_STAR
    init_u = txyz_uvwpe_ic[:, 4] / UVW_STAR
    init_v = txyz_uvwpe_ic[:, 5] / UVW_STAR
    init_w = txyz_uvwpe_ic[:, 6] / UVW_STAR
    init_p = txyz_uvwpe_ic[:, 7] / p_star
    print_for_check(
        [init_t, init_x, init_y, init_z, init_u, init_v, init_w, init_p], [
            'init_t', 'init_x', 'init_y', 'init_z', 'init_u', 'init_v',
            'init_w', 'init_p'
        ])

    # supervised data
    sup_name = dirname + r"/data/sup_data/supervised_"
    txyz_uvwpe_s_new = load_vtk(
        time_index,
        t_step=TIME_STEP,
        load_uvwp=True,
        load_txyz=True,
        name_wt_time=sup_name)
    txyz_uvwpe_s = np.zeros((0, 8))
    for x in txyz_uvwpe_s_new:
        txyz_uvwpe_s = np.concatenate((txyz_uvwpe_s, x[:, :]), axis=0)
    sup_t = txyz_uvwpe_s[:, 0] / T_STAR
    sup_x = txyz_uvwpe_s[:, 1] / XYZ_STAR
    sup_y = txyz_uvwpe_s[:, 2] / XYZ_STAR
    sup_z = txyz_uvwpe_s[:, 3] / XYZ_STAR
    sup_u = txyz_uvwpe_s[:, 4] / UVW_STAR
    sup_v = txyz_uvwpe_s[:, 5] / UVW_STAR
    sup_w = txyz_uvwpe_s[:, 6] / UVW_STAR
    sup_p = txyz_uvwpe_s[:, 7] / p_star
    print_for_check([sup_t, sup_x, sup_y, sup_z, sup_u, sup_v, sup_w, sup_p], [
        'sup_t', 'sup_x', 'sup_y', 'sup_z', 'sup_u', 'sup_v', 'sup_w', 'sup_p'
    ])

    # num of interior points
    NUM_POINTS = 10000

    # discretize node by geo
    inlet_txyz, outlet_txyz, _, _, top_txyz, bottom_txyz, cylinder_txyz, interior_txyz = \
        sample_data.sample_data(
            t_num=TIME_NUMBER, t_index=time_index, t_step=TIME_STEP, nr_points=NUM_POINTS)

    # interior nodes discre
    i_t, i_x, i_y, i_z = normalized_bc(
        interior_txyz,
        t_factor=T_STAR,
        xyz_factor=XYZ_STAR,
        random_time=USE_RANDOM_TIME)
    print_for_check([i_t, i_x, i_y, i_z], ['i_t', 'i_x', 'i_y', 'i_z'])

    # bc inlet nodes discre
    b_inlet_t, b_inlet_x, b_inlet_y, b_inlet_z = normalized_bc(
        inlet_txyz,
        t_factor=T_STAR,
        xyz_factor=XYZ_STAR,
        random_time=USE_RANDOM_TIME)
    print_for_check([b_inlet_t, b_inlet_x, b_inlet_y, b_inlet_z],
                    ['b_inlet_t', 'b_inlet_x', 'b_inlet_y', 'b_inlet_z'])

    # bc outlet nodes discre
    b_outlet_t, b_outlet_x, b_outlet_y, b_outlet_z = normalized_bc(
        outlet_txyz,
        t_factor=T_STAR,
        xyz_factor=XYZ_STAR,
        random_time=USE_RANDOM_TIME)
    print_for_check([b_outlet_t, b_outlet_x, b_outlet_y, b_outlet_z],
                    ['b_outlet_t', 'b_outlet_x', 'b_outlet_y', 'b_outlet_z'])

    # bc cylinder nodes discre
    b_cylinder_t, b_cylinder_x, b_cylinder_y, b_cylinder_z = normalized_bc(
        cylinder_txyz,
        t_factor=T_STAR,
        xyz_factor=XYZ_STAR,
        random_time=USE_RANDOM_TIME)
    print_for_check(
        [b_cylinder_t, b_cylinder_x, b_cylinder_y, b_cylinder_z],
        ['b_cylinder_t', 'b_cylinder_x', 'b_cylinder_y', 'b_cylinder_z'])

    # bc-top nodes discre
    b_top_t, b_top_x, b_top_y, b_top_z = normalized_bc(
        top_txyz,
        t_factor=T_STAR,
        xyz_factor=XYZ_STAR,
        random_time=USE_RANDOM_TIME)
    print_for_check([b_top_t, b_top_x, b_top_y, b_top_z],
                    ['b_top_t', 'b_top_x', 'b_top_y', 'b_top_z'])

    # bc-bottom nodes discre
    b_bottom_t, b_bottom_x, b_bottom_y, b_bottom_z = normalized_bc(
        bottom_txyz,
        t_factor=T_STAR,
        xyz_factor=XYZ_STAR,
        random_time=USE_RANDOM_TIME)
    print_for_check([b_bottom_t, b_bottom_x, b_bottom_y, b_bottom_z],
                    ['b_bottom_t', 'b_bottom_t', 'b_bottom_t', 'b_bottom_t'])

    # bc & interior nodes for nn
    inputeq = np.stack((i_t, i_x, i_y, i_z), axis=1)
    inputbc1 = np.stack((b_inlet_t, b_inlet_x, b_inlet_y, b_inlet_z), axis=1)
    inputbc2 = np.stack(
        (b_outlet_t, b_outlet_x, b_outlet_y, b_outlet_z), axis=1)
    inputbc3 = np.stack(
        (b_cylinder_t, b_cylinder_x, b_cylinder_y, b_cylinder_z), axis=1)
    inputbc6_top = np.stack((b_top_t, b_top_x, b_top_y, b_top_z), axis=1)
    inputbc7_bottom = np.stack(
        (b_bottom_t, b_bottom_x, b_bottom_y, b_bottom_z), axis=1)

    inputic = np.stack((init_t, init_x, init_y, init_z), axis=1)
    inputsup = np.stack((sup_t, sup_x, sup_y, sup_z), axis=1)
    refsup = np.stack((sup_u, sup_v, sup_w), axis=1)

    # N-S, Re=3900, D=80, u=0.1, nu=80/3900; nu = rho u D / Re = 1.0 * 0.1 * 80 / 3900
    pde = psci.pde.NavierStokes(nu=NU, rho=1.0, dim=3, time_dependent=True)

    # set bounday condition
    bc_inlet_u = psci.bc.Dirichlet("u", rhs=0.1 / UVW_STAR)
    bc_inlet_v = psci.bc.Dirichlet("v", rhs=0.0)
    bc_inlet_w = psci.bc.Dirichlet("w", rhs=0.0)
    bc_cylinder_u = psci.bc.Dirichlet("u", rhs=0.0)
    bc_cylinder_v = psci.bc.Dirichlet("v", rhs=0.0)
    bc_cylinder_w = psci.bc.Dirichlet("w", rhs=0.0)
    bc_outlet_p = psci.bc.Dirichlet("p", rhs=0.0)
    bc_top_u = psci.bc.Dirichlet("u", rhs=0.1 / UVW_STAR)
    bc_top_v = psci.bc.Dirichlet("v", rhs=0.0)
    bc_top_w = psci.bc.Dirichlet("w", rhs=0.0)
    bc_bottom_u = psci.bc.Dirichlet("u", rhs=0.1 / UVW_STAR)
    bc_bottom_v = psci.bc.Dirichlet("v", rhs=0.0)
    bc_bottom_w = psci.bc.Dirichlet("w", rhs=0.0)

    # add bounday and boundary condition
    pde.set_bc("inlet", bc_inlet_u, bc_inlet_v, bc_inlet_w)
    pde.set_bc("cylinder", bc_cylinder_u, bc_cylinder_v, bc_cylinder_w)
    pde.set_bc("outlet", bc_outlet_p)
    pde.set_bc("top", bc_top_u, bc_top_v, bc_top_w)
    pde.set_bc("bottom", bc_bottom_u, bc_bottom_v, bc_bottom_w)

    # add initial condition
    ic_u = psci.ic.IC("u", rhs=init_u)
    ic_v = psci.ic.IC("v", rhs=init_v)
    ic_w = psci.ic.IC("w", rhs=init_w)
    ic_p = psci.ic.IC("p", rhs=init_p)
    pde.set_ic(ic_u, ic_v, ic_w)  # 添加ic_p会使ic_loss非常大

    # Network
    net = psci.network.FCNet(
        num_ins=4,
        num_outs=4,
        num_layers=6,
        hidden_size=hidden_size,
        activation="tanh")

    outeq = net(inputeq)
    outbc1 = net(inputbc1)
    outbc2 = net(inputbc2)
    outbc3 = net(inputbc3)
    outbc6 = net(inputbc6_top)
    outbc7 = net(inputbc7_bottom)
    outic = net(inputic)
    outsup = net(inputsup)

    # eq loss
    losseq1 = psci.loss.EqLoss(pde.equations[0], netout=outeq)
    losseq2 = psci.loss.EqLoss(pde.equations[1], netout=outeq)
    losseq3 = psci.loss.EqLoss(pde.equations[2], netout=outeq)
    losseq4 = psci.loss.EqLoss(pde.equations[3], netout=outeq)

    # bc loss
    lossbc1 = psci.loss.BcLoss("inlet", netout=outbc1)
    lossbc2 = psci.loss.BcLoss("outlet", netout=outbc2)
    lossbc3 = psci.loss.BcLoss("cylinder", netout=outbc3)
    lossbc6 = psci.loss.BcLoss("top", netout=outbc6)
    lossbc7 = psci.loss.BcLoss("bottom", netout=outbc7)

    # ic loss
    lossic = psci.loss.IcLoss(netout=outic)

    # supervise loss
    losssup = psci.loss.DataLoss(netout=outsup[0:3], ref=refsup)

    # total loss
    loss = losseq1 * eq_wgt + losseq2 * eq_wgt + losseq3 * eq_wgt + losseq4 * eq_wgt + \
        lossbc1 * inlet_wgt + \
        lossbc2 * outlet_wgt + \
        lossbc3 * cylinder_wgt + \
        lossbc6 * top_wgt + \
        lossbc7 * bottom_wgt + \
        lossic * ic_wgt + \
        losssup * sup_wgt

    # Algorithm
    algo = psci.algorithm.PINNs(net=net, loss=loss)
    _lr = psci.optimizer.lr.Cosine(
        num_epoch,
        1,
        learning_rate=learning_rate,
        warmup_epoch=w_epoch,
        by_epoch=True)()
    opt = psci.optimizer.Adam(learning_rate=_lr, parameters=net.parameters())

    # Read validation reference for time step : 0, 99
    lbm_01 = load_vtk(
        [0],
        t_step=TIME_STEP,
        load_uvwp=True,
        load_txyz=True,
        name_wt_time=ic_name)[0]
    lbm_99 = load_vtk(
        [99],
        t_step=TIME_STEP,
        load_uvwp=True,
        load_txyz=True,
        name_wt_time=ic_name)[0]
    tmp = txyz_nomalization(T_STAR, XYZ_STAR, lbm_01)
    lbm_01[:, 0:4] = np.stack(tmp, axis=1)
    tmp = txyz_nomalization(T_STAR, XYZ_STAR, lbm_99)
    lbm_99[:, 0:4] = np.stack(tmp, axis=1)
    lbm = [lbm_01, lbm_99]

    # Solver
    solver = psci.solver.Solver(pde=pde, algo=algo, opt=opt, lbm=lbm)

    # Solve
    solution = solver.solve(num_epoch=num_epoch, bs=bs)
    p_star = RHO * U0 * U0  # 0.01

    # read configuration file : config.yaml
    dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
    config = psci.utils.get_config(
        fname=dirname + r'/config.yaml', config_index="hyper parameters")
    if config is not None:
        # number of epoch
        num_epoch = config['number of epoch']
        learning_rate = config['learning rate']
        hidden_size = config["hidden size"]
        # batch size
        bs = {}
        bs['interior'] = config['batch size']['interior']

        bs['inlet'] = config['batch size']['inlet']
        bs['outlet'] = config['batch size']['outlet']
        bs['cylinder'] = config['batch size']['cylinder']
        bs['top'] = config['batch size']['top']
        bs['bottom'] = config['batch size']['bottom']

        bs['ic'] = config['batch size']['initial condition']
        bs['supervised'] = config['batch size']['supervised']

        # losses weight
        ic_wgt = config['weight of losses']['initial condition']
        eq_wgt = config['weight of losses']['pde']

        front_wgt = config['weight of losses']['front']
        back_wgt = config['weight of losses']['back']
        inlet_wgt = config['weight of losses']['left inlet']
        outlet_wgt = config['weight of losses']['right outlet']
        top_wgt = config['weight of losses']['top']
        bottom_wgt = config['weight of losses']['bottom']
        cylinder_wgt = config['weight of losses']['cylinder']

        sup_wgt = config['weight of losses']['supervised']

        # simulated annealing
        w_epoch = config['Simulated Annealing']['warm epochs']

        # epoch number
        seed_number = config["random seed"]
    else:
        print("Error : mssing configure files !")

    USE_RANDOM_TIME = False

    # fix the random seed
    paddle.seed(seed_number)
    np.random.seed(seed_number)

    # time array
    INITIAL_TIME = 200000
    START_TIME = 200050
    END_TIME = 204950
    TIME_STEP = 50
    TIME_NUMBER = int((END_TIME - START_TIME) / TIME_STEP) + 1
    time_list = np.linspace(
        int((START_TIME - INITIAL_TIME) / TIME_STEP),
        int((END_TIME - INITIAL_TIME) / TIME_STEP),
        TIME_NUMBER,
        endpoint=True).astype(int)
    time_tmp = time_list * TIME_STEP
    time_index = np.random.choice(
        time_list, int(TIME_NUMBER / 2.5), replace=False)
    time_index.sort()
    time_array = time_index * TIME_STEP
    print(f"TIME_NUMBER = {TIME_NUMBER}")
    print(f"time_list = {time_list}")
    print(f"time_tmp = {time_tmp}")
    print(f"time_index = {time_index}")
    print(f"time_array = {time_array}")

    # initial value
    ic_name = dirname + r'/data/LBM_result/cylinder3d_2023_1_31_LBM_'
    txyz_uvwpe_ic = load_vtk(
        [0],
        t_step=TIME_STEP,
        load_uvwp=True,
        load_txyz=True,
        name_wt_time=ic_name)[0]
    init_t = txyz_uvwpe_ic[:, 0] / T_STAR
    init_x = txyz_uvwpe_ic[:, 1] / XYZ_STAR
    init_y = txyz_uvwpe_ic[:, 2] / XYZ_STAR
    init_z = txyz_uvwpe_ic[:, 3] / XYZ_STAR
    init_u = txyz_uvwpe_ic[:, 4] / UVW_STAR
    init_v = txyz_uvwpe_ic[:, 5] / UVW_STAR
    init_w = txyz_uvwpe_ic[:, 6] / UVW_STAR
    init_p = txyz_uvwpe_ic[:, 7] / p_star
    print_for_check(
        [init_t, init_x, init_y, init_z, init_u, init_v, init_w, init_p], [
            'init_t', 'init_x', 'init_y', 'init_z', 'init_u', 'init_v',
            'init_w', 'init_p'
        ])

    # supervised data
    sup_name = dirname + r"/data/sup_data/supervised_"
    txyz_uvwpe_s_new = load_vtk(
        time_index,
        t_step=TIME_STEP,
        load_uvwp=True,
        load_txyz=True,
        name_wt_time=sup_name)
    txyz_uvwpe_s = np.zeros((0, 8))
    for x in txyz_uvwpe_s_new:
        txyz_uvwpe_s = np.concatenate((txyz_uvwpe_s, x[:, :]), axis=0)
    sup_t = txyz_uvwpe_s[:, 0] / T_STAR
    sup_x = txyz_uvwpe_s[:, 1] / XYZ_STAR
    sup_y = txyz_uvwpe_s[:, 2] / XYZ_STAR
    sup_z = txyz_uvwpe_s[:, 3] / XYZ_STAR
    sup_u = txyz_uvwpe_s[:, 4] / UVW_STAR
    sup_v = txyz_uvwpe_s[:, 5] / UVW_STAR
    sup_w = txyz_uvwpe_s[:, 6] / UVW_STAR
    sup_p = txyz_uvwpe_s[:, 7] / p_star
    print_for_check([sup_t, sup_x, sup_y, sup_z, sup_u, sup_v, sup_w, sup_p], [
        'sup_t', 'sup_x', 'sup_y', 'sup_z', 'sup_u', 'sup_v', 'sup_w', 'sup_p'
    ])

    # num of interior points
    NUM_POINTS = 10000

    # discretize node by geo
    inlet_txyz, outlet_txyz, _, _, top_txyz, bottom_txyz, cylinder_txyz, interior_txyz = \
        sample_data.sample_data(
            t_num=TIME_NUMBER, t_index=time_index, t_step=TIME_STEP, nr_points=NUM_POINTS)

    # interior nodes discre
    i_t, i_x, i_y, i_z = normalized_bc(
        interior_txyz,
        t_factor=T_STAR,
        xyz_factor=XYZ_STAR,
        random_time=USE_RANDOM_TIME)
    print_for_check([i_t, i_x, i_y, i_z], ['i_t', 'i_x', 'i_y', 'i_z'])

    # bc inlet nodes discre
    b_inlet_t, b_inlet_x, b_inlet_y, b_inlet_z = normalized_bc(
        inlet_txyz,
        t_factor=T_STAR,
        xyz_factor=XYZ_STAR,
        random_time=USE_RANDOM_TIME)
    print_for_check([b_inlet_t, b_inlet_x, b_inlet_y, b_inlet_z],
                    ['b_inlet_t', 'b_inlet_x', 'b_inlet_y', 'b_inlet_z'])

    # bc outlet nodes discre
    b_outlet_t, b_outlet_x, b_outlet_y, b_outlet_z = normalized_bc(
        outlet_txyz,
        t_factor=T_STAR,
        xyz_factor=XYZ_STAR,
        random_time=USE_RANDOM_TIME)
    print_for_check([b_outlet_t, b_outlet_x, b_outlet_y, b_outlet_z],
                    ['b_outlet_t', 'b_outlet_x', 'b_outlet_y', 'b_outlet_z'])

    # bc cylinder nodes discre
    b_cylinder_t, b_cylinder_x, b_cylinder_y, b_cylinder_z = normalized_bc(
        cylinder_txyz,
        t_factor=T_STAR,
        xyz_factor=XYZ_STAR,
        random_time=USE_RANDOM_TIME)
    print_for_check(
        [b_cylinder_t, b_cylinder_x, b_cylinder_y, b_cylinder_z],
        ['b_cylinder_t', 'b_cylinder_x', 'b_cylinder_y', 'b_cylinder_z'])

    # bc-top nodes discre
    b_top_t, b_top_x, b_top_y, b_top_z = normalized_bc(
        top_txyz,
        t_factor=T_STAR,
        xyz_factor=XYZ_STAR,
        random_time=USE_RANDOM_TIME)
    print_for_check([b_top_t, b_top_x, b_top_y, b_top_z],
                    ['b_top_t', 'b_top_x', 'b_top_y', 'b_top_z'])

    # bc-bottom nodes discre
    b_bottom_t, b_bottom_x, b_bottom_y, b_bottom_z = normalized_bc(
        bottom_txyz,
        t_factor=T_STAR,
        xyz_factor=XYZ_STAR,
        random_time=USE_RANDOM_TIME)
    print_for_check([b_bottom_t, b_bottom_x, b_bottom_y, b_bottom_z],
                    ['b_bottom_t', 'b_bottom_t', 'b_bottom_t', 'b_bottom_t'])

    # bc & interior nodes for nn
    inputeq = np.stack((i_t, i_x, i_y, i_z), axis=1)
    inputbc1 = np.stack((b_inlet_t, b_inlet_x, b_inlet_y, b_inlet_z), axis=1)
    inputbc2 = np.stack(
        (b_outlet_t, b_outlet_x, b_outlet_y, b_outlet_z), axis=1)
    inputbc3 = np.stack(
        (b_cylinder_t, b_cylinder_x, b_cylinder_y, b_cylinder_z), axis=1)
    inputbc6_top = np.stack((b_top_t, b_top_x, b_top_y, b_top_z), axis=1)
    inputbc7_bottom = np.stack(
        (b_bottom_t, b_bottom_x, b_bottom_y, b_bottom_z), axis=1)

    inputic = np.stack((init_t, init_x, init_y, init_z), axis=1)
    inputsup = np.stack((sup_t, sup_x, sup_y, sup_z), axis=1)
    refsup = np.stack((sup_u, sup_v, sup_w), axis=1)

    # N-S, Re=3900, D=80, u=0.1, nu=80/3900; nu = rho u D / Re = 1.0 * 0.1 * 80 / 3900
    pde = psci.pde.NavierStokes(nu=NU, rho=1.0, dim=3, time_dependent=True)

    # set bounday condition
    bc_inlet_u = psci.bc.Dirichlet("u", rhs=0.1 / UVW_STAR)
    bc_inlet_v = psci.bc.Dirichlet("v", rhs=0.0)
    bc_inlet_w = psci.bc.Dirichlet("w", rhs=0.0)
    bc_cylinder_u = psci.bc.Dirichlet("u", rhs=0.0)
    bc_cylinder_v = psci.bc.Dirichlet("v", rhs=0.0)
    bc_cylinder_w = psci.bc.Dirichlet("w", rhs=0.0)
    bc_outlet_p = psci.bc.Dirichlet("p", rhs=0.0)
    bc_top_u = psci.bc.Dirichlet("u", rhs=0.1 / UVW_STAR)
    bc_top_v = psci.bc.Dirichlet("v", rhs=0.0)
    bc_top_w = psci.bc.Dirichlet("w", rhs=0.0)
    bc_bottom_u = psci.bc.Dirichlet("u", rhs=0.1 / UVW_STAR)
    bc_bottom_v = psci.bc.Dirichlet("v", rhs=0.0)
    bc_bottom_w = psci.bc.Dirichlet("w", rhs=0.0)

    # add bounday and boundary condition
    pde.set_bc("inlet", bc_inlet_u, bc_inlet_v, bc_inlet_w)
    pde.set_bc("cylinder", bc_cylinder_u, bc_cylinder_v, bc_cylinder_w)
    pde.set_bc("outlet", bc_outlet_p)
    pde.set_bc("top", bc_top_u, bc_top_v, bc_top_w)
    pde.set_bc("bottom", bc_bottom_u, bc_bottom_v, bc_bottom_w)

    # add initial condition
    ic_u = psci.ic.IC("u", rhs=init_u)
    ic_v = psci.ic.IC("v", rhs=init_v)
    ic_w = psci.ic.IC("w", rhs=init_w)
    ic_p = psci.ic.IC("p", rhs=init_p)
    pde.set_ic(ic_u, ic_v, ic_w)  # 添加ic_p会使ic_loss非常大

    # Network
    net = psci.network.FCNet(
        num_ins=4,
        num_outs=4,
        num_layers=6,
        hidden_size=hidden_size,
        activation="tanh")

    outeq = net(inputeq)
    outbc1 = net(inputbc1)
    outbc2 = net(inputbc2)
    outbc3 = net(inputbc3)
    outbc6 = net(inputbc6_top)
    outbc7 = net(inputbc7_bottom)
    outic = net(inputic)
    outsup = net(inputsup)

    # eq loss
    losseq1 = psci.loss.EqLoss(pde.equations[0], netout=outeq)
    losseq2 = psci.loss.EqLoss(pde.equations[1], netout=outeq)
    losseq3 = psci.loss.EqLoss(pde.equations[2], netout=outeq)
    losseq4 = psci.loss.EqLoss(pde.equations[3], netout=outeq)

    # bc loss
    lossbc1 = psci.loss.BcLoss("inlet", netout=outbc1)
    lossbc2 = psci.loss.BcLoss("outlet", netout=outbc2)
    lossbc3 = psci.loss.BcLoss("cylinder", netout=outbc3)
    lossbc6 = psci.loss.BcLoss("top", netout=outbc6)
    lossbc7 = psci.loss.BcLoss("bottom", netout=outbc7)

    # ic loss
    lossic = psci.loss.IcLoss(netout=outic)

    # supervise loss
    losssup = psci.loss.DataLoss(netout=outsup[0:3], ref=refsup)

    # total loss
    loss = losseq1 * eq_wgt + losseq2 * eq_wgt + losseq3 * eq_wgt + losseq4 * eq_wgt + \
        lossbc1 * inlet_wgt + \
        lossbc2 * outlet_wgt + \
        lossbc3 * cylinder_wgt + \
        lossbc6 * top_wgt + \
        lossbc7 * bottom_wgt + \
        lossic * ic_wgt + \
        losssup * sup_wgt

    # Algorithm
    algo = psci.algorithm.PINNs(net=net, loss=loss)
    _lr = psci.optimizer.lr.Cosine(
        num_epoch,
        1,
        learning_rate=learning_rate,
        warmup_epoch=w_epoch,
        by_epoch=True)()
    opt = psci.optimizer.Adam(learning_rate=_lr, parameters=net.parameters())

    # Read validation reference for time step : 0, 99
    lbm_01 = load_vtk(
        [0],
        t_step=TIME_STEP,
        load_uvwp=True,
        load_txyz=True,
        name_wt_time=ic_name)[0]
    lbm_99 = load_vtk(
        [99],
        t_step=TIME_STEP,
        load_uvwp=True,
        load_txyz=True,
        name_wt_time=ic_name)[0]
    tmp = txyz_nomalization(T_STAR, XYZ_STAR, lbm_01)
    lbm_01[:, 0:4] = np.stack(tmp, axis=1)
    tmp = txyz_nomalization(T_STAR, XYZ_STAR, lbm_99)
    lbm_99[:, 0:4] = np.stack(tmp, axis=1)
    lbm = [lbm_01, lbm_99]

    # Solver
    solver = psci.solver.Solver(pde=pde, algo=algo, opt=opt, lbm=lbm)

    # Solve
    solution = solver.solve(num_epoch=num_epoch, bs=bs)
