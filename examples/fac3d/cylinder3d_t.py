# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddlescience as psci
import paddle
import numpy as np


# Generate BC value
# Every row have 4 elements which means u,v,w,p of one point in one moment
def GenBC(xyz, bc_index):
    bc_value = np.zeros((len(bc_index), 4)).astype(np.float32)
    for i in range(len(bc_index)):
        id = bc_index[i]
        # if the point is in entrance
        if (abs(xyz[id][0] - (-8)) < 1e-4):
            bc_value[i][0] = 10.0
            bc_value[i][1] = 0.0
            bc_value[i][2] = 0.0
        # if the point is in exit
        elif (abs(xyz[id][0] - 25) < 1e-4):
            #pass
            bc_value[i][0] = 0.0
            bc_value[i][1] = 0.0
            bc_value[i][2] = 0.0
            # the pressure need to be 0
            bc_value[i][3] = 0.0
        # if the point is in top
        elif (abs(xyz[id][1] - (-8)) < 1e-4):
            pass
        # if the point is in down
        elif (abs(xyz[id][1] - 8) < 1e-4):
            pass
        # if the point is in cycle around
        else:
            pass
    return bc_value


# Generate BC weight
def GenBCWeight(xyz, bc_index):
    bc_weight = np.zeros((len(bc_index), 4)).astype(np.float32)
    for i in range(len(bc_index)):
        id = bc_index[i]
        # if the point is in entrance
        if (abs(xyz[id][0] - (-8)) < 1e-4):
            #bc_weight[i][0] = 1.0 - 0.1 * abs(xyz[id][1])
            bc_weight[i][0] = 1.0
            bc_weight[i][1] = 1.0
            bc_weight[i][2] = 1.0
        # if the point is in exit
        elif (abs(xyz[id][0] - 25) < 1e-4):
            bc_weight[i][3] = 1.0
        # if the point is in south
        elif (abs(xyz[id][1] - (-8)) < 1e-4):
            pass
        # if the point is in north
        elif (abs(xyz[id][1] - 8) < 1e-4):
            pass
        # if the point is in up
        elif (abs(xyz[id][2] - (-0.5)) < 1e-4):
            pass
        # if the point is in down
        elif (abs(xyz[id][2] - 0.5) < 1e-4):
            pass
        else:
            bc_weight[i][0] = 1.0
            bc_weight[i][1] = 1.0
            bc_weight[i][2] = 1.0
    #bc_weight = 10 * bc_weight
    return bc_weight


# Generate BC weight
def GenEqWeight(xyz):
    eq_weight = np.ones((len(xyz), 4)).astype(np.float32)
    #eq_weight = 1 * eq_weight
    for i in range(len(xyz)):
        eq_weight[i][0] = 2.0
        eq_weight[i][1] = 0.1
        eq_weight[i][2] = 0.1
        eq_weight[i][3] = 0.1
    return eq_weight


def GenInitPhyInfo(xyz):
    uvwp = np.zeros((len(xyz), 4)).astype(np.float32)
    for i in range(len(xyz)):
        if abs(xyz[i][0] - (-8)) < 1e-4:
            uvwp[i][0] = 10.0
    return uvwp


def SaveCsvFile(geo, rslt, filename):
    space_domain = geo.space_domain
    train_rslt = np.concatenate((space_domain.numpy(), rslt), axis=1)
    print("the %s is: x y z u v w p" % filename)
    np.savetxt("csv/" + filename, train_rslt, fmt='%.6f', delimiter=',')


def SaveVtkFile(geo, rslt, filename):
    u = rslt[:, 0]
    v = rslt[:, 1]
    w = rslt[:, 2]
    p = rslt[:, 3]
    # output the result
    rslt_dictionary = {'u': u, 'v': v, 'w': w, 'p': p}
    psci.visu.save_vtk_points(filename=filename, geo=geo, data=rslt_dictionary)


def GetRealPhyInfo(time):
    xyzuvwp = np.load("flow_re200/flow_re200_" + format(time, '.2f') +
                      "_xyzuvwp.npy")
    print(xyzuvwp.shape)
    return xyzuvwp


if __name__ == "__main__":
    # Geometry
    geo = psci.geometry.CylinderInRectangular(
        space_origin=(-8, -8, -0.5),
        space_extent=(25, 8, 0.5),
        circle_center=(0, 0),
        circle_radius=0.5)

    # PDE Laplace
    pdes = psci.pde.NavierStokes(
        nu=0.05, rho=1.0, dim=3, time_integration=True, dt=0.25)

    # Get real data
    xyzuvwp = GetRealPhyInfo(0.25)

    # Discretization
    geo = psci.geometry.CylinderInRectangular.sampling_discretize(
        geo,
        space_npoints=40000,
        space_nsteps=(67, 81, 3),
        circle_bc_size=1000,
        real_data=xyzuvwp)

    # bc value
    bc_value = GenBC(geo.get_space_domain(), geo.get_bc_index())
    pdes.set_bc_value(bc_value=bc_value, bc_check_dim=[0, 1, 2, 3])

    # uv value of t0 time
    uvwp = GenInitPhyInfo(geo.get_space_domain())

    # Network
    net = psci.network.FCNet(
        num_ins=3,
        num_outs=4,
        num_layers=10,
        hidden_size=50,
        dtype="float32",
        activation='tanh')

    # Loss, TO rename 
    bc_weight = GenBCWeight(geo.space_domain, geo.bc_index)
    eq_weight = GenEqWeight(geo.space_domain)
    loss = psci.loss.L2(pdes=pdes,
                        geo=geo,
                        physic_info=uvwp,
                        real_data_value=xyzuvwp,
                        real_data_loss_weight=1.0,
                        eq_weight=eq_weight,
                        bc_weight=bc_weight,
                        synthesis_method='norm')

    # Algorithm
    algo = psci.algorithm.PINNs(net=net, loss=loss)

    # Optimizer
    opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

    use_saved_model = False
    if use_saved_model is True:
        # load model
        layer_state_dict = paddle.load("checkpoint/net_params_1000")
        opt_state_dict = paddle.load("checkpoint/opt_params_1000")

        algo.net.set_state_dict(layer_state_dict)
        opt.set_state_dict(opt_state_dict)

    # Solver train t0 -> t1
    solver = psci.solver.Solver(algo=algo, opt=opt)
    solution = solver.solve(num_epoch=2000, save_model_file_path=".")
    rslt_t1 = solution(geo, physic_info=uvwp)  #
    #SaveCsvFile(geo, rslt_t1, "fac3d_train_rslt_t10.0.csv")
    SaveVtkFile(geo, rslt_t1, "train_flowRe200/fac3d_train_rslt_t10s_0.25")

    # Solver train t1 -> tn
    time_step = 39
    current_physic_info = rslt_t1
    for i in range(time_step):
        current_time = 0.25 + (i + 1) * 0.25
        print("###################### start time=%f train task ############" %
              current_time)
        # modify the loss
        current_real_data = GetRealPhyInfo(current_time)
        algo.loss.real_data_value = current_real_data  #None
        algo.loss.real_data_loss_weight = 1.0
        algo.loss.physic_info = paddle.to_tensor(
            current_physic_info, dtype="float32")
        # solve
        solution = solver.solve(num_epoch=2000, first_train=False)
        # Get the train result
        rslt = solution(geo, physic_info=current_physic_info)
        # save the result 
        #SaveCsvFile(geo, rslt, "train_rslt_t"+str(current_time)+".csv")
        SaveVtkFile(geo, rslt, "train_flowRe200/fac3d_train_rslt_t10s_" +
                    format(current_time, '.2f'))
        current_physic_info = rslt[:, 0:3]
    '''
    # Use solution for inference with another geometry
    predict_geo = psci.geometry.CylinderInRectangular(
        space_origin=(-8, -8, -0.5),
        space_extent=(25, 8, 0.5),
        circle_center=(0, 0),
        circle_radius=2)
    predict_geo = psci.geometry.CylinderInRectangular.sampling_discretize(
        predict_geo, space_npoints=60000, space_nsteps=(166, 81, 3), circle_bc_size=1000)
    uvwp = GenInitPhyInfo(predict_geo.get_space_domain())
    time_nsteps = 10
    current_physic_info = uvwp
    for i in range(time_nsteps):
        current_time = (i+1)*0.5
        print("########## inference for time=%f ######" % current_time)
        rslt = solution(predict_geo, physic_info=current_physic_info)
        # save the result 
        SaveCsvFile(predict_geo, rslt, "predict_rslt_t"+str(current_time)+".csv")
        SaveVtkFile(predict_geo, rslt, "fac3d_predict_rslt_t"+str(current_time))
        current_physic_info = rslt[:,0:3]
    '''
