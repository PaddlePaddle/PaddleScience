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

import os
import paddlescience as psci
import numpy as np
import time

import paddle
from paddle.incubate.autograd import Hessian
from transform import program_transform
from transform import dead_code_elimination
from transform import fuse_shape_fill_constant

paddle.enable_static()
paddle.seed(1234)
np.random.seed(1234)


def compile(program, loss_name=None):
    build_strategy = paddle.static.BuildStrategy()
    exec_strategy = paddle.static.ExecutionStrategy()

    exec_strategy.num_threads = 1

    compiled_program = paddle.static.CompiledProgram(
        program).with_data_parallel(
            loss_name=loss_name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

    return compiled_program


# Analytical solution
def LaplaceRecSolution(x, y, k=1.0):
    if (k == 0.0):
        return x * y
    else:
        return np.cos(k * x) * np.cosh(k * y)


# Generate analytical Solution using Geometry points
def GenSolution(xy, bc_index):
    sol = np.zeros((len(xy), 1)).astype(np.float32)
    bc_value = np.zeros((len(bc_index), 1)).astype(np.float32)
    for i in range(len(xy)):
        sol[i] = LaplaceRecSolution(xy[i][0], xy[i][1])
    for i in range(len(bc_index)):
        bc_value[i][0] = sol[bc_index[i]]
    return [sol, bc_value]


# Geometry
geo = psci.geometry.Rectangular(
    space_origin=(0.0, 0.0), space_extent=(1.0, 1.0))

# PDE Laplace
pdes = psci.pde.Laplace2D()

# Discretization
pdes, geo = psci.discretize(pdes, geo, space_nsteps=(11, 11))

# bc value
golden, bc_value = GenSolution(geo.get_space_domain(), geo.get_bc_index())
pdes.set_bc_value(bc_value=bc_value)

psci.visu.save_vtk(geo, golden, 'golden_laplace_2d')
np.save('./golden_laplace_2d.npy', golden)

place = paddle.CUDAPlace(0)
exe = paddle.static.Executor(place)

train_program = paddle.static.Program()
startup_program = paddle.static.Program()
with paddle.static.program_guard(train_program, startup_program):
    inputs = paddle.static.data(
        name='x', shape=[geo.get_domain_size(), 2], dtype='float32')
    inputs.stop_gradient = False
    # Network
    net = psci.network.FCNetStatic(
        num_ins=2,
        num_outs=1,
        num_layers=5,
        hidden_size=20,
        dtype='float32',
        activation='tanh')

    outputs = net.nn_func(inputs)

    # eq_loss
    hes = Hessian(net.nn_func, inputs, is_batched=True)
    eq_loss = paddle.norm(hes[:, 0, 0] + hes[:, 1, 1], p=2)

    # bc_loss
    bc_index = paddle.static.data(name='bc_idx', shape=[40], dtype='int32')
    bc_value = paddle.static.data(name='bc_v', shape=[40, 1], dtype='float32')
    bc_u = paddle.index_select(outputs, bc_index)
    bc_diff = bc_u - bc_value
    bc_loss = paddle.norm(bc_diff, p=2)
    loss = eq_loss + bc_loss
    paddle.optimizer.Adam(learning_rate=0.001).minimize(loss)

new_program = program_transform(train_program)
dead_code_elimination(new_program)
fuse_shape_fill_constant(new_program)
# print('startup_program: ', startup_program)
# print('train_program: ', train_program)
# print('new_program: ', new_program)

exe.run(startup_program)
num_epoch = 2010

compiled_program = compile(new_program, loss.name)
train_program = compile(train_program, loss.name)

begin = time.time()

if os.getenv('FLAGS_use_cinn') == "1":
    for i in range(num_epoch):
        if i == 10:
            paddle.device.cuda.synchronize()
            begin = time.time()
            print("begin With CINN at ", begin)

        loss_d = exe.run(compiled_program,
                         feed={
                             'x': geo.get_space_domain().astype(np.float32),
                             'bc_idx': geo.bc_index.astype(np.int32),
                             'bc_v': pdes.bc_value
                         },
                         fetch_list=[loss.name])
        print('num_epoch: ', i, '/', num_epoch, ' loss: ', loss_d[0][0])

    end = time.time()
    print("[With CINN] 2000 epoch(10~2010) time: ", end - begin, " s")
else:
    for i in range(num_epoch):
        if i == 10:
            paddle.device.cuda.synchronize()
            begin = time.time()
            print("begin Without CINN at ", begin)

        loss_d, eq_loss_d, bc_loss_d = exe.run(
            train_program,
            feed={
                'x': geo.get_space_domain().astype(np.float32),
                'bc_idx': geo.bc_index.astype(np.int32),
                'bc_v': pdes.bc_value
            },
            fetch_list=[loss.name, eq_loss.name, bc_loss.name])
        print('num_epoch: ', i, '/', num_epoch, ' loss: ', loss_d[0])

    end = time.time()
    print("[Without CINN] 2000 epoch(10~2010) time: ", end - begin, " s")

rslt = exe.run(train_program,
               feed={
                   'x': geo.get_space_domain().astype(np.float32),
                   'bc_idx': geo.bc_index.astype(np.int32),
                   'bc_v': pdes.bc_value
               },
               fetch_list=[outputs.name, ])[0]
psci.visu.save_vtk(geo, rslt, 'rslt_laplace_2d')
np.save('./rslt_laplace_2d.npy', rslt)

# Calculate diff and l2 relative error
diff = rslt - golden
psci.visu.save_vtk(geo, diff, 'diff_laplace_2d')
np.save('./diff_laplace_2d.npy', diff)
root_square_error = np.linalg.norm(diff, ord=2)
mean_square_error = root_square_error * root_square_error / geo.get_domain_size(
)
print('mean_sqeare_error: ', mean_square_error)
