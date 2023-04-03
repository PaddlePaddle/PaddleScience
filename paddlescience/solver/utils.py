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

import time

import numpy as np
import paddle

# auto parallel
import paddle.distributed.auto_parallel as auto  # isort:skip
import six  # isort:skip
from paddle import fluid  # isort:skip
from paddle.distributed.auto_parallel import dist_context as dct_module  # isort:skip
from paddle.distributed.auto_parallel.completion import Completer  # isort:skip
from paddle.distributed.auto_parallel.partitioner import Partitioner  # isort:skip
from paddle.distributed.auto_parallel.utils import set_var_dist_attr  # isort:skip
from paddle.fluid import core  # isort:skip
from paddle.fluid.framework import Variable  # isort:skip
from paddle.static import default_main_program  # isort:skip
from paddle.static import global_scope  # isort:skip

nranks = paddle.distributed.get_world_size()
rank = paddle.distributed.get_rank()


def create_inputs_var(inputs):
    inputs_var = []
    for i in range(len(inputs)):

        # data parallel partition
        shape = list(inputs[i].shape)
        if nranks > 1:
            gbsz = inputs[i].shape[0]
            lbsz = gbsz // nranks
            # uneven data partition, last rank would contain more data
            if rank == nranks - 1:
                lbsz += gbsz % nranks
            shape[0] = lbsz

        input = paddle.static.data(name="input" + str(i), shape=shape, dtype="float32")
        input.stop_gradient = False
        inputs_var.append(input)
    return inputs_var


def create_labels_var(labels, npoints, data_size):
    labels_var = []
    for i in range(len(labels)):
        if i in [0, 1, 2]:
            shape = (npoints,)
        else:
            shape = (data_size,)

        # data parallel partition
        if nranks > 1:
            gbsz = shape[0]
            lbsz = gbsz // nranks
            if rank == nranks - 1:
                lbsz += gbsz % nranks
            shape = (lbsz,)

        label = paddle.static.data(name="label" + str(i), shape=shape, dtype="float32")
        label.stop_gradient = False
        labels_var.append(label)
    return labels_var


def data_parallel_partition(data, time_step=0):
    if nranks <= 1:
        return data

    for i in range(len(data)):
        # first & last 3 labels are output from last time step and are already partitioned
        if time_step > 0 and i in [0, 1, 2, 7, 8, 9]:
            continue

        gbsz = data[i].shape[0]
        lbsz = gbsz // nranks
        start = rank * lbsz
        end = (rank + 1) * lbsz
        if rank == nranks - 1:
            end += gbsz % nranks
        data[i] = data[i][start:end]

    return data


def l2_norm_square(x, scale=None):
    if scale is None:
        l2_norm = paddle.norm(x, p=2)
    else:
        l2_norm = paddle.norm(x * scale, p=2) / scale
    return l2_norm * l2_norm


def compute_bc_loss(inputs_attr, labels_attr, outputs_var, pde_disc):
    name2index = {"u": 0, "v": 1, "w": 2, "p": 3}
    bc_loss = 0.0
    name_list = []
    for i, name_b in enumerate(inputs_attr["bc"].keys()):
        # from outputs_var[1] to outputs_var[3]
        out_el = outputs_var[i + 1]
        for j in range(len(pde_disc.bc[name_b])):
            rhs_b = labels_attr["bc"][name_b][j]["rhs"]
            wgt_b = labels_attr["bc"][name_b][j]["weight"]
            index = name2index.get(pde_disc.bc[name_b][j].name)

            bc_loss += l2_norm_square(
                (out_el[:, index] - rhs_b) * np.sqrt(wgt_b), 10000
            )
    return bc_loss


def compute_eq_loss(inputs, outputs, labels_var):
    x = inputs[:, 0]
    y = inputs[:, 1]
    z = inputs[:, 2]
    u = outputs[:, 0]
    v = outputs[:, 1]
    w = outputs[:, 2]
    p = outputs[:, 3]
    u_n = labels_var[0]
    v_n = labels_var[1]
    w_n = labels_var[2]
    (jac0,) = paddle.incubate.autograd.grad([u], [inputs])  # du/dx, du/dy, du/dz
    (jac1,) = paddle.incubate.autograd.grad([v], [inputs])  # dv/dx, dv/dy, dv/dz
    (jac2,) = paddle.incubate.autograd.grad([w], [inputs])  # dw/dx, dw/dy, dw/dz
    (jac3,) = paddle.incubate.autograd.grad([p], [inputs])  # dp/dx, dp/dy, dp/dz
    (hes0,) = paddle.incubate.autograd.grad(
        [jac0[:, 0]], [inputs]
    )  # du*du/dx*dx, du*du/dx*dy, du*du/dx*dz
    (hes1,) = paddle.incubate.autograd.grad(
        [jac0[:, 1]], [inputs]
    )  # du*du/dy*dx, du*du/dy*dy, du*du/dy*dz
    (hes2,) = paddle.incubate.autograd.grad(
        [jac0[:, 2]], [inputs]
    )  # du*du/dz*dx, du*du/dz*dy, du*du/dz*dz
    (hes3,) = paddle.incubate.autograd.grad(
        [jac1[:, 0]], [inputs]
    )  # dv*dv/dx*dx, dv*dv/dx*dy, dv*dv/dx*dz
    (hes4,) = paddle.incubate.autograd.grad(
        [jac1[:, 1]], [inputs]
    )  # dv*dv/dy*dx, dv*dv/dy*dy, dv*dv/dy*dz
    (hes5,) = paddle.incubate.autograd.grad(
        [jac1[:, 2]], [inputs]
    )  # dv*dv/dz*dx, dv*dv/dz*dy, dv*dv/dz*dz
    (hes6,) = paddle.incubate.autograd.grad(
        [jac2[:, 0]], [inputs]
    )  # dw*dw/dx*dx, dw*dw/dx*dy, dw*dw/dx*dz
    (hes7,) = paddle.incubate.autograd.grad(
        [jac2[:, 1]], [inputs]
    )  # dw*dw/dy*dx, dw*dw/dy*dy, dw*dw/dy*dz
    (hes8,) = paddle.incubate.autograd.grad(
        [jac2[:, 2]], [inputs]
    )  # dw*dw/dz*dx, dw*dw/dz*dy, dw*dw/dz*dz

    nu = 0.01
    rho = 1.0
    dt = 1.0
    continuty = jac0[:, 0] + jac1[:, 1] + jac2[:, 2]
    momentum_x = (
        u / dt
        - u_n / dt
        + u * jac0[:, 0]
        + v * jac0[:, 1]
        + w * jac0[:, 2]
        - nu / rho * hes0[:, 0]
        - nu / rho * hes1[:, 1]
        - nu / rho * hes2[:, 2]
        + 1.0 / rho * jac3[:, 0]
    )
    momentum_y = (
        v / dt
        - v_n / dt
        + u * jac1[:, 0]
        + v * jac1[:, 1]
        + w * jac1[:, 2]
        - nu / rho * hes3[:, 0]
        - nu / rho * hes4[:, 1]
        - nu / rho * hes5[:, 2]
        + 1.0 / rho * jac3[:, 1]
    )
    momentum_z = (
        w / dt
        - w_n / dt
        + u * jac2[:, 0]
        + v * jac2[:, 1]
        + w * jac2[:, 2]
        - nu / rho * hes6[:, 0]
        - nu / rho * hes7[:, 1]
        - nu / rho * hes8[:, 2]
        + 1.0 / rho * jac3[:, 2]
    )

    rhs = 0
    wgt = np.sqrt(0.01)

    eq_loss = (
        l2_norm_square((continuty - rhs) * wgt)
        + l2_norm_square((momentum_x - rhs) * wgt)
        + l2_norm_square((momentum_y - rhs) * wgt)
        + l2_norm_square((momentum_z - rhs) * wgt)
    )
    return eq_loss


# Convert the program into graph, apply the calculation graph optimizations, and turn back to the program
def compile_and_convert_back_to_program(
    program=None,
    feed=None,
    fetch_list=None,
    fetch_var_name="fetch",
    scope=None,
    use_prune=False,
    loss_name=None,
):
    def _add_fetch_ops(program, fetch_list, fetch_var_name):
        assert isinstance(program, fluid.Program)
        tmp_program = program.clone()
        global_block = tmp_program.global_block()

        if fetch_var_name in global_block.vars:
            fetch_var = global_block.var(fetch_var_name)
        else:
            fetch_var = global_block.create_var(
                name=fetch_var_name,
                type=core.VarDesc.VarType.FETCH_LIST,
                persistable=True,
            )

        # append fetch_operators
        if not fluid.executor.has_fetch_operators(
            global_block, fetch_list, fetch_var_name, "fetch"
        ):
            for i, var in enumerate(fetch_list):
                assert isinstance(var, Variable) or isinstance(
                    var, six.string_types
                ), "Wrong type for fetch_list[%s]: %s" % (i, type(var))
                global_block.append_op(
                    type="fetch",
                    inputs={"X": [var]},
                    outputs={"Out": [fetch_var]},
                    attrs={"col": i},
                )
        return tmp_program

    def _remove_fetch_ops(program):
        assert isinstance(program, fluid.Program)
        tmp_program = program.clone()
        global_block = tmp_program.global_block()
        op_num = len(global_block.ops)
        for idx in reversed(range(op_num)):
            if global_block.ops[idx].type == "fetch":
                global_block._remove_op(idx)

        return tmp_program

    def _compile(program, loss_name=None):
        build_strategy = paddle.static.BuildStrategy()

        compiled_program = paddle.static.CompiledProgram(
            program, build_strategy=build_strategy
        )

        return compiled_program

    if program is None:
        program = default_main_program()

    if scope is None:
        scope = global_scope()

    executor = paddle.static.Executor()

    fetch_list = executor._check_fetch_list(fetch_list)
    fetch_list, optimize_ops = executor._split_optimize_ops_in_fetch_list(fetch_list)

    if optimize_ops:
        raise ValueError("Unsupport to fetch optimize OP.")

    if use_prune:
        program = executor._prune_program(program, feed, fetch_list, optimize_ops)
        feed = executor._update_feed(program, feed)

    program_with_fetch_op = _add_fetch_ops(program, fetch_list, fetch_var_name)
    compiled_program = _compile(program_with_fetch_op, loss_name)
    assert isinstance(compiled_program, fluid.compiler.CompiledProgram)

    compiled_program._compile(scope, paddle.framework._current_expected_place())
    compiled_graph = compiled_program._graph
    ir_graph = fluid.framework.IrGraph(compiled_graph, for_test=True)
    ir_program = ir_graph.to_program()
    final_program = _remove_fetch_ops(ir_program)

    return final_program


def set_init_dist_attr(serial_main_prog):

    # set init dp attr
    default_dist_context = dct_module.get_default_distributed_context()
    _global_parallel_strategy = "dp"
    _global_process_mesh = auto.ProcessMesh(list(range(nranks)))
    x_tensor = serial_main_prog.global_block().var("input0")
    bc_idx_tensor = serial_main_prog.global_block().var("label0")
    tensor_dist_attr = set_var_dist_attr(
        default_dist_context,
        x_tensor,
        [-1, -1],
        _global_process_mesh,
        mark_annotated=True,
    )
    tensor_dist_attr = set_var_dist_attr(
        default_dist_context,
        bc_idx_tensor,
        [-1],
        _global_process_mesh,
        mark_annotated=True,
    )


def init_comm():
    from paddle.distributed.auto_parallel import process_group

    all_process_groups = process_group.get_all_process_groups()
    for process_group in all_process_groups:
        if rank not in process_group.ranks:
            continue
        process_group.instantiate()


def convert_to_distributed_program(serial_main_prog, serial_startup_prog, params_grads):
    set_init_dist_attr(serial_main_prog)
    dist_context = dct_module.DistributedContext(serial_main_prog, serial_startup_prog)

    # forward completion
    completer = Completer(dist_context)
    completer.complete_prim_annotation(serial_main_prog)
    dist_context.set_default_distributed_context(dist_context)
    dist_context.block_state.parse_forward_blocks(serial_main_prog)

    # backward
    dist_context.block_state.parse_backward_blocks(serial_main_prog)
    dist_context.grads_params = dict()
    for p, g in params_grads:
        dist_context.grads_params[g.name] = p.name
    dist_context.synced_gradient = set()
    dist_context.data_parallel_group = list(range(nranks))

    # parititoner
    partitioner = Partitioner(dist_context, rank)
    dist_main_prog, dist_startup_prog, dist_params_grads = partitioner.partition(
        serial_main_prog, serial_startup_prog, params_grads
    )
    assert set(dist_context.grads_params.keys()) == dist_context.synced_gradient

    init_comm()
    return dist_main_prog, dist_startup_prog


def cinn_compile(origin_program, loss_name, fetch_list):
    def cinn_optimize_program(input_program, fetch_list):
        def _remove_unused_var(program):
            all_remove_vars = []
            for block in program.blocks:
                args = []
                for op in block.ops:
                    args += op.input_arg_names
                    args += op.output_arg_names
                args = list(set(args))  # vals of all left ops
                var_names = block.vars.keys()  # all vals
                sub_block_remove_vars = []
                for var in var_names:
                    if var not in args:
                        sub_block_remove_vars.append(var)
                all_remove_vars.append(sub_block_remove_vars)

            remove_vars = [list(set(v)) for v in all_remove_vars]
            for i, block in enumerate(program.blocks):
                for v in remove_vars[i]:
                    block._remove_var(v)

        def dead_code_elimination(program, fetch_list):
            program._sync_with_cpp()
            in2ops = dict()
            out2op = dict()
            for block in program.blocks:
                ops = list(block.ops)
                for op in ops:
                    for name in op.input_arg_names:
                        if name not in in2ops:
                            in2ops[name] = 0
                        in2ops[name] += 1

                    for name in op.output_arg_names:
                        out2op[name] = op

            remove_ops = set()
            remove_names = set()
            while True:
                all_removed = True
                for name in remove_names:
                    out2op.pop(name)
                remove_names.clear()

                for name, op in out2op.items():
                    if (name not in in2ops) and (name not in fetch_list):
                        all_removed = False

                        remove_ops.add(op)
                        remove_names.add(name)

                        for in_name in op.input_arg_names:
                            in2ops[in_name] -= 1
                            if in2ops[in_name] == 0:
                                in2ops.pop(in_name)

                if all_removed:
                    break

            for block in program.blocks:
                ops = list(block.ops)
                for op in ops:
                    if op in remove_ops:
                        idx = block.ops.index(op)
                        block._remove_op(idx)

            _remove_unused_var(program)
            program._sync_with_cpp()

        def fuse_shape_fill_constant(program):
            def _insert_fill_any_like_op(block, index, shape_op, fill_constant_op):
                fill_any_like_inputs = {}
                fill_any_like_inputs["X"] = block.var(shape_op.input("Input")[0])
                fill_any_like_outputs = {}
                fill_any_like_outputs["Out"] = block.var(
                    fill_constant_op.output("Out")[0]
                )
                fill_any_like_attrs = {}
                fill_any_like_attrs["value"] = fill_constant_op.attr("value")
                fill_any_like_attrs["dtype"] = fill_constant_op.attr("dtype")
                fill_any_like_attrs["op_role"] = fill_constant_op.attr("op_role")

                fill_any_like_op = block._insert_op(
                    index,
                    type="fill_any_like",
                    inputs=fill_any_like_inputs,
                    outputs=fill_any_like_outputs,
                    attrs=fill_any_like_attrs,
                )
                return fill_any_like_op

            program._sync_with_cpp()
            block = program.block(0)
            i = 0
            while i < len(block.ops):
                # find a fill_constant op
                if block.ops[i].type == "fill_constant":
                    fill_constant_op = block.ops[i]
                    fill_constant_idx = i
                    shape_idx = -1
                    # find the preceding shape op
                    for j in reversed(range(fill_constant_idx)):
                        if block.ops[j].type == "shape":
                            shape_out_name = block.ops[j].output_arg_names[0]
                            if shape_out_name in fill_constant_op.input_arg_names:
                                shape_op = block.ops[j]
                                shape_idx = j
                                break
                    if shape_idx < 0:
                        i += 1
                        continue
                    # create and insert a new fill_any_like op
                    _insert_fill_any_like_op(
                        block, fill_constant_idx + 1, shape_op, fill_constant_op
                    )
                    # remove the old operators
                    block._remove_op(fill_constant_idx)
                    block._remove_op(shape_idx)
                    # restart scanning for elementwise add from the deleted shape's index
                    i = shape_idx
                i += 1
            _remove_unused_var(program)
            program._sync_with_cpp()

        tmp_program = input_program.clone()
        dead_code_elimination(tmp_program, fetch_list)
        fuse_shape_fill_constant(tmp_program)
        return tmp_program

    def _add_fetch_ops(program, fetch_list, fetch_var_name="fetch"):
        assert isinstance(program, fluid.Program)
        tmp_program = program.clone()
        global_block = tmp_program.global_block()

        if fetch_var_name in global_block.vars:
            fetch_var = global_block.var(fetch_var_name)
        else:
            fetch_var = global_block.create_var(
                name=fetch_var_name,
                type=core.VarDesc.VarType.FETCH_LIST,
                persistable=True,
            )

        # append fetch_operators
        if not fluid.executor.has_fetch_operators(
            global_block, fetch_list, fetch_var_name, "fetch"
        ):
            for i, var in enumerate(fetch_list):
                assert isinstance(var, Variable) or isinstance(
                    var, six.string_types
                ), "Wrong type for fetch_list[%s]: %s" % (i, type(var))
                global_block.append_op(
                    type="fetch",
                    inputs={"X": [var]},
                    outputs={"Out": [fetch_var]},
                    attrs={"col": i},
                )
        return tmp_program

    def _compile(program, loss_name):
        build_strategy = paddle.static.BuildStrategy()
        exec_strategy = paddle.static.ExecutionStrategy()

        exec_strategy.num_threads = 1

        compiled_program = paddle.static.CompiledProgram(
            program, build_strategy=build_strategy
        )

        return compiled_program

    optimized_program = cinn_optimize_program(origin_program, fetch_list)
    program_with_fetch = _add_fetch_ops(optimized_program, fetch_list)

    return _compile(program_with_fetch, loss_name)


class Timer:
    def __init__(self):
        self.tstart = time.time()
        self.tend = 0.0

    def start(self):
        self.tstart = time.time()

    def end(self):
        self.tend = time.time()

    def print(self):
        elap = self.tend - self.tstart
        print("Computation time is (sencond): ", elap)
