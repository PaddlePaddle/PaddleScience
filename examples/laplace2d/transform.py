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

import paddle
import six
from paddle import compat as cpt
from paddle.fluid.framework import Block, Operator
from paddle.fluid.backward import _create_op_desc_
import paddle.fluid.core as core

paddle.enable_static()


class NameGen(object):
    def __init__(self):
        self.cnt = 0

    def get_name(self):
        name = 'name_gen_' + str(self.cnt)
        self.cnt = self.cnt + 1
        return name

    def get_var(self, block=None, ref_var=None, shape=None):
        name = self.get_name()
        new_shape = ref_var.shape if shape is None else shape
        block.create_var(
            name=name,
            shape=new_shape,
            dtype=ref_var.dtype,
            type=ref_var.type,
            persistable=False,
            stop_gradient=False)
        return name


name_gen = NameGen()


def transhape(ref_shape):
    assert len(ref_shape) == 2
    return [ref_shape[1], ref_shape[0]]


def slice_assign_shape(old_shape, decrease_axis):
    new_shape = []
    j = 0
    for i in range(len(old_shape) + len(decrease_axis)):
        if i in decrease_axis:
            new_shape.append(1)
        else:
            new_shape.append(old_shape[j])
            j += 1
    return new_shape


def _remove_unused_var(program):
    all_remove_vars = []
    for block in program.blocks:
        args = []
        for op in block.ops:
            args += op.input_arg_names
            args += op.output_arg_names
        args = list(set(args))  #vals of all left ops
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


def dead_code_elimination(program):
    program._sync_with_cpp()
    all_input_arg_names = set()
    for block in program.blocks:
        ops = list(block.ops)
        for op in ops:
            for name in op.input_arg_names:
                all_input_arg_names.add(name)

    for block in program.blocks:
        ops = list(block.ops)
        for op in ops:
            if op.type == "fill_constant_p" and (
                    op.output('Y')[0] not in all_input_arg_names):
                idx = block.ops.index(op)
                block._remove_op(idx)

    _remove_unused_var(program)
    program._sync_with_cpp()


def _adjust_input(block, input_map):
    for i in range(len(block.ops)):
        current_op = block.ops[i]
        for input_arg in current_op.input_arg_names:
            if input_arg in input_map:
                current_op._rename_input(input_arg, input_map[input_arg])


def _insert_fill_any_like_op(block, index, shape_op, fill_constant_op):
    fill_any_like_inputs = {}
    fill_any_like_inputs['X'] = block.var(shape_op.input('Input')[0])
    fill_any_like_outputs = {}
    fill_any_like_outputs['Out'] = block.var(fill_constant_op.output('Out')[0])
    fill_any_like_attrs = {}
    fill_any_like_attrs['value'] = fill_constant_op.attr('value')
    fill_any_like_attrs['dtype'] = fill_constant_op.attr('dtype')
    fill_any_like_attrs['op_role'] = fill_constant_op.attr('op_role')

    fill_any_like_op = block._insert_op(
        index,
        type='fill_any_like',
        inputs=fill_any_like_inputs,
        outputs=fill_any_like_outputs,
        attrs=fill_any_like_attrs)
    return fill_any_like_op


def fuse_shape_fill_constant(program):
    program._sync_with_cpp()
    block = program.block(0)
    i = 0
    while i < len(block.ops):
        # find a fill_constant op
        if block.ops[i].type == 'fill_constant':
            fill_constant_op = block.ops[i]
            fill_constant_idx = i
            shape_idx = -1
            # find the preceding shape op
            for j in reversed(range(fill_constant_idx)):
                if block.ops[j].type == 'shape':
                    shape_out_name = block.ops[j].output_arg_names[0]
                    if shape_out_name in fill_constant_op.input_arg_names:
                        shape_op = block.ops[j]
                        shape_idx = j
                        break
            if shape_idx < 0:
                i += 1
                continue
            # create and insert a new fill_any_like op
            _insert_fill_any_like_op(block, fill_constant_idx + 1, shape_op,
                                     fill_constant_op)
            # remove the old operators
            block._remove_op(fill_constant_idx)
            block._remove_op(shape_idx)
            # restart scanning for elementwise add from the deleted shape's index
            i = shape_idx
        i += 1
    _remove_unused_var(program)
    program._sync_with_cpp()


def program_transform(program):
    assert program.num_blocks == 1

    # copy original vars into new_program
    new_program = paddle.static.Program()
    new_block = new_program.block(0)
    new_block.vars = program.block(0).vars

    block = program.block(0)
    block_desc = block.desc

    for old_var in block_desc.all_vars():
        if old_var.has_is_parameter() and old_var.is_parameter():
            new_block.create_parameter(
                name=old_var.name(),
                shape=old_var.shape(),
                dtype=old_var.dtype(),
                type=old_var.type())
        else:
            new_block.create_var(
                name=old_var.name(),
                shape=old_var.shape(),
                dtype=old_var.dtype(),
                type=old_var.type(),
                persistable=old_var.persistable(),
                stop_gradient=old_var.stop_gradient())

    for op_idx in six.moves.range(block_desc.op_size()):
        op_desc = block_desc.op(op_idx)
        in_names = op_desc.input_arg_names()
        out_names = op_desc.output_arg_names()

        dtype_f32 = block_desc.find_var(cpt.to_bytes(out_names[0])).dtype()
        to_insert = []
        if op_desc.type() == 'fill_constant' and len(in_names) == 0:
            to_insert.append(
                _create_op_desc_('fill_constant_p', {}, {'Y': [out_names[0]]},
                                 {
                                     'shape': op_desc.attr('shape'),
                                     'value': op_desc.attr('value'),
                                     'dtype': op_desc.attr('dtype')
                                 }))

        elif op_desc.type() == 'matmul_v2':
            to_insert.append(
                _create_op_desc_('matmul_p', {
                    'X': [in_names[0]],
                    'Y': [in_names[1]]
                }, {'Z': [out_names[0]]}, {}))

        elif op_desc.type() == 'elementwise_add':
            if block.var(in_names[0]).shape != block.var(in_names[1]).shape:
                # print(block.var(in_names[0]).shape)
                # print(block.var(in_names[1]).shape)
                tmp_1 = name_gen.get_var(new_block, block.var(in_names[0]))
                to_insert.append(
                    _create_op_desc_('broadcast_p', {'X': [in_names[1]], }, {
                        'Y': [tmp_1]
                    }, {'shape': block.var(in_names[0]).shape}))
                to_insert.append(
                    _create_op_desc_(
                        'add_p', {'X': [in_names[0]],
                                  'Y': [tmp_1]}, {'Z': [out_names[0]]}, {}))
            else:
                to_insert.append(
                    _create_op_desc_('add_p', {
                        'X': [in_names[0]],
                        'Y': [in_names[1]]
                    }, {'Z': [out_names[0]]}, {}))

        elif op_desc.type() == 'tanh':
            to_insert.append(
                _create_op_desc_('tanh_p', {'X': [in_names[0]]},
                                 {'Y': [out_names[0]]}, {}))

        elif op_desc.type() == 'assign':
            tmp_1 = name_gen.get_var(new_block, block.var(in_names[0]))
            to_insert.append(
                _create_op_desc_('fill_constant_p', {}, {'Y': [tmp_1]}, {
                    'shape': block.var(in_names[0]).shape,
                    'value': 0.0,
                    'dtype': core.VarDesc.VarType.FP32
                }))
            to_insert.append(
                _create_op_desc_('add_p', {'X': [in_names[0]],
                                           'Y': [tmp_1]},
                                 {'Z': [out_names[0]]}, {}))

        elif op_desc.type() == 'reshape2':
            to_insert.append(
                _create_op_desc_('reshape_p', {'X': [in_names[0]]}, {
                    'Y': [out_names[0]]
                }, {'shape': op_desc.attr('shape')}))

        elif op_desc.type() == 'concat':
            to_insert.append(
                _create_op_desc_('concat_p', {'X': in_names}, {
                    'Y': [out_names[0]]
                }, {'axis': op_desc.attr('axis')}))

        elif op_desc.type() == 'slice':
            if op_desc.attr('decrease_axis') is None:
                to_insert.append(
                    _create_op_desc_('slice_select_p', {'X': [in_names[
                        0]]}, {'Y': [out_names[0]]}, {
                            'axis': op_desc.attr('axes'),
                            'starts': op_desc.attr('starts'),
                            'ends': op_desc.attr('ends'),
                            'strides': [1] * len(op_desc.attr('axes')),
                        }))
            else:
                tmp_shape = list(block.var(in_names[0]).shape)
                for axis in op_desc.attr('decrease_axis'):
                    tmp_shape[axis] = 1
                tmp_0 = name_gen.get_var(
                    new_block, block.var(in_names[0]), shape=tuple(tmp_shape))
                to_insert.append(
                    _create_op_desc_('slice_select_p', {'X': [in_names[
                        0]]}, {'Y': [tmp_0]}, {
                            'axis': op_desc.attr('axes'),
                            'starts': op_desc.attr('starts'),
                            'ends': op_desc.attr('ends'),
                            'strides': [1] * len(op_desc.attr('axes')),
                        }))
                to_insert.append(
                    _create_op_desc_('reshape_p', {'X': [tmp_0]}, {
                        'Y': [out_names[0]]
                    }, {'shape': block.var(out_names[0]).shape}))

        elif op_desc.type() == 'slice_grad':
            tmp_1 = name_gen.get_var(new_block, block.var(in_names[0]))
            to_insert.append(
                _create_op_desc_('fill_constant_p', {}, {'Y': [tmp_1]}, {
                    'shape': block.var(in_names[0]).shape,
                    'value': 0.0,
                    'dtype': core.VarDesc.VarType.FP32
                }))
            if op_desc.attr('decrease_axis') is None:
                to_insert.append(
                    _create_op_desc_('slice_assign_p', {
                        'X': [tmp_1],
                        'Y': [in_names[1]]
                    }, {'Z': [out_names[0]]}, {
                        'axis': op_desc.attr('axes'),
                        'starts': op_desc.attr('starts'),
                        'ends': op_desc.attr('ends'),
                        'strides': [1] * len(op_desc.attr('axes'))
                    }))
            else:
                tmp_shape = slice_assign_shape(
                    block.var(in_names[1]).shape,
                    op_desc.attr('decrease_axis'))
                tmp_2 = name_gen.get_var(
                    new_block, block.var(in_names[1]), shape=tuple(tmp_shape))
                to_insert.append(
                    _create_op_desc_('reshape_p', {'X': [in_names[1]]},
                                     {'Y': [tmp_2]}, {'shape': tmp_shape}))
                to_insert.append(
                    _create_op_desc_('slice_assign_p', {
                        'X': [tmp_1],
                        'Y': [tmp_2]
                    }, {'Z': [out_names[0]]}, {
                        'axis': op_desc.attr('axes'),
                        'starts': op_desc.attr('starts'),
                        'ends': op_desc.attr('ends'),
                        'strides': [1] * len(op_desc.attr('axes'))
                    }))

        elif op_desc.type() == 'concat_grad':
            to_insert.append(
                _create_op_desc_('split_p', {'X': [in_names[0]], },
                                 {'Y': out_names}, {
                                     'axis': op_desc.attr('axis'),
                                     'num_or_sections': [len(out_names)]
                                 }))

        elif op_desc.type() == 'reshape2_grad':
            to_insert.append(
                _create_op_desc_('reshape_p', {'X': [in_names[0]], }, {
                    'Y': [out_names[0]]
                }, {'shape': block.var(out_names[0]).shape}))

        elif op_desc.type() == 'elementwise_add_grad':
            tmp_1 = name_gen.get_var(new_block, block.var(in_names[0]))
            to_insert.append(
                _create_op_desc_('fill_constant_p', {}, {'Y': [tmp_1]}, {
                    'shape': block.var(in_names[0]).shape,
                    'value': 0.0,
                    'dtype': core.VarDesc.VarType.FP32
                }))
            if block.var(in_names[1]).shape != block.var(in_names[0]).shape:
                tmp_2 = name_gen.get_var(new_block, block.var(in_names[0]))
                to_insert.append(
                    _create_op_desc_('add_p',
                                     {'X': [in_names[0]],
                                      'Y': [tmp_1]}, {'Z': [tmp_2]}, {}))
                to_insert.append(
                    _create_op_desc_('reduce_p', {'X': [tmp_2]}, {
                        'Y': [out_names[0]]
                    }, {'axis': 0,
                        'keepdim': False}))
            else:
                to_insert.append(
                    _create_op_desc_(
                        'add_p', {'X': [in_names[0]],
                                  'Y': [tmp_1]}, {'Z': [out_names[0]]}, {}))
            if block.var(in_names[2]).shape != block.var(in_names[0]).shape:
                tmp_2 = name_gen.get_var(new_block, block.var(in_names[0]))
                to_insert.append(
                    _create_op_desc_('add_p',
                                     {'X': [in_names[0]],
                                      'Y': [tmp_1]}, {'Z': [tmp_2]}, {}))
                to_insert.append(
                    _create_op_desc_('reduce_p', {'X': [tmp_2]}, {
                        'Y': [out_names[1]]
                    }, {'axis': 0,
                        'keepdim': False}))
            else:
                to_insert.append(
                    _create_op_desc_(
                        'add_p', {'X': [in_names[0]],
                                  'Y': [tmp_1]}, {'Z': [out_names[1]]}, {}))

        elif op_desc.type() == 'matmul_v2_grad':
            tmp_0 = name_gen.get_var(
                new_block,
                block.var(out_names[0]),
                shape=transhape(block.var(in_names[2]).shape))
            to_insert.append(
                _create_op_desc_('transpose_p', {'X': [in_names[2]]},
                                 {'Y': [tmp_0]}, {}))
            to_insert.append(
                _create_op_desc_('matmul_p',
                                 {'X': [in_names[0]],
                                  'Y': [tmp_0]}, {'Z': [out_names[0]]}, {}))
            tmp_1 = name_gen.get_var(
                new_block,
                block.var(out_names[0]),
                shape=transhape(block.var(in_names[1]).shape))
            to_insert.append(
                _create_op_desc_('transpose_p', {'X': [in_names[1]]},
                                 {'Y': [tmp_1]}, {}))
            to_insert.append(
                _create_op_desc_('matmul_p', {
                    'X': [tmp_1],
                    'Y': [in_names[0]]
                }, {'Z': [out_names[1]]}, {}))

        elif op_desc.type() == 'tanh_grad':
            tmp_1 = name_gen.get_var(new_block, block.var(in_names[0]))
            to_insert.append(
                _create_op_desc_('fill_constant_p', {}, {'Y': [tmp_1]}, {
                    'shape': block.var(in_names[0]).shape,
                    'value': 1.0,
                    'dtype': core.VarDesc.VarType.FP32
                }))
            tmp_2 = name_gen.get_var(new_block, block.var(in_names[0]))
            to_insert.append(
                _create_op_desc_('mul_p',
                                 {'X': [in_names[0]],
                                  'Y': [in_names[0]]}, {'Z': [tmp_2]}, {}))
            tmp_3 = name_gen.get_var(new_block, block.var(in_names[0]))
            to_insert.append(
                _create_op_desc_('sub_p', {'X': [tmp_1],
                                           'Y': [tmp_2]}, {'Z': [tmp_3]}, {}))
            to_insert.append(
                _create_op_desc_('mul_p', {'X': [tmp_3],
                                           'Y': [in_names[1]]},
                                 {'Z': [out_names[0]]}, {}))

        elif op_desc.type() == 'fill_zeros_like':
            tmp_1 = name_gen.get_var(new_block, block.var(in_names[0]))
            to_insert.append(
                _create_op_desc_('fill_constant_p', {}, {'Y': [out_names[0]]},
                                 {
                                     'shape': block.var(in_names[0]).shape,
                                     'value': 0.0,
                                     'dtype': core.VarDesc.VarType.FP32
                                 }))

        elif op_desc.type() == 'matmul_v2_grad_grad':
            tmp_0 = name_gen.get_var(new_block, block.var(out_names[0]))
            to_insert.append(
                _create_op_desc_('matmul_p',
                                 {'X': [in_names[0]],
                                  'Y': [in_names[4]]}, {'Z': [tmp_0]}, {}))
            tmp_1 = name_gen.get_var(new_block, block.var(out_names[0]))
            to_insert.append(
                _create_op_desc_('matmul_p',
                                 {'X': [in_names[3]],
                                  'Y': [in_names[1]]}, {'Z': [tmp_1]}, {}))
            to_insert.append(
                _create_op_desc_('add_p', {'X': [tmp_0],
                                           'Y': [tmp_1]},
                                 {'Z': [out_names[0]]}, {}))
            tmp_2 = name_gen.get_var(
                new_block,
                block.var(out_names[0]),
                shape=transhape(block.var(in_names[1]).shape))
            to_insert.append(
                _create_op_desc_('transpose_p', {'X': [in_names[1]]},
                                 {'Y': [tmp_2]}, {}))
            to_insert.append(
                _create_op_desc_('matmul_p',
                                 {'X': [in_names[2]],
                                  'Y': [tmp_2]}, {'Z': [out_names[1]]}, {}))
            tmp_3 = name_gen.get_var(
                new_block,
                block.var(out_names[0]),
                shape=transhape(block.var(in_names[0]).shape))
            to_insert.append(
                _create_op_desc_('transpose_p', {'X': [in_names[0]]},
                                 {'Y': [tmp_3]}, {}))
            to_insert.append(
                _create_op_desc_('matmul_p', {
                    'X': [tmp_3],
                    'Y': [in_names[2]]
                }, {'Z': [out_names[2]]}, {}))

        elif op_desc.type() == 'elementwise_add_grad_grad':
            if block.var(in_names[0]).shape != block.var(in_names[1]).shape:
                tmp_1 = name_gen.get_var(new_block, block.var(in_names[0]))
                to_insert.append(
                    _create_op_desc_('broadcast_p', {'X': [in_names[1]], }, {
                        'Y': [tmp_1]
                    }, {'shape': block.var(in_names[0]).shape}))
                to_insert.append(
                    _create_op_desc_(
                        'add_p', {'X': [in_names[0]],
                                  'Y': [tmp_1]}, {'Z': [out_names[0]]}, {}))
            else:
                to_insert.append(
                    _create_op_desc_('add_p', {
                        'X': [in_names[0]],
                        'Y': [in_names[1]]
                    }, {'Z': [out_names[0]]}, {}))

        elif op_desc.type() == 'tanh_grad_grad':
            tmp_1 = name_gen.get_var(new_block, block.var(in_names[2]))
            to_insert.append(
                _create_op_desc_('fill_constant_p', {}, {'Y': [tmp_1]}, {
                    'shape': block.var(in_names[2]).shape,
                    'value': 1.0,
                    'dtype': core.VarDesc.VarType.FP32
                }))
            tmp_2 = name_gen.get_var(new_block, block.var(in_names[2]))
            to_insert.append(
                _create_op_desc_('mul_p',
                                 {'X': [in_names[2]],
                                  'Y': [in_names[2]]}, {'Z': [tmp_2]}, {}))
            tmp_3 = name_gen.get_var(new_block, block.var(in_names[2]))
            to_insert.append(
                _create_op_desc_('sub_p', {'X': [tmp_1],
                                           'Y': [tmp_2]}, {'Z': [tmp_3]}, {}))
            to_insert.append(
                _create_op_desc_('mul_p', {'X': [tmp_3],
                                           'Y': [in_names[0]]},
                                 {'Z': [out_names[0]]}, {}))
            tmp_4 = name_gen.get_var(new_block, block.var(in_names[2]))
            to_insert.append(
                _create_op_desc_('fill_constant_p', {}, {'Y': [tmp_4]}, {
                    'shape': block.var(in_names[2]).shape,
                    'value': -2.0,
                    'dtype': core.VarDesc.VarType.FP32
                }))
            tmp_5 = name_gen.get_var(new_block, block.var(in_names[2]))
            to_insert.append(
                _create_op_desc_('mul_p', {'X': [tmp_4],
                                           'Y': [in_names[1]]},
                                 {'Z': [tmp_5]}, {}))
            tmp_6 = name_gen.get_var(new_block, block.var(in_names[2]))
            to_insert.append(
                _create_op_desc_('mul_p', {'X': [tmp_5],
                                           'Y': [in_names[2]]},
                                 {'Z': [tmp_6]}, {}))
            to_insert.append(
                _create_op_desc_('mul_p', {'X': [tmp_6],
                                           'Y': [in_names[0]]},
                                 {'Z': [out_names[1]]}, {}))

        elif op_desc.type() == 'reshape2_grad_grad':
            to_insert.append(
                _create_op_desc_('reshape_p', {'X': [in_names[0]], }, {
                    'Y': [out_names[0]]
                }, {'shape': block.var(in_names[1]).shape}))

        elif op_desc.type() == 'sum':
            assert len(in_names) > 1
            if len(in_names) == 2:
                to_insert.append(
                    _create_op_desc_('add_p', {
                        'X': [in_names[0]],
                        'Y': [in_names[1]]
                    }, {'Z': [out_names[0]]}, {}))
            else:
                tmp = name_gen.get_var(new_block, block.var(in_names[0]))
                to_insert.append(
                    _create_op_desc_('add_p',
                                     {'X': [in_names[0]],
                                      'Y': [in_names[1]]}, {'Z': [tmp]}, {}))
                for i in range(len(in_names) - 3):
                    new_tmp = name_gen.get_var(new_block,
                                               block.var(in_names[0]))
                    to_insert.append(
                        _create_op_desc_('add_p', {
                            'X': [tmp],
                            'Y': [in_names[2 + i]]
                        }, {'Z': [new_tmp]}, {}))
                    tmp = new_tmp
                to_insert.append(
                    _create_op_desc_('add_p', {
                        'X': [tmp],
                        'Y': [in_names[-1]]
                    }, {'Z': [out_names[0]]}, {}))

        elif op_desc.type() == 'p_norm':
            tmp_0 = name_gen.get_var(new_block, block.var(in_names[0]))
            to_insert.append(
                _create_op_desc_('mul_p',
                                 {'X': [in_names[0]],
                                  'Y': [in_names[0]]}, {'Z': [tmp_0]}, {}))
            tmp_1 = name_gen.get_var(new_block, block.var(out_names[0]))
            if len(block.var(tmp_0).shape) == 1 and block.var(tmp_0).shape[
                    0] == 10201:
                tmp_2 = name_gen.get_var(new_block,
                                         block.var(tmp_0), [101, 101])
                to_insert.append(
                    _create_op_desc_('reshape_p', {'X': [tmp_0]},
                                     {'Y': [tmp_2]}, {'shape': [101, 101]}))
                to_insert.append(
                    _create_op_desc_('reduce_p', {'X': [tmp_2]}, {
                        'Y': [tmp_1]
                    }, {'axis': [0, 1],
                        'keepdim': False}))
            else:
                to_insert.append(
                    _create_op_desc_('reduce_p', {'X': [tmp_0]}, {
                        'Y': [tmp_1]
                    }, {'axis': 0,
                        'keepdim': False}))
            to_insert.append(
                _create_op_desc_('sqrt_p', {'X': [tmp_1], },
                                 {'Y': [out_names[0]]}, {}))

        elif op_desc.type() == 'index_select':
            to_insert.append(
                _create_op_desc_('index_select_p', {
                    'IndexTensor': [in_names[0]],
                    'X': [in_names[1]]
                }, {'Y': [out_names[0]]}, {'indexes': None,
                                           'axis': 0}))

        elif op_desc.type() == 'elementwise_sub':
            if block.var(in_names[0]).shape != block.var(in_names[1]).shape:
                tmp_1 = name_gen.get_var(new_block, block.var(in_names[0]))
                to_insert.append(
                    _create_op_desc_('broadcast_p', {'X': [in_names[1]], }, {
                        'Y': [tmp_1]
                    }, {'shape': block.var(in_names[0]).shape}))
                to_insert.append(
                    _create_op_desc_(
                        'sub_p', {'X': [in_names[0]],
                                  'Y': [tmp_1]}, {'Z': [out_names[0]]}, {}))
            else:
                to_insert.append(
                    _create_op_desc_('sub_p', {
                        'X': [in_names[0]],
                        'Y': [in_names[1]]
                    }, {'Z': [out_names[0]]}, {}))

        elif op_desc.type() == 'p_norm_grad':
            tmp_1 = name_gen.get_var(new_block, block.var(in_names[0]))
            to_insert.append(
                _create_op_desc_('div_p',
                                 {'X': [in_names[1]],
                                  'Y': [in_names[0]]}, {'Z': [tmp_1]}, {}))
            tmp_2 = name_gen.get_var(new_block, block.var(in_names[2]))
            to_insert.append(
                _create_op_desc_('broadcast_p', {'X': [tmp_1], }, {
                    'Y': [tmp_2]
                }, {'shape': block.var(in_names[2]).shape}))
            to_insert.append(
                _create_op_desc_('mul_p', {'X': [in_names[2]],
                                           'Y': [tmp_2]},
                                 {'Z': [out_names[0]]}, {}))

        elif op_desc.type() == 'elementwise_sub_grad':
            tmp_1 = name_gen.get_var(new_block, block.var(in_names[0]))
            to_insert.append(
                _create_op_desc_('fill_constant_p', {}, {'Y': [tmp_1]}, {
                    'shape': block.var(in_names[0]).shape,
                    'value': 0.0,
                    'dtype': core.VarDesc.VarType.FP32
                }))
            if block.var(in_names[1]).shape != block.var(in_names[0]).shape:
                tmp_2 = name_gen.get_var(new_block, block.var(in_names[0]))
                to_insert.append(
                    _create_op_desc_('add_p',
                                     {'X': [in_names[0]],
                                      'Y': [tmp_1]}, {'Z': [tmp_2]}, {}))
                to_insert.append(
                    _create_op_desc_('reduce_p', {'X': [tmp_2]}, {
                        'Y': [out_names[0]]
                    }, {'axis': 0,
                        'keepdim': False}))
            else:
                to_insert.append(
                    _create_op_desc_(
                        'add_p', {'X': [in_names[0]],
                                  'Y': [tmp_1]}, {'Z': [out_names[0]]}, {}))
            # The stop_graddients for Y is True, thus we don't need this part
            # If block.var(in_names[2]).shape != block.var(in_names[0]).shape:
            #     # print(block.var(in_names[2]).shape)
            #     tmp_2 = name_gen.get_var(new_block, block.var(out_names[0]))
            #     to_insert.append(
            #         _create_op_desc_('sub_p',
            #                          {'X': [tmp_1],
            #                           'Y': [in_names[0]]}, {'Z': [tmp_2]}, {}))
            #     to_insert.append(
            #         _create_op_desc_('reduce_p', {'X': [tmp_2]}, {
            #             'Y': [out_names[1]]
            #         }, {'axis': 0,
            #             'keepdim': False}))
            # Else:
            #     to_insert.append(
            #         _create_op_desc_(
            #             'sub_p', {'X': [tmp_1],
            #                       'Y': [in_names[0]]}, {'Z': [out_names[1]]}, {}))

        elif op_desc.type() == 'index_select_grad':
            tmp_0 = name_gen.get_var(new_block, block.var(in_names[2]))
            to_insert.append(
                _create_op_desc_('fill_constant_p', {}, {'Y': [tmp_0]}, {
                    'shape': block.var(in_names[2]).shape,
                    'value': 0.0,
                    'dtype': core.VarDesc.VarType.FP32
                }))
            to_insert.append(
                _create_op_desc_('index_assign_p', {
                    'IndexTensor': [in_names[0]],
                    'X': [tmp_0],
                    'Y': [in_names[1]]
                }, {'Z': [out_names[0]]}, {'indexes': None,
                                           'axis': 0}))

        elif op_desc.type() == 'scale':
            tmp_1 = name_gen.get_var(new_block, block.var(in_names[0]))
            to_insert.append(
                _create_op_desc_(
                    'fill_constant_p',
                    {},
                    {'Y': [tmp_1]},
                    {
                        'shape': block.var(in_names[0]).shape,
                        'value': op_desc.attr('scale'),
                        'dtype': core.VarDesc.VarType.
                        FP32  # `scale` doesn't has [dtype] attr
                    }))
            to_insert.append(
                _create_op_desc_('mul_p', {'X': [in_names[0]],
                                           'Y': [tmp_1]},
                                 {'Z': [out_names[0]]}, {}))

        elif op_desc.type() == 'matmul_v2_triple_grad':
            # x_grad_grad_grad
            tmp_0 = name_gen.get_var(
                new_block,
                block.var(out_names[0]),
                shape=transhape(block.var(in_names[7]).shape))
            to_insert.append(
                _create_op_desc_('transpose_p', {'X': [in_names[7]]},
                                 {'Y': [tmp_0]}, {}))
            tmp_1 = name_gen.get_var(new_block, block.var(out_names[0]))
            to_insert.append(
                _create_op_desc_('matmul_p',
                                 {'X': [in_names[3]],
                                  'Y': [tmp_0]}, {'Z': [tmp_1]}, {}))
            tmp_2 = name_gen.get_var(
                new_block,
                block.var(out_names[0]),
                shape=transhape(block.var(in_names[5]).shape))
            to_insert.append(
                _create_op_desc_('transpose_p', {'X': [in_names[5]]},
                                 {'Y': [tmp_2]}, {}))
            tmp_3 = name_gen.get_var(new_block, block.var(out_names[0]))
            to_insert.append(
                _create_op_desc_('matmul_p',
                                 {'X': [in_names[2]],
                                  'Y': [tmp_2]}, {'Z': [tmp_3]}, {}))
            to_insert.append(
                _create_op_desc_('add_p', {'X': [tmp_1],
                                           'Y': [tmp_3]},
                                 {'Z': [out_names[0]]}, {}))

            # y_grad_grad_grad
            tmp_4 = name_gen.get_var(
                new_block,
                block.var(out_names[0]),
                shape=transhape(block.var(in_names[6]).shape))
            to_insert.append(
                _create_op_desc_('transpose_p', {'X': [in_names[6]]},
                                 {'Y': [tmp_4]}, {}))
            tmp_5 = name_gen.get_var(new_block, block.var(out_names[1]))
            to_insert.append(
                _create_op_desc_('matmul_p',
                                 {'X': [tmp_4],
                                  'Y': [in_names[3]]}, {'Z': [tmp_5]}, {}))
            tmp_6 = name_gen.get_var(
                new_block,
                block.var(out_names[0]),
                shape=transhape(block.var(in_names[4]).shape))
            to_insert.append(
                _create_op_desc_('transpose_p', {'X': [in_names[4]]},
                                 {'Y': [tmp_6]}, {}))
            tmp_7 = name_gen.get_var(new_block, block.var(out_names[1]))
            to_insert.append(
                _create_op_desc_('matmul_p',
                                 {'X': [tmp_6],
                                  'Y': [in_names[2]]}, {'Z': [tmp_7]}, {}))
            to_insert.append(
                _create_op_desc_('add_p', {'X': [tmp_5],
                                           'Y': [tmp_7]},
                                 {'Z': [out_names[1]]}, {}))

            # z_grad_grad_new
            tmp_8 = name_gen.get_var(new_block, block.var(out_names[2]))
            to_insert.append(
                _create_op_desc_('matmul_p',
                                 {'X': [in_names[4]],
                                  'Y': [in_names[1]]}, {'Z': [tmp_8]}, {}))
            tmp_9 = name_gen.get_var(new_block, block.var(out_names[2]))
            to_insert.append(
                _create_op_desc_('matmul_p',
                                 {'X': [in_names[0]],
                                  'Y': [in_names[5]]}, {'Z': [tmp_9]}, {}))
            to_insert.append(
                _create_op_desc_('add_p', {'X': [tmp_8],
                                           'Y': [tmp_9]},
                                 {'Z': [out_names[2]]}, {}))

            # x_grad_new_new
            tmp_10 = name_gen.get_var(
                new_block,
                block.var(out_names[0]),
                shape=transhape(block.var(in_names[1]).shape))
            to_insert.append(
                _create_op_desc_('transpose_p', {'X': [in_names[1]]},
                                 {'Y': [tmp_10]}, {}))
            to_insert.append(
                _create_op_desc_('matmul_p',
                                 {'X': [in_names[3]],
                                  'Y': [tmp_10]}, {'Z': [out_names[3]]}, {}))

            # y_grad_new_new
            tmp_11 = name_gen.get_var(
                new_block,
                block.var(out_names[0]),
                shape=transhape(block.var(in_names[0]).shape))
            to_insert.append(
                _create_op_desc_('transpose_p', {'X': [in_names[0]]},
                                 {'Y': [tmp_11]}, {}))
            to_insert.append(
                _create_op_desc_('matmul_p', {
                    'X': [tmp_11],
                    'Y': [in_names[3]]
                }, {'Z': [out_names[4]]}, {}))

        elif op_desc.type() == 'tanh_triple_grad':
            # x_grad_grad_grad
            tmp_1 = name_gen.get_var(new_block, block.var(in_names[4]))
            to_insert.append(
                _create_op_desc_('fill_constant_p', {}, {'Y': [tmp_1]}, {
                    'shape': block.var(in_names[4]).shape,
                    'value': -2.0,
                    'dtype': core.VarDesc.VarType.FP32
                }))
            tmp_2 = name_gen.get_var(new_block, block.var(in_names[4]))
            to_insert.append(
                _create_op_desc_('mul_p', {'X': [tmp_1],
                                           'Y': [in_names[4]]},
                                 {'Z': [tmp_2]}, {}))
            tmp_3 = name_gen.get_var(new_block, block.var(in_names[4]))
            to_insert.append(
                _create_op_desc_('mul_p', {'X': [tmp_2],
                                           'Y': [in_names[1]]},
                                 {'Z': [tmp_3]}, {}))
            tmp_4 = name_gen.get_var(new_block, block.var(in_names[4]))
            to_insert.append(
                _create_op_desc_('mul_p', {'X': [tmp_3],
                                           'Y': [in_names[3]]},
                                 {'Z': [tmp_4]}, {}))
            tmp_5 = name_gen.get_var(new_block, block.var(in_names[4]))
            to_insert.append(
                _create_op_desc_('fill_constant_p', {}, {'Y': [tmp_5]}, {
                    'shape': block.var(in_names[4]).shape,
                    'value': 1.0,
                    'dtype': core.VarDesc.VarType.FP32
                }))
            tmp_6 = name_gen.get_var(new_block, block.var(in_names[4]))
            to_insert.append(
                _create_op_desc_('mul_p',
                                 {'X': [in_names[4]],
                                  'Y': [in_names[4]]}, {'Z': [tmp_6]}, {}))
            tmp_7 = name_gen.get_var(new_block, block.var(in_names[4]))
            to_insert.append(
                _create_op_desc_('sub_p', {'X': [tmp_5],
                                           'Y': [tmp_6]}, {'Z': [tmp_7]}, {}))
            tmp_8 = name_gen.get_var(new_block, block.var(in_names[4]))
            to_insert.append(
                _create_op_desc_('mul_p', {'X': [tmp_7],
                                           'Y': [in_names[2]]},
                                 {'Z': [tmp_8]}, {}))
            to_insert.append(
                _create_op_desc_('add_p', {'X': [tmp_4],
                                           'Y': [tmp_8]},
                                 {'Z': [out_names[0]]}, {}))

            # y_grad_grad_new
            tmp_9 = name_gen.get_var(new_block, block.var(in_names[4]))
            to_insert.append(
                _create_op_desc_('mul_p', {'X': [tmp_2],
                                           'Y': [in_names[0]]},
                                 {'Z': [tmp_9]}, {}))
            to_insert.append(
                _create_op_desc_('mul_p', {'X': [tmp_9],
                                           'Y': [in_names[3]]},
                                 {'Z': [out_names[1]]}, {}))

            # y_grad_new_new
            tmp_10 = name_gen.get_var(new_block, block.var(in_names[4]))
            to_insert.append(
                _create_op_desc_('mul_p',
                                 {'X': [in_names[1]],
                                  'Y': [in_names[3]]}, {'Z': [tmp_10]}, {}))
            tmp_11 = name_gen.get_var(new_block, block.var(in_names[4]))
            to_insert.append(
                _create_op_desc_('mul_p',
                                 {'X': [in_names[2]],
                                  'Y': [in_names[4]]}, {'Z': [tmp_11]}, {}))
            tmp_12 = name_gen.get_var(new_block, block.var(in_names[4]))
            to_insert.append(
                _create_op_desc_('add_p', {'X': [tmp_10],
                                           'Y': [tmp_11]}, {'Z': [tmp_12]},
                                 {}))
            tmp_13 = name_gen.get_var(new_block, block.var(in_names[4]))
            to_insert.append(
                _create_op_desc_('mul_p', {'X': [tmp_1],
                                           'Y': [tmp_12]}, {'Z': [tmp_13]},
                                 {}))
            to_insert.append(
                _create_op_desc_('mul_p', {'X': [tmp_13],
                                           'Y': [in_names[0]]},
                                 {'Z': [out_names[2]]}, {}))

        elif op_desc.type() == 'elementwise_add_triple_grad':
            tmp_1 = name_gen.get_var(new_block, block.var(in_names[2]))
            to_insert.append(
                _create_op_desc_('fill_constant_p', {}, {'Y': [tmp_1]}, {
                    'shape': block.var(in_names[2]).shape,
                    'value': 0.0,
                    'dtype': core.VarDesc.VarType.FP32
                }))
            if block.var(in_names[0]).shape != block.var(in_names[2]).shape:
                tmp_2 = name_gen.get_var(new_block, block.var(in_names[2]))
                to_insert.append(
                    _create_op_desc_('add_p',
                                     {'X': [in_names[2]],
                                      'Y': [tmp_1]}, {'Z': [tmp_2]}, {}))
                to_insert.append(
                    _create_op_desc_('reduce_p', {'X': [tmp_2]}, {
                        'Y': [out_names[0]]
                    }, {'axis': 0,
                        'keepdim': False}))
            else:
                to_insert.append(
                    _create_op_desc_(
                        'add_p', {'X': [in_names[2]],
                                  'Y': [tmp_1]}, {'Z': [out_names[0]]}, {}))
            if block.var(in_names[1]).shape != block.var(in_names[2]).shape:
                tmp_2 = name_gen.get_var(new_block, block.var(in_names[2]))
                to_insert.append(
                    _create_op_desc_('add_p',
                                     {'X': [in_names[2]],
                                      'Y': [tmp_1]}, {'Z': [tmp_2]}, {}))
                to_insert.append(
                    _create_op_desc_('reduce_p', {'X': [tmp_2]}, {
                        'Y': [out_names[1]]
                    }, {'axis': 0,
                        'keepdim': False}))
            else:
                to_insert.append(
                    _create_op_desc_(
                        'add_p', {'X': [in_names[2]],
                                  'Y': [tmp_1]}, {'Z': [out_names[1]]}, {}))
        else:
            # print(op_desc.type())
            assert op_desc.type() in {'adam', 'shape', 'fill_constant'}
            to_insert.append(op_desc)

        for new_op_desc in to_insert:
            _new_op_desc = new_block.desc.append_op()
            _new_op_desc.copy_from(new_op_desc)
            op = Operator(block=new_block, desc=_new_op_desc)
            new_block.ops.append(op)
    return new_program
