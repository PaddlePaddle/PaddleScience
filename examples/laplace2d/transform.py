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

paddle.enable_static()


class NameGen(object):
    def __init__(self):
        self.cnt = 0

    def get_name(self):
        name = 'name_gen_' + str(self.cnt)
        self.cnt = self.cnt + 1
        return name


name_gen = NameGen()


def program_transform(program):
    assert program.num_blocks == 1

    # copy original vars into new_program
    new_program = paddle.static.Program()
    new_block = new_program.block(0)
    new_block.vars = program.block(0).vars

    block = program.block(0)
    block_desc = block.desc
    for op_idx in six.moves.range(block_desc.op_size()):
        op_desc = block_desc.op(op_idx)
        # print(op_desc.type())
        in_names = op_desc.input_arg_names()
        out_names = op_desc.output_arg_names()
        to_insert = []
        if op_desc.type() == 'fill_constant':
            if len(in_names) > 0:
                to_insert.append(
                    _create_op_desc_('fill_constant_p', {
                        'ShapeTensor': [in_names[0]]
                    }, {'Y': [out_names[0]]
                        }, {'shape': None,
                            'value': op_desc.attr('value')}))
            else:
                to_insert.append(
                    _create_op_desc_('fill_constant_p', {},
                                     {'Y': [out_names[0]]}, {
                                         'shape': op_desc.attr('shape'),
                                         'value': op_desc.attr('value')
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
                tmp_0 = name_gen.get_name()
                to_insert.append(
                    _create_op_desc_('shape_p', {'X': [in_names[0]]},
                                     {'Y': [tmp_0]}, {}))
                tmp_1 = name_gen.get_name()
                to_insert.append(
                    _create_op_desc_('broadcast_p', {
                        'X': [in_names[1]],
                        'ShapeTensor': [tmp_0]
                    }, {'Y': [tmp_1]}, {'shape': None}))
            to_insert.append(
                _create_op_desc_('add_p', {'X': [in_names[0]],
                                           'Y': [tmp_1]},
                                 {'Z': [out_names[0]]}, {}))

        elif op_desc.type() == 'tanh':
            to_insert.append(
                _create_op_desc_('tanh_p', {'X': [in_names[0]]},
                                 {'Y': [out_names[0]]}, {}))

        elif op_desc.type() == 'reshape2':
            to_insert.append(
                _create_op_desc_('shape_p', {'X': [in_names[0]]},
                                 {'Y': [out_names[1]]}, {}))
            to_insert.append(
                _create_op_desc_('reshape_p', {'X': [in_names[0]]}, {
                    'Y': [out_names[0]]
                }, {'shape': op_desc.attr('shape')}))

        elif op_desc.type() == 'concat':
            to_insert.append(
                _create_op_desc_('concat_p', {'X': [in_names[0]]}, {
                    'Y': [out_names[0]]
                }, {'axis': op_desc.attr('axis')}))

        elif op_desc.type() == 'slice':
            to_insert.append(
                _create_op_desc_('slice_select_p', {'X': [in_names[0]]},
                                 {'Y': [out_names[0]]}, {
                                     'axis': op_desc.attr('axes'),
                                     'starts': op_desc.attr('starts'),
                                     'ends': op_desc.attr('ends'),
                                     'strides': op_desc.attr('decrease_axis')
                                 }))

        elif op_desc.type() == 'shape':
            to_insert.append(
                _create_op_desc_('shape_p', {'X': [in_names[0]]},
                                 {'Y': [out_names[0]]}, {}))

        elif op_desc.type() == 'slice_grad':
            tmp_0 = name_gen.get_name()
            to_insert.append(
                _create_op_desc_('shape_p', {'X': [in_names[0]]},
                                 {'Y': [tmp_0]}, {}))
            tmp_1 = name_gen.get_name()
            to_insert.append(
                _create_op_desc_('fill_constant_p', {'ShapeTensor': [tmp_0]}, {
                    'Y': [tmp_1]
                }, {'shape': None,
                    'value': 0.0}))
            to_insert.append(
                _create_op_desc_('slice_assign_p',
                                 {'X': [tmp_1],
                                  'Y': [in_names[1]]}, {'Z': [out_names[0]]}, {
                                      'axis': op_desc.attr('axes'),
                                      'starts': op_desc.attr('starts'),
                                      'ends': op_desc.attr('ends'),
                                      'strides': op_desc.attr('decrease_axis')
                                  }))

        elif op_desc.type() == 'concat_grad':
            tmp_0 = name_gen.get_name()
            to_insert.append(
                _create_op_desc_('shape_p', {'X': [in_names[1]]},
                                 {'Y': [tmp_0]}, {}))
            to_insert.append(
                _create_op_desc_('split_p', {
                    'X': [in_names[0]],
                    'ShapeTensors': [tmp_0]
                }, {'Y': [out_names[0]]}, {'axis': op_desc.attr('axis')}))

        elif op_desc.type() == 'reshape2_grad':
            to_insert.append(
                _create_op_desc_('reshape_p', {
                    'X': [in_names[0]],
                    'ShapeTensor': [in_names[1]]
                }, {'Y': [out_names[0]]}, {'shape': None}))

        elif op_desc.type() == 'elementwise_add_grad':
            tmp_0 = name_gen.get_name()
            to_insert.append(
                _create_op_desc_('shape_p', {'X': [in_names[0]]},
                                 {'Y': [tmp_0]}, {}))
            tmp_1 = name_gen.get_name()
            to_insert.append(
                _create_op_desc_('fill_constant_p', {'ShapeTensor': [tmp_0]}, {
                    'Y': [tmp_1]
                }, {'shape': None,
                    'value': 0.0}))
            if block.var(out_names[0]).shape != block.var(in_names[0]).shape:
                # print(block.var(out_names[0]).shape)
                tmp_2 = name_gen.get_name()
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
            if block.var(out_names[1]).shape != block.var(in_names[0]).shape:
                # print(block.var(out_names[1]).shape)
                tmp_2 = name_gen.get_name()
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
            tmp_0 = name_gen.get_name()
            to_insert.append(
                _create_op_desc_('transpose_p', {'X': [in_names[2]]},
                                 {'Y': [tmp_0]}, {}))
            to_insert.append(
                _create_op_desc_('matmul_p',
                                 {'X': [in_names[0]],
                                  'Y': [tmp_0]}, {'Z': [out_names[0]]}, {}))
            tmp_1 = name_gen.get_name()
            to_insert.append(
                _create_op_desc_('transpose_p', {'X': [in_names[1]]},
                                 {'Y': [tmp_1]}, {}))
            to_insert.append(
                _create_op_desc_('matmul_p', {
                    'X': [tmp_1],
                    'Y': [in_names[0]]
                }, {'Z': [out_names[1]]}, {}))

        elif op_desc.type() == 'tanh_grad':
            tmp_0 = name_gen.get_name()
            to_insert.append(
                _create_op_desc_('shape_p', {'X': [in_names[0]]},
                                 {'Y': [tmp_0]}, {}))
            tmp_1 = name_gen.get_name()
            to_insert.append(
                _create_op_desc_('fill_constant_p', {'ShapeTensor': [tmp_0]}, {
                    'Y': [tmp_1]
                }, {'shape': None,
                    'value': 1.0}))
            tmp_2 = name_gen.get_name()
            to_insert.append(
                _create_op_desc_('mul_p',
                                 {'X': [in_names[0]],
                                  'Y': [in_names[0]]}, {'Z': [tmp_2]}, {}))
            tmp_3 = name_gen.get_name()
            to_insert.append(
                _create_op_desc_('sub_p', {'X': [tmp_1],
                                           'Y': [tmp_2]}, {'Z': [tmp_3]}, {}))
            to_insert.append(
                _create_op_desc_('mul_p', {'X': [tmp_3],
                                           'Y': [in_names[1]]},
                                 {'Z': [out_names[0]]}, {}))

        elif op_desc.type() == 'fill_zeros_like':
            tmp_0 = name_gen.get_name()
            to_insert.append(
                _create_op_desc_('shape_p', {'X': [in_names[0]]},
                                 {'Y': [tmp_0]}, {}))
            tmp_1 = name_gen.get_name()
            to_insert.append(
                _create_op_desc_('fill_constant_p', {'ShapeTensor': [tmp_0]}, {
                    'Y': [out_names[0]]
                }, {'shape': None,
                    'value': 0.0}))

        else:
            print(op_desc.type())
            # to_insert.append(op_desc)

        for new_op_desc in to_insert:
            op = Operator(block=new_block, desc=new_op_desc)
            new_block.ops.append(op)
    # print(new_program)
    return new_program
