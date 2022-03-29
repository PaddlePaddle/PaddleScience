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

        elif op_desc.type() == 'reshape2':
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

        elif op_desc.type() == 'slice_grad':
            tmp_1 = name_gen.get_var(new_block, block.var(in_names[0]))
            to_insert.append(
                _create_op_desc_('fill_constant_p', {}, {'Y': [tmp_1]}, {
                    'shape': block.var(in_names[0]).shape,
                    'value': 0.0
                }))
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
            to_insert.append(
                _create_op_desc_('split_p', {'X': [in_names[0]], }, {
                    'Y': [out_names[0]]
                }, {'axis': op_desc.attr('axis'),
                    'num_or_sections': [1]}))

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
                    'value': 0.0
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
                    'value': 1.0
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
                _create_op_desc_('fill_constant_p', {}, {
                    'Y': [out_names[0]]
                }, {'shape': block.var(in_names[0]).shape,
                    'value': 0.0}))

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
                    'value': 1.0
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
                    'value': -2.0
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
            to_insert.append(
                _create_op_desc_('reduce_p', {'X': [tmp_0]}, {'Y': [tmp_1]},
                                 {'axis': 0,
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
                    'value': 0.0
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
                    'value': 0.0
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
                _create_op_desc_('fill_constant_p', {}, {'Y': [tmp_1]}, {
                    'shape': block.var(in_names[0]).shape,
                    'value': op_desc.attr('scale')
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
                    'value': -2.0
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
                    'value': 1.0
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
                    'value': 0.0
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
            assert op_desc.type() in {'adam', 'shape', 'fill_constant'}
            to_insert.append(op_desc)

        for new_op_desc in to_insert:
            _new_op_desc = new_block.desc.append_op()
            _new_op_desc.copy_from(new_op_desc)
            op = Operator(block=new_block, desc=_new_op_desc)
            new_block.ops.append(op)
    return new_program
