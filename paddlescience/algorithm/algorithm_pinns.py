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
from .algorithm_base import AlgorithmBase
from ..inputs import InputsAttr
from ..labels import LabelInt, LabelHolder
from ..loss import FormulaLoss

from collections import OrderedDict
import numpy as np


class PINNs(AlgorithmBase):
    """
    The Physics Informed Neural Networks Algorithm.

    Parameters:
        net(Network): The network used in PINNs algorithm.
        loss(Loss, optional): The loss used in PINNs algorithm.

    Example:
        >>> # 1. train
        >>> import paddlescience as psci
        >>> algo = psci.algorithm.PINNs(net=net, loss=loss)

        >>> # 2. predict
        >>> import paddlescience as psci
        >>> algo = psci.algorithm.PINNs(net=net)
    """

    def __init__(self, net, loss=None):
        super(PINNs, self).__init__()
        self.net = net
        self.loss = loss

    def create_inputs(self, pde):
        if type(self.loss) is FormulaLoss:
            inputs, inputs_attr = self.create_inputs_from_loss(pde)
        else:
            inputs, inputs_attr = self.create_inputs_from_pde(pde)

        # self.__print_input(inputs)
        # self.__print_input_attr(inputs_attr)

        return inputs, inputs_attr

    def create_labels(self, pde, interior_shape=None, supervised_shape=None):
        if type(self.loss) is FormulaLoss:
            labels, labels_attr = self.create_labels_from_loss(pde)
        else:
            labels, labels_attr = self.create_labels_from_pde(
                pde, interior_shape, supervised_shape)

        # self.__print_label(labels)
        # self.__print_label_attr(labels_attr)

        return labels, labels_attr

    # create inputs used as net input
    def create_inputs_from_pde(self, pde):

        inputs = list()
        inputs_attr = OrderedDict()

        # TODO: remove hard code "0"

        # interior: inputs_attr["interior"]["0"]
        inputs_attr_i = OrderedDict()
        points = pde.geometry.interior
        if pde.time_dependent == True and pde.time_disc_method is None:
            # time dependent equation with continue method
            data = self.__timespace(pde.time_array[1::], points)
        else:
            data = points
        inputs.append(data)
        inputs_attr_i["0"] = InputsAttr(0, 0)
        inputs_attr["interior"] = inputs_attr_i

        # boundary: inputs_attr["boundary"][name]
        inputs_attr_b = OrderedDict()
        for name in pde.bc.keys():
            points = pde.geometry.boundary[name]
            # time dependent equation with continue method
            if pde.time_dependent == True and pde.time_disc_method is None:
                data = self.__timespace(pde.time_array[1::], points)
            else:
                data = points
            inputs.append(data)
            inputs_attr_b[name] = InputsAttr(0, 0)
        inputs_attr["bc"] = inputs_attr_b

        # initial condition for time-dependent
        # inputs_attr["ic"]["0"]
        if pde.time_dependent == True and pde.time_disc_method is None:
            inputs_attr_it = OrderedDict()
            points = pde.geometry.interior
            data = self.__timespace(pde.time_array[0:1], points)
            inputs.append(data)
            inputs_attr_it["0"] = InputsAttr(0, 0)
            inputs_attr["ic"] = inputs_attr_it
        else:
            inputs_attr["ic"] = OrderedDict()

        # data: inputs_attr["user"]["0"]
        inputs_attr_d = OrderedDict()
        points = pde.geometry.user
        if points is not None:
            if pde.time_dependent == True and pde.time_disc_method is None:
                # time dependent equation with continue method
                data = self.__timespace(pde.time_array[1::], points)
            else:
                data = points
            inputs.append(data)
            inputs_attr_d["0"] = InputsAttr(0, 0)
        inputs_attr["user"] = inputs_attr_d

        # padding
        nprocs = paddle.distributed.get_world_size()
        for i in range(len(inputs)):
            inputs[i] = self.__padding_array(nprocs, inputs[i])

        return inputs, inputs_attr

    # create labels used in computing loss, but not used as net input 
    def create_labels_from_pde(self,
                               pde,
                               interior_shape=None,
                               supervised_shape=None):

        labels = list()
        labels_attr = OrderedDict()

        # interior
        #   - labels_attr["interior"]["equations"][i]["rhs"]
        #   - labels_attr["interior"]["equations"][i]["weight"]
        #   - labels_attr["interior"]["equations"][i]["parameter"]
        #   - labels_attr["interior"]["data_cur"][i]
        labels_attr["interior"] = OrderedDict()
        labels_attr["interior"]["equations"] = list()
        for i in range(len(pde.equations)):
            attr = dict()

            # rhs
            rhs = pde.rhs_disc["interior"][i]
            if (rhs is None) or np.isscalar(rhs):
                attr["rhs"] = rhs
            elif type(rhs) is np.ndarray:
                attr["rhs"] = LabelInt(len(labels))
                if pde.time_dependent == True and pde.time_disc_method is None:
                    data = self.__repeatspace(pde.time_array[1::], rhs)
                else:
                    data = rhs
                labels.append(data)

            # weight
            weight = pde.weight_disc[i]
            if (weight is None) or np.isscalar(weight):
                attr["weight"] = weight
            elif type(weight) is np.ndarray:
                attr["weight"] = LabelInt(len(labels))
                labels.append(weight)

            labels_attr["interior"]["equations"].append(attr)

        # interior data_cur: soluiton of current time step on interior points (time-discretized)
        if pde.time_disc_method is not None:
            labels_attr["interior"]["data_cur"] = list()
            for i in range(len(pde.dvar_n)):
                labels_attr["interior"]["data_cur"].append(
                    LabelInt(len(labels)))
                if interior_shape == None:
                    labels.append(LabelHolder())  # placeholder with shape
                else:
                    labels.append(LabelHolder(interior_shape))

        # bc
        #   - labels_attr["bc"][name_b][i]["rhs"]
        #   - labels_attr["bc"][name_b][i]["weight"]
        #   - labels_attr["bc"][name_b][i]["normal"]
        labels_attr["bc"] = OrderedDict()
        for name_b, bc in pde.bc.items():
            labels_attr["bc"][name_b] = list()
            for b in bc:
                attr = dict()
                rhs = b.rhs_disc
                weight = b.weight_disc
                normal = b.normal_disc

                # rhs
                if (rhs is None) or np.isscalar(rhs):
                    attr["rhs"] = rhs
                elif type(rhs) is np.ndarray:
                    attr["rhs"] = LabelInt(len(labels))
                    if pde.time_dependent == True and pde.time_disc_method is None:
                        data = self.__repeatspace(pde.time_array[1::], rhs)
                    else:
                        data = rhs
                    labels.append(data)

                # weight
                if (weight is None) or np.isscalar(weight):
                    attr["weight"] = weight
                elif type(weight) is np.ndarray:
                    attr["weight"] = LabelInt(len(labels))
                    if pde.time_dependent == True and pde.time_disc_method is None:
                        data = np.tile(weight, len(pde.time_array[1::]))
                    else:
                        data = weight
                    labels.append(data)

                # normal
                # print(normal)
                if normal is None:
                    attr["normal"] = normal
                elif np.isscalar(normal):
                    attr["normal"] = LabelInt(len(labels))
                    data = normal
                    labels.append(data)
                else:
                    attr["normal"] = LabelInt(len(labels))
                    if pde.time_dependent == True and pde.time_disc_method is None:
                        data = np.tile(normal, len(pde.time_array[1::]))
                    else:
                        data = normal
                    print(data)
                    labels.append(data)

                labels_attr["bc"][name_b].append(attr)

        # ic
        #   - labels_attr["ic"][i]["rhs"] 
        #   - labels_attr["ic"][i]["weight"], weight is None or scalar 
        labels_attr["ic"] = list()
        for ic in pde.ic:
            attr = dict()
            # rhs
            rhs = ic.rhs_disc
            if (rhs is None) or np.isscalar(rhs):
                attr["rhs"] = rhs
            elif type(rhs) is np.ndarray:
                attr["rhs"] = LabelInt(len(labels))
                labels.append(rhs)
            # weight
            weight = ic.weight_disc
            attr["weight"] = weight

            labels_attr["ic"].append(attr)

        # user points
        #   - labels_attr["user"]["equation"][i]["rhs"]
        #   - labels_attr["user"]["equation"][i]["weight"]
        #   - labels_attr["user"]["equation"][i]["parameter"]
        #   - labels_attr["user"]["data_cur"][i]
        #   - labels_attr["user"]["data_next"][i]

        # data_cur: solution of current time step on user points 
        # data_next: reference solution of next time step on user points 
        if pde.geometry.user is not None:
            labels_attr["user"] = OrderedDict()

            # equation
            labels_attr["user"]["equations"] = list()
            for i in range(len(pde.equations)):
                attr = dict()

                # rhs
                rhs = pde.rhs_disc["user"][i]
                if (rhs is None) or np.isscalar(rhs):
                    attr["rhs"] = rhs
                elif type(rhs) is np.ndarray:
                    attr["rhs"] = LabelInt(len(labels))
                    labels.append(rhs)

                # weight
                weight = pde.weight_disc[i]
                if (weight is None) or np.isscalar(weight):
                    attr["weight"] = weight
                elif type(weight) is np.ndarray:
                    attr["weight"] = LabelInt(len(labels))
                    labels.append(weight)

                labels_attr["user"]["equations"].append(attr)

            # data next
            labels_attr["user"]["data_next"] = list()
            for i in range(len(pde.dvar)):
                labels_attr["user"]["data_next"].append(LabelInt(len(labels)))
                if supervised_shape == None:
                    labels.append(LabelHolder())  # placeholder with shape
                else:
                    labels.append(LabelHolder(supervised_shape))

            # data cur
            if pde.time_disc_method is not None:
                labels_attr["user"]["data_cur"] = list()
                for i in range(len(pde.dvar_n)):
                    labels_attr["user"]["data_cur"].append(
                        LabelInt(len(labels)))
                    if supervised_shape == None:
                        # placeholder with shape
                        labels.append(LabelHolder())
                    else:
                        labels.append(LabelHolder(supervised_shape))

        return labels, labels_attr

    def create_inputs_from_loss(self, pde):

        inputs = list()
        inputs_attr = OrderedDict()

        inputs_attr_i = OrderedDict()
        inputs_attr_b = OrderedDict()
        inputs_attr_it = OrderedDict()
        inputs_attr_d = OrderedDict()

        # interior
        for eq in pde.equations:
            if eq in self.loss._eqlist:
                idx = self.loss._eqlist.index(eq)
                inputs.append(self.loss._eqinput[idx])
                inputs_attr_i["0"] = InputsAttr(0, 0)
                inputs_attr["interior"] = inputs_attr_i
                break
        inputs_attr["interior"] = inputs_attr_i

        # boundary
        for name in pde.bc.keys():
            if name in self.loss._bclist:
                idx = self.loss._bclist.index(name)
                inputs.append(self.loss._bcinput[idx])
                inputs_attr_b[name] = InputsAttr(0, 0)
        inputs_attr["bc"] = inputs_attr_b

        # ic
        if True in self.loss._iclist:
            inputs.append(self.loss._icinput[0])
            inputs_attr_it["0"] = InputsAttr(0, 0)
            inputs_attr["ic"] = inputs_attr_it

        # data
        if True in self.loss._suplist:
            inputs.append(self.loss._supinput[0])
            inputs_attr_d["0"] = InputsAttr(0, 0)

        inputs_attr["interior"] = inputs_attr_i
        inputs_attr["bc"] = inputs_attr_b
        inputs_attr["ic"] = inputs_attr_it
        inputs_attr["user"] = inputs_attr_d

        # padding
        nprocs = paddle.distributed.get_world_size()
        for i in range(len(inputs)):
            inputs[i] = self.__padding_array(nprocs, inputs[i])

        return inputs, inputs_attr

    # print input
    def __print_input(self, input):
        print(" ** inputs ** ")
        for i in input:
            print(i.shape)
        print("")

    # print input_attr
    def __print_input_attr(self, attr):
        for key in attr.keys():
            print(" ** ", key, " **")
            print(attr[key])
        print("")

    def create_labels_from_loss(self, pde):

        labels = list()
        labels_attr = OrderedDict()

        # equation
        labels_attr["interior"] = OrderedDict()
        labels_attr["interior"]["equations"] = list()
        for i in range(len(pde.equations)):
            eq = pde.equations[i]
            attr = dict()
            if eq in self.loss._eqlist:
                idx = self.loss._eqlist.index(eq)
                # rhs
                rhs = pde.rhs[i]
                if (rhs is None) or np.isscalar(rhs):
                    attr["rhs"] = rhs
                # weight
                weight = self.loss._eqwgt[idx]
                if (weight is None) or np.isscalar(weight):
                    attr["weight"] = weight

                labels_attr["interior"]["equations"].append(attr)

        # bc
        labels_attr["bc"] = OrderedDict()
        for name_b, bc in pde.bc.items():
            if name_b in self.loss._bclist:
                idx = self.loss._bclist.index(name_b)
                labels_attr["bc"][name_b] = list()
                for b in bc:
                    attr = dict()
                    # rhs
                    rhs = b.rhs
                    if (rhs is None) or np.isscalar(rhs):
                        attr["rhs"] = rhs
                    elif type(rhs) is np.ndarray:
                        attr["rhs"] = LabelInt(len(labels))
                        labels.append(rhs)
                    # weight
                    weight = self.loss._bcwgt[idx]
                    if (weight is None) or np.isscalar(weight):
                        attr["weight"] = weight

                    labels_attr["bc"][name_b].append(attr)

        # ic
        labels_attr["ic"] = list()
        if True in self.loss._iclist:
            for ic in pde.ic:
                attr = dict()
                rhs = ic.rhs  # rhs
                if (rhs is None) or np.isscalar(rhs):
                    attr["rhs"] = rhs
                elif type(rhs) is np.ndarray:
                    attr["rhs"] = LabelInt(len(labels))
                    labels.append(rhs)
                weight = self.loss._icwgt[0]  # weight # TODO multiple ic
                if (weight is None) or np.isscalar(weight):
                    attr["weight"] = weight

                labels_attr["ic"].append(attr)

        # sup
        if True in self.loss._suplist:
            labels_attr["user"] = OrderedDict()

            # equation
            labels_attr["user"]["equations"] = list()
            for i in range(len(pde.equations)):
                eq = pde.equations[i]
                attr = dict()
                if eq in self.loss._eqlist:
                    idx = self.loss._eqlist.index(eq)
                    # rhs
                    rhs = pde.rhs[i]
                    if (rhs is None) or np.isscalar(rhs):
                        attr["rhs"] = rhs
                    # weight
                    weight = self.loss._eqwgt[idx]
                    if (weight is None) or np.isscalar(weight):
                        attr["weight"] = weight

                    labels_attr["user"]["equations"].append(attr)

            # data next
            labels_attr["user"]["data_next"] = list()
            n = self.loss._supref[0].shape[-1]
            for i in range(n):
                labels_attr["user"]["data_next"].append(LabelInt(len(labels)))
                labels.append(self.loss._supref[0][:, i])

        # padding
        nprocs = paddle.distributed.get_world_size()
        for i in range(len(labels)):
            labels[i] = self.__padding_array(nprocs, labels[i])

        return labels, labels_attr

    # print label
    def __print_label(self, label):
        print(" ** labels ** ")
        for i in label:
            print(i.shape)
        print("")

    # print label_attr 
    def __print_label_attr(self, attr):

        print("** interior-equations ** ")
        for i in attr["interior"]["equations"]:
            print(i)

        print("** bc **")
        for k in attr["bc"].keys():
            print("- key: ", k)
            for i in attr["bc"][k]:
                print(i)

        print("** ic **")
        for i in attr["ic"]:
            print(i)

        print("** user-equations ** ")
        for i in attr["user"]["equations"]:
            print(i)

        print("** user-data ** ")
        for i in attr["user"]["data_next"]:
            print(i)

        print("")

    def feed_data_interior_cur(self, labels, labels_attr, data):
        n = len(labels_attr["interior"]["data_cur"])
        for i in range(n):
            idx = labels_attr["interior"]["data_cur"][i]
            labels[idx] = data[:, i]
            # print("idx int cur: ", idx)
        return labels

    def feed_data_user_cur(self, labels, labels_attr, data):
        n = len(labels_attr["user"]["data_cur"])
        for i in range(n):
            idx = labels_attr["user"]["data_cur"][i]
            labels[idx] = data[:, i]
            # print("idx user cur: ", idx)
        return labels

    def feed_data_user_next(self, labels, labels_attr, data):
        n = len(labels_attr["user"]["data_next"])
        for i in range(n):
            idx = labels_attr["user"]["data_next"][i]
            labels[idx] = data[:, i]
            # print("idx user next: ", idx)
        return labels

    def compute_forward(self, params, *inputs):

        outs = list()

        for i in inputs:
            out = self.net.nn_func(i, params)
            outs.append(out)

        return outs

    def compute(self, params, *inputs_labels, ninputs, inputs_attr, nlabels,
                labels_attr, pde):

        outs = list()

        inputs = inputs_labels[0:ninputs]  # inputs is from zero to ninputs
        labels = inputs_labels[ninputs::]  # labels is the rest

        # loss 
        #   - interior points: eq loss
        #   - boundary points: bc loss
        #   - initial points:  ic loss
        #   - data points: data loss and eq loss 
        loss_eq = 0.0
        loss_bc = 0.0
        loss_ic = 0.0
        loss_data = 0.0

        n = 0
        # interior points: compute eq_loss
        for name_i, input_attr in inputs_attr["interior"].items():
            input = inputs[n]

            # print("int: ", len(input))
            # print(input[0:5, :])

            loss_i, out_i = self.loss.eq_loss(
                pde,
                self.net,
                input,
                input_attr,
                labels,
                labels_attr["interior"],
                bs=-1,
                params=params)  # TODO: bs is not used
            loss_eq += loss_i
            outs.append(out_i)
            n += 1

        # boundary points: compute bc_loss 
        for name_b, input_attr in inputs_attr["bc"].items():
            input = inputs[n]

            # print("bc: ", len(input))

            loss_b, out_b = self.loss.bc_loss(
                pde,
                self.net,
                name_b,
                input,
                input_attr,
                labels,
                labels_attr,
                bs=-1,
                params=params)  # TODO: bs is not used
            loss_bc += loss_b
            outs.append(out_b)
            n += 1

        # initial points: compute ic_loss
        for name_ic, input_attr in inputs_attr["ic"].items():
            input = inputs[n]
            loss_it, out_it = self.loss.ic_loss(
                pde,
                self.net,
                input,
                input_attr,
                labels,
                labels_attr,
                bs=-1,
                params=params)
            loss_ic += loss_it
            outs.append(out_it)
            n += 1

        # data points: compute data_loss and eq_loss
        for name_d, input_attr in inputs_attr["user"].items():
            input = inputs[n]

            # print("user: ", len(input))

            # eq loss
            loss_id, out_id = self.loss.eq_loss(
                pde,
                self.net,
                input,
                input_attr,
                labels,
                labels_attr["user"],
                bs=-1,
                params=params)
            loss_eq += loss_id

            # data loss
            loss_d, out_d = self.loss.data_loss(
                pde,
                self.net,
                input,
                input_attr,
                labels,
                labels_attr["user"],
                bs=-1,
                params=params)  # TODO: bs is not used
            loss_data += loss_d
            outs.append(out_id)

            n += 1

        # loss
        p = self.loss.norm_p
        if p == 1:
            loss = self.__sqrt(loss_eq) + self.__sqrt(loss_bc) + self.__sqrt(
                loss_ic) + self.__sqrt(loss_data)
        elif p == 2:
            loss = self.__sqrt(loss_eq + loss_bc + loss_ic + loss_data)
        else:
            pass
            # TODO: error out

        loss_details = list()
        loss_details.append(self.__sqrt(loss_eq))
        loss_details.append(self.__sqrt(loss_bc))
        loss_ic = (loss - loss) if isinstance(loss_ic, float) else loss_ic
        loss_details.append(self.__sqrt(loss_ic))
        loss_data = (loss - loss) if isinstance(loss_data,
                                                float) else loss_data
        loss_details.append(self.__sqrt(loss_data))

        return loss, outs, loss_details

    def __timespace(self, time, space):

        nt = len(time)
        ns = len(space)
        ndims = len(space[0])
        time_r = np.repeat(time, ns).reshape((nt * ns, 1))
        space_r = np.tile(space, (nt, 1)).reshape((nt * ns, ndims))
        timespace = np.concatenate([time_r, space_r], axis=1)
        return timespace

    def __repeatspace(self, time, space):

        nt = len(time)
        ns = len(space)
        space_r = np.tile(space, (nt, 1)).reshape((nt * ns))
        return space_r

    def __sqrt(self, x):
        if np.isscalar(x):
            return np.sqrt(x)
        else:
            return paddle.sqrt(x)

    def __padding_array(self, nprocs, array):
        npad = (nprocs - len(array) % nprocs) % nprocs  # pad npad elements
        if array.ndim == 2:
            datapad = array[-1, :].reshape((-1, array[-1, :].shape[0]))
            for i in range(npad):
                array = np.append(array, datapad, axis=0)
        elif array.ndim == 1:
            datapad = array[-1]
            for i in range(npad):
                array = np.append(array, datapad)
        return array
