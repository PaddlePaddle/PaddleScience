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

from collections import OrderedDict
import numpy as np


class PINNs(AlgorithmBase):
    """
    The Physics Informed Neural Networks Algorithm.

    Parameters:
        net(NetworkBase): The NN network used in PINNs algorithm.
        loss(LossBase): The loss used in PINNs algorithm.

    Example:
        >>> import paddlescience as psci
        >>> algo = psci.algorithm.PINNs(net=net, loss=loss)
    """

    def __init__(self, net, loss):
        super(PINNs, self).__init__()
        self.net = net
        self.loss = loss

    def create_inputs(self, pde):

        inputs = list()
        inputs_attr = OrderedDict()

        # TODO: hard code

        # interior: inputs_attr["interior"]["0"]
        inputs_attr_i = OrderedDict()
        points = pde.geometry.interior

        if pde.time_dependent == True and pde.time_disc_method is None:
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
            if pde.time_dependent == True and pde.time_disc_method is None:
                data = self.__timespace(pde.time_array[1::], points)
            else:
                data = points
            inputs.append(data)
            inputs_attr_b[name] = InputsAttr(0, 0)
        inputs_attr["boundary"] = inputs_attr_b

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

        # data: inputs_attr["data"]["0"]
        inputs_attr_d = OrderedDict()
        points = pde.geometry.data
        if points is not None:
            data = points
            inputs.append(data)
            inputs_attr_d["0"] = InputsAttr(0, 0)
        inputs_attr["data"] = inputs_attr_d

        return inputs, inputs_attr

    def feed_labels_data_n(self, labels, labels_attr, data_n):
        for i in range(len(data_n[0])):
            idx = labels_attr["data_n"][i]
            labels[idx] = data_n[:, i]
        return labels

    def feed_labels_data(self, labels, labels_attr, data):
        for i in range(len(data[0])):
            idx = labels_attr["data"][i]
            labels[idx] = data[:, i]
        return labels

    def create_labels(self, pde):

        labels = list()
        labels_attr = OrderedDict()

        # equation: rhs, weight, parameter
        #   - labels_attr["equation"][i]["rhs"]
        #   - labels_attr["equation"][i]["weight"]
        #   - labels_attr["equation"][i]["parameter"]
        labels_attr["equations"] = list()
        for i in range(len(pde.equations)):
            attr = dict()

            # rhs
            rhs = pde.rhs_disc[i]
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

            labels_attr["equations"].append(attr)

        # bc: rhs and weight
        #   - labels_attr["bc"][name_b][i]["rhs"]
        #   - labels_attr["bc"][name_b][i]["weight"]
        labels_attr["bc"] = OrderedDict()
        for name_b, bc in pde.bc.items():
            labels_attr["bc"][name_b] = list()
            for b in bc:
                attr = dict()
                rhs = b.rhs_disc
                weight = b.weight_disc
                if (rhs is None) or np.isscalar(rhs):
                    attr["rhs"] = rhs
                elif type(rhs) is np.ndarray:
                    attr["rhs"] = LabelInt(len(labels))
                    labels.append(rhs)

                if (weight is None) or np.isscalar(weight):
                    attr["weight"] = weight
                elif type(weight) is np.ndarray:
                    attr["weight"] = LabelInt(len(labels))
                    if pde.time_dependent == True and pde.time_disc_method is None:
                        data = np.tile(weight, len(pde.time_array[1::]))
                    else:
                        data = weight
                    labels.append(data)

                labels_attr["bc"][name_b].append(attr)

        # ic: rhs and weight
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

        # data_n: real data of previous time-step
        # in time-discretized equation
        #   - labels_attr["data_n"][i]
        if pde.time_disc_method is not None:
            labels_attr["data_n"] = list()
            for i in range(len(pde.dvar_n)):
                labels_attr["data_n"].append(LabelInt(len(labels)))
                labels.append(LabelHolder())  # placeholder with shape

        # data: real data
        #   - labels_attr["data"][0]
        if pde.geometry.data is not None:
            labels_attr["data"] = list()
            for i in range(len(pde.dvar)):
                labels_attr["data"].append(LabelInt(len(labels)))
                labels.append(LabelHolder())  # placeholder with shape

        return labels, labels_attr

    def compute(self, *inputs_labels, ninputs, inputs_attr, nlabels,
                labels_attr, pde):

        outs = list()

        inputs = inputs_labels[0:ninputs]  # inputs is from zero to ninputs
        labels = inputs_labels[ninputs::]  # labels is the rest

        n = 0
        # interior outputs and loss
        loss_eq = 0.0
        for name_i, input_attr in inputs_attr["interior"].items():
            input = inputs[n]
            loss_i, out_i = self.loss.eq_loss(
                pde,
                self.net,
                name_i,
                input,
                input_attr,
                labels,
                labels_attr,
                bs=-1)  # TODO: bs is not used
            loss_eq += loss_i
            outs.append(out_i)
            n += 1

        # boundary outputs and loss
        loss_bc = 0.0
        for name_b, input_attr in inputs_attr["boundary"].items():
            input = inputs[n]
            loss_b, out_b = self.loss.bc_loss(
                pde,
                self.net,
                name_b,
                input,
                input_attr,
                labels,
                labels_attr,
                bs=-1)  # TODO: bs is not used
            loss_bc += loss_b
            outs.append(out_b)
            n += 1

        # ic loss
        loss_ic = 0.0
        for name_ic, input_attr in inputs_attr["ic"].items():
            input = inputs[n]
            loss_it, out_it = self.loss.ic_loss(
                pde,
                self.net,
                name_ic,
                input,
                input_attr,
                labels,
                labels_attr,
                bs=-1)
            loss_ic += loss_it
            outs.append(out_it)
            n += 1

        # data loss
        loss_data = 0.0
        for name_d, input_attr in inputs_attr["data"].items():
            input = inputs[n]
            loss_d, out_d = self.loss.data_loss(
                pde, self.net, input, input_attr, labels, labels_attr,
                bs=-1)  # TODO: bs is not used
            loss_data += loss_d
            outs.append(out_d)
            n += 1

        # print("loss eq:   ", loss_eq)
        # print("loss bc:   ", loss_bc)
        # print("loss ic:   ", loss_ic)
        # print("loss data: ", loss_data)

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

        return loss, outs  # TODO: return more

    def __timespace(self, time, space):

        nt = len(time)
        ns = len(space)
        time_r = np.repeat(time, ns).reshape((nt * ns, 1))
        space_r = np.tile(space, (nt, 1)).reshape((nt * ns, 2))
        timespace = np.concatenate([time_r, space_r], axis=1)
        return timespace

    def __sqrt(self, x):
        if np.isscalar(x):
            return np.sqrt(x)
        else:
            return paddle.sqrt(x)
