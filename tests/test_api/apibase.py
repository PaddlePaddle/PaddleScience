"""
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
"""

from inspect import isfunction
import copy
import logging
import pytest
import numpy as np
import paddle
from paddle import to_tensor


class APIBase(object):
    """
    API test base object
    """

    def _layertypes(self, func):
        """
        define layertypes
        """
        types = {0: "func", 1: "class"}
        # 设置函数执行方式，函数式还是声明式.
        if isfunction(func):
            self.__layertype = types[0]
        else:
            self.__layertype = types[1]

    def __init__(self, func):
        self.seed = 33
        np.random.seed(self.seed)
        # debug mode
        self.debug = False
        # if debug mode=True choose whether test dygrpah or static
        self.dygraph = True
        self.static = True
        self.enable_backward = True
        self.dtype = None
        # function for paddle api
        self.func = func
        self.types = []
        self.places = []
        self.backward_dtype = [np.float16, np.float32, np.float64]
        # no grad var
        self.no_grad_var = []
        # calculate grad delta, You can rewrite these value
        self.delta = 1e-6
        self.gap = 0.001
        self.rtol = 1e-7
        # choose layertypes [functional or classional]
        self._layertypes(func)
        # run hook, use user define vars and initials
        self.hook()
        # check self.types
        if not isinstance(self.types, list):
            raise TypeError("Types must be a list.")
        if len(self.types) == 0:
            raise TypeError("You must define types in hook function.")
        # 设置执行device
        if len(self.places) == 0 and paddle.device.is_compiled_with_cuda(
        ) is True:
            self.places = [paddle.CPUPlace(), paddle.CUDAPlace(0)]
        else:
            # default
            self.places = [paddle.CPUPlace()]
        # 日志等级
        if self.debug:
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s - %(levelname)s - %(message)s")
        else:
            logging.basicConfig(
                level=logging.ERROR,
                format="%(asctime)s - %(levelname)s - %(message)s")

    def hook(self):
        """
        hook function
        """
        raise NotImplementedError

    def base(self, res, data=None, **kwargs):
        """
        base test for usual tests
        such as dtype test, broadcast test etc.
        """
        self._check_dtype(res, data, **kwargs)

    def exception(self, etype, mode="c", data=None, **kwargs):
        """
        exception test
        TODO: 框架还没有实现异常处理部分，现在异常没有类型
        """
        # 禁止输入res
        if "res" in kwargs.keys():
            assert False, "exception检测不需要输入res参数"
        # 复用前向计算函数， 随便定义res
        res = np.array([0])
        if mode == "c":
            try:
                self.run(res, data, **kwargs)
            except Exception as e:
                e = str(e)
                if etype in e:
                    assert True
                else:
                    assert False, "异常校验失败,异常类型为" + etype
                # print(str(e))
        if mode == "python":
            with pytest.raises(etype):
                self.run(res, data, **kwargs)
                # print(excinfo.value)
                # assert "!23" in excinfo.value

    def run(self, res, data=None, **kwargs):
        """
        run
        Args:
            res: expect result
            **kwargs: kwargs

        Returns:
            Assertion
        """
        # 取默认type
        if self.dtype is None:
            if np.float64 in self.types:
                self.dtype = np.float64
            else:
                self.dtype = self.types[0]
        if self.debug:
            for place in self.places:
                self.place = place
                logging.info(
                    "[Place] is ===============================>>>>>>>>" + str(
                        self.place))
                # start run paddle dygraph
                if self.dygraph:
                    paddle.disable_static(self.place)
                    if isinstance(self.place, paddle.CPUPlace):
                        paddle.set_device("cpu")
                    else:
                        paddle.set_device("gpu:0")
                    logging.info("[start] run " + self.__class__.__name__ +
                                 " dygraph")
                    paddle.seed(self.seed)
                    self._check_params(res, data, **kwargs)
                    dygraph_forward_res = self._dygraph_forward()
                    logging.info("dygraph forward result is :")
                    if isinstance(dygraph_forward_res, (list, tuple)):
                        compare(dygraph_forward_res, res, self.delta,
                                self.rtol)
                        logging.info(dygraph_forward_res)
                    else:
                        compare(dygraph_forward_res.numpy(), res, self.delta,
                                self.rtol)
                        logging.info(dygraph_forward_res.numpy())
                    if self.enable_backward:
                        dygraph_backward_res = self._dygraph_backward(
                            dygraph_forward_res)
                        logging.info("[dygraph grad]")
                        logging.info(dygraph_backward_res)
                    paddle.enable_static()
                if self.static:
                    # start run paddle static
                    logging.info("[start] run " + self.__class__.__name__ +
                                 " static")
                    if self.enable_backward:
                        static_forward_res, static_backward_res = self._static_forward(
                            res, data, **kwargs)
                        logging.info("static forward result is :")
                        logging.info(static_forward_res)
                        logging.info("[static grad]")
                        logging.info(static_backward_res)
                    else:
                        static_forward_res = self._static_forward(res, data,
                                                                  **kwargs)
                        logging.info("static forward result is :")
                        logging.info(static_forward_res)
                    compare(static_forward_res, res, self.delta, self.rtol)
                    # start run torch
                if self.enable_backward:
                    grad = self.compute_grad(res, data, **kwargs)
                    logging.info("[numeric grad]")
                    logging.info(grad)
                    if self.static and self.dygraph:
                        compare_grad(
                            static_backward_res,
                            dygraph_backward_res,
                            mode="both",
                            no_grad_var=self.no_grad_var)
                    if self.dygraph:
                        compare_grad(
                            dygraph_backward_res,
                            grad,
                            mode="dygraph",
                            delta=self.delta,
                            rtol=self.rtol,
                            no_grad_var=self.no_grad_var, )
                    if self.static:
                        compare_grad(
                            static_backward_res,
                            grad,
                            mode="static",
                            delta=self.delta,
                            rtol=self.rtol,
                            no_grad_var=self.no_grad_var, )
        else:
            for place in self.places:
                self.place = place
                logging.info(
                    "[Place] is ===============================>>>>>>>>" + str(
                        self.place))

                # (1) start run paddle dygraph
                if self.dygraph:
                    paddle.disable_static(self.place)
                    if isinstance(self.place, paddle.CPUPlace):
                        paddle.set_device("cpu")
                    else:
                        paddle.set_device("gpu:0")
                    logging.info("[start] run " + self.__class__.__name__ +
                                 " dygraph")
                    # paddle.disable_static(self.place)
                    paddle.seed(self.seed)
                    self._check_params(res, data, **kwargs)

                    # ① calculate forward result
                    dygraph_forward_res = self._dygraph_forward()
                    # ② compare forward result
                    if isinstance(dygraph_forward_res, (list, tuple)):
                        compare(dygraph_forward_res, res, self.delta,
                                self.rtol)
                    else:
                        compare(dygraph_forward_res.numpy(), res, self.delta,
                                self.rtol)
                    # ③ calculate backward result
                    if self.enable_backward:
                        dygraph_backward_res = self._dygraph_backward(
                            dygraph_forward_res)

                # (2) start run paddle static
                if self.static:
                    paddle.enable_static()
                    logging.info("[start] run " + self.__class__.__name__ +
                                 " static")
                    # ① calculate forward and backward result
                    if self.enable_backward:
                        static_forward_res, static_backward_res = self._static_forward(
                            res, data, **kwargs)
                    else:
                        static_forward_res = self._static_forward(res, data,
                                                                  **kwargs)
                    # ② compare forward result
                    compare(static_forward_res, res, self.delta, self.rtol)

                # (3) check gradients
                if self.enable_backward:
                    # ① calculate numerical gradient
                    grad = self.compute_grad(res, data, **kwargs)
                    # ② compare  gradient
                    if self.static and self.dygraph:
                        compare_grad(
                            static_backward_res,
                            dygraph_backward_res,
                            mode="both",
                            no_grad_var=self.no_grad_var)
                    if self.dygraph:
                        compare_grad(
                            dygraph_backward_res,
                            grad,
                            mode="dygraph",
                            delta=self.delta,
                            rtol=self.rtol,
                            no_grad_var=self.no_grad_var, )
                    if self.static:
                        compare_grad(
                            static_backward_res,
                            grad,
                            mode="static",
                            delta=self.delta,
                            rtol=self.rtol,
                            no_grad_var=self.no_grad_var, )

    def _baserun(self, res, data=None, **kwargs):
        """
        baserun
        Args:
            res: expect result
            **kwargs: kwargs

        Returns:
            Assertion
        """
        if self.debug:
            # start run paddle dygraph
            if self.dygraph:
                paddle.disable_static(self.place)
                if isinstance(self.place, paddle.CPUPlace):
                    paddle.set_device("cpu")
                else:
                    paddle.set_device("gpu:0")
                paddle.seed(self.seed)
                logging.info("[start] run " + self.__class__.__name__ +
                             " dygraph")
                self._check_params(res, data, **kwargs)
                dygraph_forward_res = self._dygraph_forward()
                logging.info("dygraph forward result is :")
                if isinstance(dygraph_forward_res, (list, tuple)):
                    compare(dygraph_forward_res, res, self.delta, self.rtol)
                    logging.info(dygraph_forward_res)
                else:
                    compare(dygraph_forward_res.numpy(), res, self.delta,
                            self.rtol)
                    logging.info(dygraph_forward_res.numpy())
                if self.enable_backward:
                    dygraph_backward_res = self._dygraph_backward(
                        dygraph_forward_res)
                    logging.info("[dygraph grad]")
                    logging.info(dygraph_backward_res)
                paddle.enable_static()
            if self.static:
                # start run paddle static
                logging.info("[start] run " + self.__class__.__name__ +
                             " static")
                if self.enable_backward:
                    static_forward_res, static_backward_res = self._static_forward(
                        res, data, **kwargs)
                    logging.info("static forward result is :")
                    logging.info(static_forward_res)
                    logging.info("[static grad]")
                    logging.info(static_backward_res)
                else:
                    static_forward_res = self._static_forward(res, data,
                                                              **kwargs)
                    logging.info("static forward result is :")
                    logging.info(static_forward_res)
                compare(static_forward_res, res, self.delta, self.rtol)
                # start run torch
            if self.enable_backward:
                grad = self.compute_grad(res, data, **kwargs)
                logging.info("[numeric grad]")
                logging.info(grad)
                if self.static and self.dygraph:
                    compare_grad(
                        static_backward_res,
                        dygraph_backward_res,
                        mode="both",
                        no_grad_var=self.no_grad_var)
                if self.dygraph:
                    compare_grad(
                        dygraph_backward_res,
                        grad,
                        mode="dygraph",
                        delta=self.delta,
                        rtol=self.rtol,
                        no_grad_var=self.no_grad_var, )
                if self.static:
                    compare_grad(
                        static_backward_res,
                        grad,
                        mode="static",
                        delta=self.delta,
                        rtol=self.rtol,
                        no_grad_var=self.no_grad_var, )
        else:
            # (1) start run paddle dygraph
            if self.dygraph:
                paddle.disable_static(self.place)
                if isinstance(self.place, paddle.CPUPlace):
                    paddle.set_device("cpu")
                else:
                    paddle.set_device("gpu:0")
                paddle.seed(self.seed)
                logging.info("[start] run " + self.__class__.__name__ +
                             " dygraph")
                self._check_params(res, data, **kwargs)

                # ① calculate forward result
                dygraph_forward_res = self._dygraph_forward()
                # ② check forward result
                if isinstance(dygraph_forward_res, (list, tuple)):
                    compare(dygraph_forward_res, res, self.delta, self.rtol)
                else:
                    compare(dygraph_forward_res.numpy(), res, self.delta,
                            self.rtol)
                # ③ calculate backward result
                if self.enable_backward:
                    dygraph_backward_res = self._dygraph_backward(
                        dygraph_forward_res)

            # (2) start run paddle static
            if self.static:
                paddle.enable_static()
                logging.info("[start] run " + self.__class__.__name__ +
                             " static")
                # ① calculate forward and backward result
                if self.enable_backward:
                    static_forward_res, static_backward_res = self._static_forward(
                        res, data, **kwargs)
                else:
                    static_forward_res = self._static_forward(res, data,
                                                              **kwargs)
                # ② compare forward result
                compare(static_forward_res, res, self.delta, self.rtol)

            # (3) check gradient
            if self.enable_backward:
                # ① calculate numerical gradient
                grad = self.compute_grad(res, data, **kwargs)
                # ② compare gradient
                if self.dygraph and self.static:
                    compare_grad(
                        static_backward_res,
                        dygraph_backward_res,
                        mode="both",
                        no_grad_var=self.no_grad_var)
                if self.dygraph:
                    compare_grad(
                        dygraph_backward_res,
                        grad,
                        mode="dygraph",
                        delta=self.delta,
                        rtol=self.rtol,
                        no_grad_var=self.no_grad_var, )
                if self.static:
                    compare_grad(
                        static_backward_res,
                        grad,
                        mode="static",
                        delta=self.delta,
                        rtol=self.rtol,
                        no_grad_var=self.no_grad_var, )

    def _check_dtype(self, res, data, **kwargs):
        """
        check dtype
        Args:
            res: expect result
            **kwargs: kwargs

        Returns:
            Assertion
        """
        # check whether dtype is wrong, but it won't stop test cases behind, it will report at last
        # remember user enable_backward
        backward_tag = self.enable_backward
        for place in self.places:
            self.place = place
            logging.info("[Place] is ===============================>>>>>>>>" +
                         str(self.place))
            tag = True
            for dtype in self.types:
                # 判断是否应该做反向计算，只有float类型的需要反向，同时如果api明确没有反向，需要根据配置进行反向截断。
                if dtype in self.backward_dtype and backward_tag:
                    self.enable_backward = True
                else:
                    self.enable_backward = False
                logging.info("[test dtype] " + self.__class__.__name__ + str(
                    dtype))
                try:
                    self.dtype = dtype
                    self._baserun(res, data, **kwargs)
                except Exception as e:
                    logging.error("[test dtype] " + self.__class__.__name__ +
                                  str(dtype) + " failed!!!")
                    tag = False
                    # assume(tag, "[Place {}] type check Error {}".format(str(self.place), str(dtype)))
                    assert tag, "[Place {}] type check Error {}".format(
                        str(self.place), str(dtype))
                    if self.debug:
                        logging.error(e)
        self.dtype = None
        self.enable_backward = backward_tag

    def _check_params(self, res, data=None, **kwargs):
        """
        check params
        Args:
            res: expect result
            **kwargs: kwargs
        Returns:
            None
        """
        if not isinstance(res, (list, np.generic, np.ndarray)):
            raise TypeError("res must be numpy")
        self.kwargs = copy.deepcopy(kwargs)
        for k, v in self.kwargs.items():
            if isinstance(v, (np.generic, np.ndarray)):
                # no_grad_Var不需要转换类型
                if self.no_grad_var is not None and k in self.no_grad_var:
                    self.kwargs[k] = to_tensor(v)
                else:
                    self.kwargs[k] = to_tensor(v.astype(self.dtype))
                # enable compute gradient
                if self.enable_backward is True:
                    self.kwargs[k].stop_gradient = False
        if data is not None:
            self.data = to_tensor(data.astype(self.dtype))
            # enable compute gradient
            if self.enable_backward is True:
                self.data.stop_gradient = False
        self.res = res

    def compute_grad(self, res, data=None, **kwargs):
        """numeric compute grad, compute by dygraph forward

        Args:
            res (int|float): [result]
            data ([numpy], optional): [input data]. Defaults to None.
            delta (float, optional): [delta]. Defaults to 0.001.
        """
        paddle.disable_static(self.place)
        logging.info("[grad] compute " + self.__class__.__name__ + " grad")
        self._check_params(res, data, **kwargs)
        loss = self._numeric_grad()
        self.kwargs = copy.deepcopy(kwargs)
        numeric_grad = {}
        for k, v in self.kwargs.items():
            if isinstance(v, (np.generic, np.ndarray)):
                # no_grad_Var不需要转换类型
                if self.no_grad_var is not None and k in self.no_grad_var:
                    self.kwargs[k] = to_tensor(v)
                else:
                    self.kwargs[k] = to_tensor(v.astype(self.dtype))
                # enable compute gradient
                if self.enable_backward is True:
                    self.kwargs[k].stop_gradient = False
        if data is None:
            for k, v in self.kwargs.items():
                if isinstance(v, paddle.Tensor):
                    grad = []
                    shape = v.numpy().shape
                    for i in range(len(v.numpy().flatten())):
                        tmp = v.numpy().flatten()
                        tmp[i] = tmp[i] + self.gap
                        tmp = tmp.reshape(shape)
                        # print(tmp)
                        self.kwargs[k] = to_tensor(tmp.astype(self.dtype))
                        # enable compute gradient
                        if self.enable_backward is True:
                            self.kwargs[k].stop_gradient = False
                        loss_delta = self._numeric_grad()
                        g = (loss_delta - loss) / self.gap
                        # print("-----> {}".format(g))
                        grad.append(g[0])
                        # recover v to self.kwargs
                        self.kwargs[k] = v
                    numeric_grad[k] = np.array(grad).reshape(shape)
        else:
            # change data to correct dtype
            data = data.astype(self.dtype)
            grad = []
            shape = data.shape
            for i in range(len(data.flatten())):
                tmp = copy.deepcopy(data.flatten())
                tmp[i] = tmp[i] + self.gap
                tmp = tmp.reshape(shape)
                self.data = to_tensor(tmp.astype(self.dtype))
                # enable compute gradient
                if self.enable_backward is True:
                    self.data.stop_gradient = False
                loss_delta = self._numeric_grad()
                g = (loss_delta - loss) / self.gap
                grad.append(g[0])
                # recover v to self.kwargs
                self.data = data
            numeric_grad["data"] = np.array(grad).reshape(shape)
        paddle.enable_static()
        return numeric_grad

    def _numeric_grad(self):
        """
        _numeric_grad
        Returns:
            result
        """
        if self.__layertype == "func":
            res = self.func(**self.kwargs)
            loss = paddle.mean(res).numpy()
            return loss
        elif self.__layertype == "class":
            obj = self.func(**self.kwargs)
            res = obj(self.data)
            loss = paddle.mean(res).numpy()
            return loss

    def _dygraph_forward(self):
        """
        _dygraph_forward
        Returns:
            result
        """
        if self.__layertype == "func":
            res = self.func(**self.kwargs)
            return res
        elif self.__layertype == "class":
            obj = self.func(**self.kwargs)
            res = obj(self.data)
            return res

    def _dygraph_backward(self, res):
        """dygraph backward

        Args:
            res ([variable]): forward_res
        """
        loss = paddle.mean(res)
        loss.backward()
        grad = {}
        for k, v in self.kwargs.items():
            # 判断是不是Variable类型
            if isinstance(v, paddle.Tensor):
                grad[k] = v.gradient()
        if self.__layertype == "class":
            grad["data"] = self.data.gradient()
        # grad["res"] = res.gradient()
        return grad

    def _static_forward(self, res, data=None, **kwargs):
        """
        _static_forward
        """
        if self.__layertype == "func":
            paddle.seed(self.seed)
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            params = copy.deepcopy(kwargs)
            with paddle.utils.unique_name.guard():
                with paddle.static.program_guard(
                        main_program=main_program,
                        startup_program=startup_program):
                    # PS:没有单列出一个函数做值传递，因为self.kwargs只有一个，就没单列出来
                    xyz = []
                    for k, v in kwargs.items():
                        if isinstance(v, (np.generic, np.ndarray)):
                            # no_grad_Var不需要转换类型
                            if self.no_grad_var is not None and k in self.no_grad_var:
                                kwargs[k] = v
                            else:
                                kwargs[k] = v.astype(self.dtype)
                    for k, v in params.items():
                        if isinstance(v, (np.generic, np.ndarray)):
                            # no_grad_Var不需要转换类型
                            if self.no_grad_var is not None and k in self.no_grad_var:
                                params[k] = paddle.static.data(
                                    name=k, shape=v.shape, dtype=v.dtype)
                            else:
                                params[k] = paddle.static.data(
                                    name=k, shape=v.shape, dtype=self.dtype)
                            xyz.append(k)
                            # enable compute gradient
                            if self.enable_backward is True:
                                params[k].stop_gradient = False
                    output = self.func(**params)
                    if self.enable_backward:
                        loss = paddle.mean(output)
                        grad_var = {}
                        for k in xyz:
                            grad_var[k] = paddle.static.gradients(loss,
                                                                  params[k])
                        exe = paddle.static.Executor(self.place)
                        exe.run(startup_program)
                        # print(list(grad_var.values()))
                        # print([output] + list(grad_var.values()))
                        res = exe.run(
                            main_program,
                            feed=kwargs,
                            fetch_list=[output] + list(grad_var.values()),
                            return_numpy=True)
                        # combine grad
                        grad = dict(zip(xyz, res[1:]))
                        return res[0], grad
                    else:
                        exe = paddle.static.Executor(self.place)
                        exe.run(startup_program)
                        # print(list(grad_var.values()))
                        # print([output] + list(grad_var.values()))
                        res = exe.run(main_program,
                                      feed=kwargs,
                                      fetch_list=[output],
                                      return_numpy=True)
                        return res[0]
        elif self.__layertype == "class":
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            main_program.random_seed = self.seed
            startup_program.random_seed = self.seed
            params = copy.deepcopy(kwargs)
            with paddle.utils.unique_name.guard():
                with paddle.static.program_guard(
                        main_program=main_program,
                        startup_program=startup_program):
                    # PS:没有单列出一个函数做值传递，因为self.kwargs只有一个，就没单列出来
                    for k, v in kwargs.items():
                        if isinstance(v, (np.generic, np.ndarray)):
                            # no_grad_Var不需要转换类型
                            if self.no_grad_var is not None and k in self.no_grad_var:
                                kwargs[k] = v
                            else:
                                kwargs[k] = v.astype(self.dtype)
                    for k, v in params.items():
                        if isinstance(v, (np.generic, np.ndarray)):
                            # no_grad_Var不需要转换类型
                            if self.no_grad_var is not None and k in self.no_grad_var:
                                params[k] = paddle.static.data(
                                    name=k, shape=v.shape, dtype=v.dtype)
                            else:
                                params[k] = paddle.static.data(
                                    name=k, shape=v.shape, dtype=self.dtype)
                            # enable compute gradient
                            if self.enable_backward is True:
                                params[k].stop_gradient = False
                    if data is not None:
                        data = data.astype(self.dtype)
                        self.data = paddle.static.data(
                            name="data", shape=data.shape, dtype=self.dtype)
                        if self.enable_backward is True:
                            self.data.stop_gradient = False
                    data = dict({"data": data}, **kwargs)
                    obj = self.func(**params)
                    output = obj(self.data)
                    if self.enable_backward:
                        loss = paddle.mean(output)
                        g = paddle.static.gradients(loss, self.data)
                        exe = paddle.static.Executor(self.place)
                        exe.run(startup_program)
                        res = exe.run(main_program,
                                      feed=data,
                                      fetch_list=[output, g],
                                      return_numpy=True)
                        grad = {"data": res[1]}
                        return res[0], grad
                    else:
                        exe = paddle.static.Executor(self.place)
                        exe.run(startup_program)
                        res = exe.run(main_program,
                                      feed=data,
                                      fetch_list=[output],
                                      return_numpy=True)
                        return res[0]


def compare_grad(result,
                 expect,
                 delta=1e-6,
                 rtol=0.001,
                 mode=None,
                 no_grad_var=None):
    """compare grad

    Args:
        result ([dict]): [result]
        expect ([dict]): [expect]
        delta ([delta], optional): [delta]. Defaults to 1e-6.
    """
    if delta < 1e-4:
        delta = 1e-3 * 5
    if rtol < 1e-4:
        rtol = 1e-3
    logging.info("[{}] start check grad:".format(mode))
    if no_grad_var is not None:
        for key in no_grad_var:
            if key in result.keys():
                result.pop(key)
            if key in expect.keys():
                expect.pop(key)
    if result.keys() != expect.keys():
        logging.error(result.keys())
        logging.error(expect.keys())
        assert False, "grad KeyError"
    for k in result.keys():
        logging.info("check " + k + " grad ... ")
        compare(result[k], expect[k], delta, rtol)
        logging.info("check " + k + " grad ... ok")


def compare(result, expect, delta=1e-6, rtol=1e-5):
    """
    比较函数
    :param result: 输入值
    :param expect: 输出值
    :param delta: 误差值
    :return:
    """
    if isinstance(result, np.ndarray):
        expect = np.array(expect)
        res = np.allclose(
            result, expect, atol=delta, rtol=rtol, equal_nan=True)
        # 出错打印错误数据
        if res is False:
            logging.error("the result is {}".format(result))
            logging.error("the expect is {}".format(expect))
        # tools.assert_true(res)
        assert res
        # tools.assert_equal(result.shape, expect.shape)
        assert result.shape == expect.shape
    elif isinstance(result, (list, tuple)):
        for i, j in enumerate(result):
            if isinstance(j, (np.generic, np.ndarray)):
                compare(j, expect[i], delta, rtol)
            else:
                compare(j.numpy(), expect[i], delta, rtol)
        # result = np.array(result)
        # expect = np.array(expect)
        # res = np.allclose(result, expect, atol=delta)
        # # 出错打印错误数据
        # if res is False:
        #     print("the result is {}".format(result))
        #     print("the expect is {}".format(expect))
        # # tools.assert_true(res)
        # assert res
        # # tools.assert_equal(result.shape, expect.shape)
        # assert result.shape == expect.shape
    elif isinstance(result, str):
        res = result == expect
        if res is False:
            logging.error("the result is {}".format(result))
            logging.error("the expect is {}".format(expect))
        assert res
    else:
        assert result == pytest.approx(expect, delta)
        # tools.assert_almost_equal(result, expect, delta=delta)


def randtool(dtype, low, high, shape):
    """
    np random tools
    """
    if dtype == "int":
        return np.random.randint(low, high, shape)

    elif dtype == "float":
        return low + (high - low) * np.random.random(shape)


def sigmoid(x):
    """
    sigmoid
    """
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """
    tanh
    """
    s1 = np.exp(x) - np.exp(-x)
    s2 = np.exp(x) + np.exp(-x)
    s = s1 / s2
    return s


def relu(x):
    """
    relu
    """
    s = np.where(x < 0, 0, x)
    return s
