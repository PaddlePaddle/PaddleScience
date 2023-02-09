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

import yaml
import numpy as np


class YamlLoader(object):
    """
    yaml_loader
    """

    def __init__(self, yml):
        """initialize"""
        try:
            with open(yml, encoding="utf-8") as f:
                self.yml = yaml.load(f, Loader=yaml.FullLoader)
        except Exception as e:
            print(e)
        # self.logger = logger
        self.case_num = len(self.yml)

    def __str__(self):
        """str"""
        return str(self.yml)

    def _get_py_dir(self, n):
        """
        get py dir
        """
        return self.yml[n].get("Dir")

    def _get_time(self, n):
        time = self.yml[n].get("Time")
        if time:
            return int((time.get("end_time") - time.get("start_time")) /
                       time.get("time_step"))
        else:
            return 1

    def _get_label(self, n):
        label = self.yml[n].get("label")
        if label:
            return '110'
        else:
            return ''

    def _get_dis(self, n):
        dis = self.yml[n].get("dis")
        if dis:
            return True
        else:
            return False

    def get_all_case(self):
        """
        get all case name
        """
        # 获取全部case name
        i = 0
        while i < self.case_num:
            py_dir = self._get_py_dir(i)
            yield i, py_dir
            i += 1

    def get_solution_dir(self, n):
        """get_solution_dir"""
        pp = self.yml[n].get("Post-processing")
        num = self._get_time(n)
        label = self._get_label(n)
        return pp.get("solution_filename") + label + "-t{}-p0.npy".format(
            num), pp.get("solution_save_dir")

    def get_global_config(self, n):
        """get global config"""
        gf = self.yml[n].get("Global")
        return gf.get("static_enable"), gf.get("prim_enable")

    def get_global_epochs(self, n):
        """get global epochs"""
        ge = self.yml[n].get("Global")
        return ge.get("epochs")

    def get_geometry_npoints(self, n):
        """get geometry npoints"""
        gn = self.yml[n].get("Geometry")
        return gn.get("npoints")

    def get_docstring(self, n):
        """get docstring"""
        return self.yml[n].get("docs")

    def get_case(self, i):
        """
        debug
        """
        py_dir = self._get_py_dir(i)
        return i, py_dir


class GenerateOrder(object):
    """
    generate order
    """

    def __init__(self, case):
        """initialize"""
        self.case = case

    def get_order(self, py, case_num, dis_flag):
        """get order"""
        if dis_flag:
            order = "python3.7 -m paddle.distributed.launch --devices=0,1 {} -c {} -i {}".format(
                py, self.case, case_num)
        else:
            order = "python3.7 {} -c {} -i {}".format(py, self.case, case_num)
        return order

    def __call__(self, py, case_num, dis_flag=False):
        """call"""
        return self.get_order(py, case_num, dis_flag)


class CompareSolution(object):
    """
    compare solution
    """

    def __init__(self, standard_dir, solution_dir):
        """initialize"""
        self.standard = np.load(standard_dir, allow_pickle=True)
        self.solution = np.load(solution_dir, allow_pickle=True)

    def accur_verify(self, static=False):
        """accuracy verify"""
        if isinstance(self.standard, np.lib.npyio.NpzFile):
            standard = self._convert_standard(static)
            solution = self._convert_solution(self.solution)
            compare(standard, solution)
        else:
            compare(self.standard, self.solution)

    def converge_verify(self, static=False, npoints=10):
        """converge verify"""
        if isinstance(self.standard, np.lib.npyio.NpzFile):
            standard = self._convert_standard(static)
            solution = self._convert_solution(self.solution)
            compare_CE(standard, solution, npoints)
        else:
            compare_CE(self.standard, self.solution, npoints)

    def _convert_solution(self, solution):
        """convert run solution shape"""
        rslt = np.delete(solution, [0, 1], axis=1)
        return rslt

    def _convert_standard(self, static=False):
        """convert standard solution shape"""
        label = None
        if static is False:
            label = "dyn_solution"
        elif static is True:
            label = "stc_solution"
        standard = self.standard.get(label)
        length = len(standard)
        s = standard[0]
        for i in range(1, length):
            s = np.vstack((s, standard[i]))
        return s


def compare(res, expect, delta=1e-6, rtol=1e-5, mode="close"):
    """
    比较函数
    :param paddle: paddle结果
    :param torch: torch结果
    :param delta: 误差值
    :return:
    """
    if isinstance(res, np.ndarray):
        assert res.shape == expect.shape
        if mode == "close":
            assert np.allclose(
                res, expect, atol=delta, rtol=rtol, equal_nan=True)
        elif mode == "equal":
            res = res.astype(expect.dtype)
            assert np.array_equal(res, expect, equal_nan=True)
    elif isinstance(res, (list, tuple)):
        for i, j in enumerate(res):
            compare(j, expect[i], delta, rtol, mode=mode)
    elif isinstance(res, (int, float, complex, bool)):
        if mode == "close":
            assert np.allclose(
                res, expect, atol=delta, rtol=rtol, equal_nan=True)
        elif mode == "equal":
            assert np.array_equal(res, expect, equal_nan=True)
    else:
        assert TypeError


def compare_CE(res, expect, npoints, delta=1e-6, rtol=1e-5, mode="close"):
    if isinstance(res, np.ndarray):
        assert res.shape == expect.shape
        if mode == "close":
            index = np.divide(
                (res - expect),
                expect,
                np.zeros_like(res - expect),
                where=expect != 0)
            print("The result is:")
            print(np.linalg.norm(index) / npoints)
            assert (np.array(np.linalg.norm(index) / npoints) <= 1e-3).all()
        elif mode == "equal":
            res = res.astype(expect.dtype)
            assert np.array_equal(res, expect, equal_nan=True)
    else:
        assert TypeError
