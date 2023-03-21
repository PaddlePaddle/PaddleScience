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

from dis import dis
import os
from configtrans import YamlLoader, GenerateOrder, CompareSolution


class RunCases(object):
    """
    run cases
    """

    def __init__(self, yml_dir):
        self.yaml_dir = yml_dir
        self.obj = YamlLoader(self.yaml_dir)

    def run(self):
        cases = self.obj.get_all_case()
        for item in cases:
            case_num, py = item
            static, prim = self.obj.get_global_config(case_num)
            solution, standard_str = self.obj.get_solution_dir(case_num)
            npoints = self.obj.get_geometry_npoints(case_num)
            epochs = self.obj.get_global_epochs(case_num)
            print(solution)
            print(standard_str)
            order = GenerateOrder(self.yaml_dir)(py, case_num)
            os.system(order)
            cs = CompareSolution(standard_str, solution)
            if epochs <= 1000:
                cs.accur_verify(static)
            else:
                cs.converge_verify(static, npoints)

    def run_case(self, k):
        case = self.obj.get_case(k)
        case_num, py = case
        static, prim = self.obj.get_global_config(case_num)
        solution, standard_str = self.obj.get_solution_dir(case_num)
        npoints = self.obj.get_geometry_npoints(case_num)
        epochs = self.obj.get_global_epochs(case_num)
        dis_flag = self.obj._get_dis(case_num)
        print(solution)
        print(standard_str)
        order = GenerateOrder(self.yaml_dir)(py, case_num, dis_flag)
        os.system(order)
        cs = CompareSolution(standard_str, solution)
        if epochs <= 1000:
            cs.accur_verify(static)
        else:
            cs.converge_verify(static, npoints)


if __name__ == '__main__':
    file = "../standard_data/ce_week_standard/laplace2d.yaml"
    obj = RunCases(file)
    obj.run()
