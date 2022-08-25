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

import argparse
from configtrans import YamlLoader


def parse_args():
    """
    parse args
    """
    parser = argparse.ArgumentParser(description='ScienceTest')
    parser.add_argument(
        '-a', '--autotest', default='CI', help='chose autotest mode: CI/CE')
    parser.add_argument('-f', '--file', help='set yaml file')
    args = parser.parse_args()
    return args


args = parse_args()
filedir = "../config/{}/{}.yaml".format(args.autotest, args.file)
obj = YamlLoader(filedir)
all_cases = obj.get_all_case()

with open("test_{}.py".format(args.file), "a") as f:
    f.write(("#!/bin/env python\n"
             "# -*- coding: utf-8 -*-\n"
             "# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python\n"
             '"""\n'
             "test {} cases\n"
             '"""\n'
             "\n"
             "import os\n"
             "import sys\n"
             "import pytest\n"
             "from runcase import RunCases\n"
             "\n"
             "\n"
             'filedir = "{}" \n'
             "obj = RunCases(filedir)\n"
             "\n"
             "\n").format(args.file, filedir))

for case in all_cases:
    case_num, py = case
    docs = obj.get_docstring(case_num)
    with open("test_{}.py".format(args.file), "a") as f:
        f.write(("def test_{}():\n"
                 '   """\n'
                 "   {}\n"
                 '   """\n'
                 '   obj.run_case({})\n'
                 "\n"
                 "\n").format(case_num, docs, case_num))

with open("test_{}.py".format(args.file), "a") as f:
    f.write(('if __name__ == "__main__":\n'
             '    code=pytest.main(["-sv", sys.argv[0]])\n    sys.exit(code)\n'
             "\n"))
