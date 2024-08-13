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
"""This *.py file is an example of downloading data of the 2d-unsteady-cylinder–flow case"""
import os
import tarfile

import wget

DATASETS = "https://paddle-org.bj.bcebos.com/paddlescience/datasets/cylinder2d_unsteady_Re100/cylinder2d_unsteady_Re100_dataset.tar"

dirname = os.path.dirname(os.path.abspath(__file__)) + "/"
print("* Running [download_dataset.py]")
wget.download(DATASETS, out=dirname)

with tarfile.open("cylinder2d_unsteady_Re100_dataset.tar") as tar:
    tar.extractall()
