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

import os
import wget
import zipfile
import tarfile

def make_targz(output_filename, source_dir):
    """
    一次性打包目录为tar.gz
    :param output_filename: 压缩文件名
    :param source_dir: 需要打包的目录
    :return: bool
    """
    try:
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))

        return True
    except Exception as e:
        print(e)
        return False

# make_targz('aa.tar.gz','folder')

def untar(fname, dirs):
    """
    解压tar.gz文件
    :param fname: 压缩文件名
    :param dirs: 解压后的存放路径
    :return: bool
    """
    try:
        t = tarfile.open(fname)
        t.extractall(path = dirs)
        return True
    except Exception as e:
        print(e)
        return False

# untar('aa.tar.gz','./')



# datasets = 'https://dataset.bj.bcebos.com/PaddleScience/cylinder2D_continuous/datasets.zip'

# wget.download(datasets)

# with zipfile.ZipFile('datasets.zip', 'r') as zip_ref:
#     zip_ref.extractall('./')

dataset1 = 'https://cylinder3d-ic-data.bj.bcebos.com/ic_data.tar.gz'
dataset2 = 'https://cylinder3d-ic-data.bj.bcebos.com/new_sp_data.tar.gz'

wget.download(dataset1)
untar('ic_data.tar.gz','./')

wget.download(dataset2)
untar('new_sp_data.tar.gz','./')
