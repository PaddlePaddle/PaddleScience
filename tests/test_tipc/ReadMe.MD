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

# PaddleScience TIPC benchmark test script (测试脚本运行说明)
# Docker enviroment (运行环境)
docker image: registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.2-cudnn8-gcc82
paddle = 0.0.0
paddlescience = v1.0 Beta
python = 3.7
# Process of running TIPC test (运行benchmark测试步骤)
git clone https://github.com/PaddlePaddle/PaddleScience.git
cd PaddleScience
# Kick off the modle : [/workspace/PaddleScience/examples/cylinder/2d_unsteady_continuous/cylinder2d_unsteady_train_newapi.py]
# 运行模型(以2d非定常圆柱绕流为例子)
## Single Card, automatic profiling (单卡:自动运行打开Profiling)
export CUDA_VISIBLE_DEVICES=0 
bash tests/test_tipc/benchmark_train.sh /workspace/PaddleScience/tests/test_tipc/configs/train_2d_unsteady_continuous.txt benchmark_train
## Multi-Card (多卡)
