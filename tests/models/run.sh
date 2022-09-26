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

rm -rf test_*.py
cases=`find ../config/$1 -name "*.yaml" | sort`

echo $cases
ignoe=""
bug=0
cp ../../examples/cylinder/3d_steady/re20_5.0.npy ./
echo "============ failed cases =============" > result.txt
for file_dir in ${cases}
do
    name=`basename -s .yaml $file_dir`
    echo ${name}
    python3.7 generate.py -f ${name} -a $1
    python3.7 test_${name}.py
    if [ $? -ne 0 ]; then
        echo test_${name} >> result.txt
        bug=`expr ${bug} + 1`
    fi
done
echo "total bugs: "${bug} >> result.txt
cat result.txt
exit ${bug}
