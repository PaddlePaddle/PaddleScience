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

home=$PWD

# api
cd test_examples
bash ./run.sh
api=$?
echo ${api}
cd $home


# examples
cd test_api
bash ./run.sh
example=$?
echo ${example}
cd $home

# result
echo "=============== result ================="
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
if [ `expr ${api} + ${example}` -eq 0 ]; then
  result=`find . -name "result.txt"`
  for file in ${result}
    do
      cat ${file}
      echo "----------------------"
    done
  echo "success!"
else
  result=`find . -name "result.txt"`
  for file in ${result}
    do
      cat ${file}
    done
  echo "error!"
  exit 8
fi
