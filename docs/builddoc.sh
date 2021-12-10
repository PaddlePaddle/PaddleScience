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

# apt-get install python3-sphinx texlive-latex-recommended texlive-latex-extra dvipng
# pip3.7 install sphinx-rtd-theme recommonmark sphinx_copybutton sphinx_markdown_tables

#python3.7 -m sphinx.cmd.quickstart --sep -p PaddleScience -a PaddlePaddle -v 0.1 -l en
#python3.7 -msphinx --sep -p PaddleScience -a PaddlePaddle -v 0.1 -l en
#sphinx-quickstart --sep -p PaddleScience -a PaddlePaddle -v 0.1 -l en

# sphinx-build -b html source build

rm -rf build

export SPHINXBUILD="python3.7 -msphinx"

make html
