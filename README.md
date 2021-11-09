# Introduction
PaddleScience is a toolkits based on the PaddlePaddle framework, it aims to solve problems in the field of scientific computing.

# Install and run examples

step1: Install PaddlePaddle

You can refer to https://github.com/PaddlePaddle/Paddle

step2: Download PaddleScience and set PYTHONPATH

```
git colne git@github.com:PaddlePaddle/PaddleScience.git
cd PaddleScience
export PYTHONPATH=$PWD:$PYTHONPATH
```

step3: Run examples

```
cd examples
python3.7 laplace2d.py
python3.7 darcy2d.py
python3.7 ldc2d.py
```
