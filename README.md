# Introduction
PaddleScience is a toolkit that facilitates the development of AI-Science applications based on the PaddlePaddle framework.

# Install and run examples

step1: Install PaddlePaddle

You can refer to https://github.com/PaddlePaddle/Paddle

step2: Download PaddleScience and set PYTHONPATH

```
git clone git@github.com:PaddlePaddle/PaddleScience.git
cd PaddleScience
export PYTHONPATH=$PWD:$PYTHONPATH
```

step3: Run examples

```
cd examples/laplace2d
python3.7 laplace2d.py

cd examples/darcy2d
python3.7 darcy2d.py

cd examples/ldc2d
python3.7 ldc2d.py
```
