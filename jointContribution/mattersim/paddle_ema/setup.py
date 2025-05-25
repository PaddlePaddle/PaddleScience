from setuptools import setup, find_packages
from pathlib import Path

# see https://packaging.python.org/guides/single-sourcing-package-version/
version_dict = {}
with open(Path(__file__).parents[0] / "paddle_ema/_version.py") as fp:
    exec(fp.read(), version_dict)
version = version_dict["__version__"]
del version_dict

url = 'https://github.com/fadel/pytorch_ema'

# install_requires = ["paddlepaddle-gpu"]
install_requires = []
setup_requires = []
tests_require = []

setup(
    name='paddle_ema',
    version=version,
    description='PaddlePaddle library for computing moving averages of model parameters.',
    author='Ruibin Cheung',
    author_email='beinggod@foxmail.com',
    url=url,
    keywords=['paddlepaddle', 'parameters', 'deep-learning'],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=find_packages(),
)
