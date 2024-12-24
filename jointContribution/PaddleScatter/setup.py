from setuptools import find_packages
from setuptools import setup

install_requires = []

test_requires = [
    "pytest",
    "pytest-cov",
]

setup(
    name="paddle_scatter",
    version="1.0",
    description="Paddle Extension Library of Optimized Scatter Operations, originally from https://github.com/rusty1s/pytorch_scatter",
    author="NKNaN",
    url="https://github.com/PaddlePaddle/PaddleScience/jointContribution/paddle_scatter",
    keywords=["paddle", "scatter", "segment", "gather"],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "test": test_requires,
    },
    packages=find_packages(),
)
