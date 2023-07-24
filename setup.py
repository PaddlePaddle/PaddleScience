"""
Setup configuration
"""

import setuptools


def readme():
    """README"""
    with open("README.md") as f:
        return f.read()


if __name__ == "__main__":
    setuptools.setup(
        name="paddlesci",
        version="1.1.0",
        author="PaddlePaddle",
        url="https://github.com/PaddlePaddle/PaddleScience",
        description=(
            "PaddleScience is SDK and library for developing AI-driven scientific computing"
            " applications based on PaddlePaddle."
        ),
        long_description=readme(),
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(
            exclude=(
                "docs",
                "examples",
                "jointContribution",
                "test_tipc",
                "test",
                "tools",
            )
        ),
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Mathematics",
        ],
        install_requires=[
            "numpy>=1.20.0",
            "scipy",
            "sympy",
            "matplotlib",
            "vtk",
            "pyevtk",
            "wget",
            "scipy",
            "visualdl",
            "pyvista==0.37.0",
            "pyyaml",
            "scikit-optimize",
            "h5py",
            "meshio==5.3.4",
            "tqdm",
            "imageio",
        ],
    )
