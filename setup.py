"""
Setup configuration
"""

import setuptools


def get_readme() -> str:
    """get README"""
    with open("README.md", encoding="utf-8") as f:
        return f.read()


def get_requirements() -> list:
    """get requirements from PaddleScience/requirements.txt"""
    req_list = []
    with open("requirements.txt", "r") as f:
        req_list = f.read().splitlines()
    return req_list


if __name__ == "__main__":
    setuptools.setup(
        name="paddlesci",
        author="PaddlePaddle",
        url="https://github.com/PaddlePaddle/PaddleScience",
        description=(
            "PaddleScience is SDK and library for developing AI-driven scientific computing"
            " applications based on PaddlePaddle."
        ),
        long_description=get_readme(),
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(
            exclude=(
                "docs",
                "examples",
                "jointContribution",
                "test_tipc",
                "test",
                "tools",
                "ppsci/externals*",
                "ppsci/externals/*",
            )
        ),
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Mathematics",
        ],
        install_requires=get_requirements(),
        use_scm_version={
            "version_scheme": "release-branch-semver",  # 确保发布分支生成干净版本
            "local_scheme": "node-and-date",  # 本地开发版本包含日期和提交信息
            "version_file": "ppsci/_version.py",
            "fallback_version": "1.4.0",
            "root": "..",
            "tag_regex": r"^v(\d+\.\d+\.\d+)$",  # 严格匹配标签格式 vX.Y.Z
        },
        setup_requires=["setuptools_scm"],  # 确保 setuptools_scm 可用
    )
