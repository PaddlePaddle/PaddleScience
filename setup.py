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

from io import open

from setuptools import setup

with open("requirements.txt", encoding="utf-8-sig") as f:
    requirements = f.readlines()

setup(
    name="paddlescience",
    packages=["paddlescience"],
    package_dir={"paddlescience": ""},
    include_package_data=True,
    entry_points={
        "console_scripts": ["paddlescience=paddlescience.paddlescience:main"]
    },
    version="0.0.0",
    install_requires=requirements,
    license="Apache License 2.0",
    description="A treasure chest for visual recognition powered by PaddlePaddle.",
    long_description_content_type="text/markdown",
    url="https://github.com/PaddlePaddle/PaddleScience",
    download_url="https://github.com/PaddlePaddle/PaddleScience.git",
    keywords=["AI for Science"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
    ], )
