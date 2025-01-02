<!--
SPDX-FileCopyrightText: Copyright (c) 2022 The paddle-harmonics Authors. All rights reserved.

SPDX-License-Identifier: BSD-3-Clause

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

<!-- <div align="center">
    <img src="https://raw.githubusercontent.com/NVIDIA/torch-harmonics/main/images/logo/logo.png"  width="568">
    <br>
    <a href="https://github.com/NVIDIA/torch-harmonics/actions/workflows/tests.yml"><img src="https://github.com/NVIDIA/torch-harmonics/actions/workflows/tests.yml/badge.svg"></a>
    <a href="https://pypi.org/project/torch_harmonics/"><img src="https://img.shields.io/pypi/v/torch_harmonics"></a>
</div> -->

<!--
[![pypi](https://img.shields.io/pypi/v/torch_harmonics)](https://pypi.org/project/torch_harmonics/)
-->

<!-- # spherical harmonic transforms -->

# paddle-harmonics

![paddle_logo](images/logo/paddle_logo.png)

> [!IMPORTANT]
> paddle-harmonics is develop base [torch-harmonics](https://github.com/NVIDIA/torch-harmonics/tree/v0.6.5).
>
> It is recommended to install **nightly-build(develop)** Paddle before running any code in this repo.

[**Overview**](#overview) | [**Installation**](#installation) | [**More information**](#more-about-torch-harmonics) | [**Getting started**](#getting-started) | [**Contributors**](#contributors) | [**Cite us**](#cite-us) | [**References**](#references)


## Overview

paddle-harmonics is a differentiable implementation of the Spherical Harmonic transform in Paddle. It was originally implemented to enable Spherical Fourier Neural Operators (SFNO). It uses quadrature rules to compute the projection onto the associated Legendre polynomials and FFTs for the projection onto the harmonic basis. This algorithm tends to outperform others with better asymptotic scaling for most practical purposes.

paddle-harmonics uses Paddle primitives to implement these operations, making it fully differentiable. Moreover, the quadrature can be distributed onto multiple ranks making it spatially distributed.

paddle-harmonics has been used to implement a variety of differentiable PDE solvers which generated the animations below. Moreover, it has enabled the development of Spherical Fourier Neural Operators (SFNOs) [1].

<div align="center">
<table border="0" cellspacing="0" cellpadding="0">
    <tr>
        <td><img src="https://media.githubusercontent.com/media/NVIDIA/torch-harmonics/main/images/sfno.gif"  width="240"></td>
        <td><img src="https://media.githubusercontent.com/media/NVIDIA/torch-harmonics/main/images/zonal_jet.gif"  width="240"></td>
        <td><img src="https://media.githubusercontent.com/media/NVIDIA/torch-harmonics/main/images/allen-cahn.gif"  width="240"></td>
    </tr>
<!--     <tr>
        <td style="text-align:center; border-style : hidden!important;">Shallow Water Eqns.</td>
        <td style="text-align:center; border-style : hidden!important;">Ginzburg-Landau Eqn.</td>
        <td style="text-align:center; border-style : hidden!important;">Allen-Cahn Eqn.</td>
    </tr>  -->
</table>
</div>


## Installation
Build in your environment using the Python package:

install dependencies
```bash
# install paddle nightly
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu123/

# install cartopy
sudo apt install libgeos-dev -y
pip install shapely==1.8.0 pyshp==2.3.0 pyproj==3.3.1
pip install cartopy

# instal other dependencies
pip install parameterized
pip install imageio

# install tensorly
git clone https://github.com/tensorly/tensorly
cd tensorly
pip install -e .
```

install paddle-harmonics
```bash
git clone https://github.com/PaddlePaddle/PaddleScience.git
cd jointContribution/paddle-harmonics
pip install -e .
```

## More about paddle-harmonics

### Spherical harmonics

The [spherical harmonics](https://en.wikipedia.org/wiki/Spherical_harmonics) are special functions defined on the two-dimensional sphere $S^2$ (embedded in three dimensions). They form an orthonormal basis of the space of square-integrable functions defined on the sphere $L^2(S^2)$ and are comparable to the harmonic functions defined on a circle/torus. The spherical harmonics are defined as

$$
Y_l^m(\theta, \lambda) = \sqrt{\frac{(2l + 1)}{4 \pi} \frac{(l - m)!}{(l + m)!}} P_l^m(\cos \theta)  \exp(im\lambda),
$$

where $\theta$ and $\lambda$ are colatitude and longitude respectively, and $P_l^m$ the normalized, [associated Legendre polynomials](https://en.wikipedia.org/wiki/Associated_Legendre_polynomials).

<div align="center">
<img src="https://media.githubusercontent.com/media/NVIDIA/torch-harmonics/main/images/spherical_harmonics.gif" width="432">
<br>
Spherical harmonics up to degree 5
</div>

### Spherical harmonic transform

The spherical harmonic transform (SHT)

$$
f_l^m = \int_{S^2}  \overline{Y_{l}^{m}}(\theta, \lambda) f(\theta, \lambda) \mathrm{d} \mu(\theta, \lambda)
$$

realizes the projection of a signal $f(\theta, \lambda)$ on $S^2$ onto the spherical harmonics basis. The SHT generalizes the Fourier transform on the sphere. Conversely, a truncated series expansion of a function $f$ can be written in terms of spherical harmonics as

$$
f (\theta, \lambda) = \sum_{m=-M}^{M} \exp(im\lambda) \sum_{l=|m|}^{M} \hat f_l^m  P_l^m (\cos \theta),
$$

where $\hat{f}_l^m$, are the expansion coefficients associated to the mode $m$, $n$.

The implementation of the SHT follows the algorithm as presented in [2]. A direct spherical harmonic transform can be accomplished by a Fourier transform

$$
\hat f^m(\theta) = \frac{1}{2 \pi} \int_{0}^{2\pi} f(\theta, \lambda) \exp(-im\lambda) \mathrm{d} \lambda
$$

in longitude and a Legendre transform

$$
\hat f_l^m = \frac{1}{2} \int^{\pi}_0 \hat f^{m} (\theta) P_l^m (\cos \theta) \sin \theta \mathrm{d} \theta
$$

in latitude.

### Discrete Legendre transform

The second integral, which computed the projection onto the Legendre polynomials is realized with quadrature. On the Gaussian grid, we use Gaussian quadrature in the $\cos \theta$ domain. The integral

$$
\hat f_l^m = \frac{1}{2} \int_{-1}^1 \hat{f}^m(\arccos x) P_l^m (x) \mathrm{d} x
$$

is obtained with the substitution $x = \cos \theta$ and then approximated by the sum

$$
\hat f_l^m = \sum_{j=1}^{N_\theta}  \hat{f}^m(\arccos x_j) P_l^m(x_j) w_j.
$$

Here, $x_j \in [-1,1]$ are the quadrature nodes with the respective quadrature weights $w_j$.

## Getting started

The main functionality of `paddle_harmonics` is provided in the form of `paddle.nn.Layers` for composability. A minimum example is given by:

```python
import paddle
import paddle_harmonics as th

device = paddle.device.set_device('gpu' if paddle.device.cuda.device_count() > 0 else 'cpu')

nlat = 512
nlon = 2*nlat
batch_size = 32
signal = paddle.randn([batch_size, nlat, nlon])

# transform data on an equiangular grid
sht = th.RealSHT(nlat, nlon, grid="equiangular").to(device)

coeffs = sht(signal)
```

Detailed usage of paddle-harmonics, alongside helpful analysis provided in a series of notebooks:

1. [Getting started](./notebooks/getting_started.ipynb)
2. [Quadrature](./notebooks/quadrature.ipynb)
3. [Visualizing the spherical harmonics](./notebooks/plot_spherical_harmonics.ipynb)
4. [Spectral fitting vs. SHT](./notebooks/gradient_analysis.ipynb)
5. [Conditioning of the Gramian](./notebooks/conditioning_sht.ipynb)
6. [Solving the Helmholtz equation](./notebooks/helmholtz.ipynb)
7. [Solving the shallow water equations](./notebooks/shallow_water_equations.ipynb)
8. [Training Spherical Fourier Neural Operators](./notebooks/train_sfno.ipynb)

## Remarks on automatic mixed precision (AMP) support

Note that paddle-harmonics uses Fourier transforms from `paddle.fft` which in turn uses kernels from the optimized `cuFFT` library. This library supports fourier transforms of `float32` and `float64` (i.e. `single` and `double` precision) tensors for all input sizes. For `float16` (i.e. `half` precision) and `bfloat16` inputs however, the dimensions which are transformed are restricted to powers of two. Since data is converted to one of these reduced precision floating point formats when `paddle.amp.auto_cast` is used, paddle-harmonics will issue an error when the input shapes are not powers of two. For these cases, we recommend disabling auto_cast for the harmonics transform specifically:

```python
import paddle
import paddle_harmonics as th

sht = th.RealSHT(512, 1024, grid="equiangular").to('gpu')

with paddle.amp.auto_cast():
   # do some AMP converted math here
   x = some_math(x)
   # convert tensor to float32
   x = x.to(paddle.float32)
   # now disable auto_cast specifically for the transform,
   # making sure that the tensors are not converted
   # back to reduced precision internally
   with paddle.amp.auto_cast(enabled = False):
      xt = sht(x)

   # continue operating on the transformed tensor
   xt = some_more_math(xt)
```

Depending on the problem, it might be beneficial to upcast data to `float64` instead of `float32` precision for numerical stability.

## Contributors

[Boris Bonev](https://bonevbs.github.io) (bbonev@nvidia.com), [Thorsten Kurth](https://github.com/azrael417) (tkurth@nvidia.com), [Christian Hundt](https://github.com/gravitino) (chundt@nvidia.com), [Nikola Kovachki](https://kovachki.github.io) (nkovachki@nvidia.com), [Jean Kossaifi](http://jeankossaifi.com) (jkossaifi@nvidia.com)  

## Cite us

If you use `paddle-harmonics` in an academic paper, please cite [1]

```bibtex
@misc{bonev2023spherical,
      title={Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere},
      author={Boris Bonev and Thorsten Kurth and Christian Hundt and Jaideep Pathak and Maximilian Baust and Karthik Kashinath and Anima Anandkumar},
      year={2023},
      eprint={2306.03838},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## References

<a id="1">[1]</a>
Bonev B., Kurth T., Hundt C., Pathak, J., Baust M., Kashinath K., Anandkumar A.;
Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere;
arXiv 2306.0383, 2023.

<a id="1">[2]</a>
Schaeffer N.;
Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations;
G3: Geochemistry, Geophysics, Geosystems, 2013.

<a id="1">[3]</a>
Wang B., Wang L., Xie Z.;
Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids;
Adv Comput Math, 2018.
