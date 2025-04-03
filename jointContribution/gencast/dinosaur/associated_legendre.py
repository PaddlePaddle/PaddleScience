# Copyright 2023 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Associated Legendre function evaluation, and derivative coefficients."""

import functools

import numpy as np
import scipy.special as sps


def _evaluate_rhombus(n_l: int,
                      n_m: int,
                      x: np.ndarray,
                      truncation='rhombus') -> np.ndarray:
  """Associated Legendre Function (rhombus indexing).

  Evaluates the associated Legendre functions on the nodes `x`.

  Arguments:
    n_l: the maximum (exclusive) degree (often denoted `k` or `l`) of the
      computed polynomials.
    n_m: the maximum (exclusive) order (often denoted `m`) of the computed
      polynomials.
    x: vector of nodes in the range [-1, 1].
    truncation: 'rhombus' or 'triangle'. See below.

  Returns:
    Array p of shape (n_l, n_m, len(x)) such that

      p[k, m, i] = cₖₘ Pᵐₘ₊ₖ(xᵢ)

    for indices in range, and 0 otherwise. By default (truncation == 'rhombus'),
    all indices are in range. If truncation == 'triangle', then p[k, m, i] is
    nonzero only when l = m + k < n_l .

    The normalization constants cₖₘ are chosen such that each basis function
    has unit L²([-1, 1]) norm.
  """
  y = np.sqrt(1 - x * x)
  p = np.zeros((n_l, n_m, len(x)))
  p[0, 0] = p[0, 0] + 1 / np.sqrt(2)
  for m in range(1, n_m):
    p[0, m] = -np.sqrt(1 + 1 / (2 * m)) * y * p[0, m - 1]
  m_max = n_m
  for k in range(1, n_l):
    if truncation == 'triangle':
      m_max = min(n_m, n_l - k)
    m = np.arange(m_max).reshape((-1, 1))
    m2 = np.square(m)
    mk2 = np.square(m + k)
    mkp2 = np.square(m + k - 1)
    a = np.sqrt((4 * mk2 - 1) / (mk2 - m2))
    b = np.sqrt((mkp2 - m2) / (4 * mkp2 - 1))
    p[k, :m_max] = a * (x * p[k - 1, :m_max] - b * p[k - 2, :m_max])
  return p


def evaluate(n_m: int,
             n_l: int,
             x: np.ndarray) -> np.ndarray:
  """Associated Legendre Function.

  Evaluates the associated Legendre functions on the nodes x.

  Arguments:
    n_m: number of m (azimuthal wavenumber) modes. Must satisfy `n_m <= n_l`
    n_l: number of l (total wavenumber) modes.
    x: vector of nodes.

  Returns:
    Array p of shape (n_m, len(x), n_l) such that

      p[m, i, l] = cₗₘ Pᵐₗ(xᵢ) .

    Note p[m, i, l] = 0 for l < m. The normalization constants cₗₘ are chosen
    such that each basis function has unit L²([-1, 1]) norm.

  Raises:
    ValueError: if `n_m > n_l`.
  """
  if n_m > n_l:
    raise ValueError(f'Expected n_m <= n_l; got n_m = {n_m} and n_l = {n_l}.')
  r = np.transpose(
      _evaluate_rhombus(n_l=n_l, n_m=n_m, x=x, truncation='triangle'),
      (1, 2, 0))
  p = np.zeros((n_m, len(x), n_l))
  for m in range(n_m):
    p[m, :, m:n_l] = r[m, :, 0:n_l - m]
  return p


def gauss_legendre_nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
  """Returns nodes and weights for Gauss-Legendre quadrature.

  Args:
    n: the number of nodes and weights to return.

  Returns:
    A pair (nodes, weights); values and weights to use for Gauss-Legendre
    quadrature, with dtype `np.float64`.
  """
  return sps.roots_legendre(n)


@functools.lru_cache(maxsize=128)
def equiangular_nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
  """Returns equally spaced nodes and associated weights.

  The nodes can be interpreted as the midpoints of `n` equal sized segments. So,
  for `n = 3`, we would get `x = sin(y)` where
  `y = [-π / 3, 0, π / 3]`.

  Args:
    n: the number of nodes and weights to return.

  Returns:
    A pair (nodes, weights); values and weights to use for quadrature, with
    dtype `np.float64`.
  """
  spacing = np.pi / n
  theta = np.linspace(-np.pi / 2 + spacing / 2,
                      np.pi / 2 - spacing / 2,
                      n)
  x = np.sin(theta)
  w = _compute_weights(x)
  return x, w


@functools.lru_cache(maxsize=128)
def equiangular_nodes_with_poles(n: int) -> tuple[np.ndarray, np.ndarray]:
  """Returns equally spaced nodes and associated weights.

  The nodes equally spaced sample points between [-π/2, π/2] so for
  for `n = 3`, we would get `x = sin(y)` where `y = [-π / 2, 0, π / 2]`.

  Args:
    n: the number of nodes and weights to return.

  Returns:
    A pair (nodes, weights); values and weights to use for quadrature, with
    dtype `np.float64`.
  """
  theta = np.linspace(-np.pi / 2, np.pi / 2, n)
  x = np.sin(theta)
  w = _compute_weights(x)
  return x, w


def _compute_weights(x: np.ndarray) -> np.ndarray:
  """Computes weights for a vector of `x`."""
  # Letting Pₖ be the Legendre polynomial of degree k, we note that
  # ∫ Pₖ = 0 for all k > 0. Therefore, our weights `w` should satisfy
  # `w.dot(Pₖ(x)) = 𝛿(k, 0)`. So, we can find weights `w` using a linear solve.

  # The matrix `legendre` has `legendre[k, j] = Pₖ(x[j])`.
  legendre = evaluate(n_m=1, n_l=x.shape[0], x=x)[0].T
  z = np.zeros_like(x)
  z[0] = 1
  w = np.linalg.solve(legendre, z)

  # Since `x` spans the interval [-1, 1], the weights should sum to 2 so that we
  # get the right result when integrating `f(x) = 1` on the interval [-1, 1].
  return w / w.sum() * 2
