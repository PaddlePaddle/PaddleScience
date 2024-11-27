import math

import numpy as np
import paddle
import tensorly as tl
from tensorly import tenalg
from tensorly.decomposition import parafac
from tensorly.decomposition import tensor_train
from tensorly.decomposition import tucker

from ..utils import FactorList
from .core import FactorizedTensor

tl.set_backend("paddle")


# Author: Jean Kossaifi
# License: BSD 3 clause


class DenseTensor(FactorizedTensor, name="Dense"):
    """Dense tensor"""

    def __init__(self, tensor, shape=None, rank=None):
        super().__init__()
        if shape is not None and rank is not None:
            self.shape, self.rank = shape, rank
        else:
            self.shape = tensor.shape
            self.rank = None
        self.order = len(self.shape)
        if isinstance(tensor, paddle.base.framework.EagerParamBase):
            self.add_parameter("tensor", tensor)
        else:
            self.register_buffer("tensor", tensor)

    @classmethod
    def new(cls, shape, rank=None, device=None, dtype=None, **kwargs):
        # Register the parameters
        tensor = paddle.base.framework.EagerParamBase.from_tensor(
            paddle.empty(shape, dtype=dtype)
        )

        return cls(tensor)

    @classmethod
    def from_tensor(cls, tensor, rank="same", **kwargs):
        return cls(paddle.base.framework.EagerParamBase.from_tensor(tl.copy(tensor)))

    def init_from_tensor(self, tensor, l2_reg=1e-5, **kwargs):
        with paddle.no_grad():
            self.tensor = paddle.base.framework.EagerParamBase.from_tensor(
                tl.copy(tensor)
            )
        return self

    @property
    def decomposition(self):
        return self.tensor

    def to_tensor(self):
        return self.tensor

    def normal_(self, mean=0, std=1):
        with paddle.no_grad():
            self.tensor.data.normal_(mean, std)
        return self

    def __getitem__(self, indices):
        # slice(None, ...) is not supported on paddle
        tensor_temp = self.tensor
        axes = [i for i in range(len(indices))]
        starts = [0 if i.start is None else i.start for i in indices]
        ends = [
            tensor_temp.shape[i] if indices[i].stop is None else indices[i].stop
            for i in range(len(indices))
        ]
        target_tensor = paddle.slice(tensor_temp, axes=axes, starts=starts, ends=ends)
        return self.__class__(target_tensor)


class CPTensor(FactorizedTensor, name="CP"):
    """CP Factorization

    Parameters
    ----------
    weights
    factors
    shape
    rank
    """

    def __init__(self, weights, factors, shape=None, rank=None):
        super().__init__()
        if shape is not None and rank is not None:
            self.shape, self.rank = shape, rank
        else:
            self.shape, self.rank = tl.cp_tensor._validate_cp_tensor((weights, factors))
        self.order = len(self.shape)

        # self.weights = weights
        if isinstance(weights, paddle.base.framework.EagerParamBase):
            self.add_parameter("weights", weights)
        else:
            self.register_buffer("weights", weights)

        self.factors = FactorList(factors)

    @classmethod
    def new(cls, shape, rank, device=None, dtype=None, **kwargs):
        rank = tl.cp_tensor.validate_cp_rank(shape, rank)

        # Register the parameters
        weights = paddle.base.framework.EagerParamBase.from_tensor(
            paddle.empty([rank], dtype=dtype)
        )
        # Avoid the issues with ParameterList
        factors = [
            paddle.base.framework.EagerParamBase.from_tensor(
                paddle.empty((s, rank), dtype=dtype)
            )
            for s in shape
        ]

        return cls(weights, factors)

    @classmethod
    def from_tensor(cls, tensor, rank="same", **kwargs):
        shape = tensor.shape
        rank = tl.cp_tensor.validate_cp_rank(shape, rank)
        dtype = tensor.dtype

        with paddle.no_grad():
            weights, factors = parafac(tensor.to(paddle.float64), rank, **kwargs)

        return cls(
            paddle.base.framework.EagerParamBase.from_tensor(weights.to(dtype)),
            [
                paddle.base.framework.EagerParamBase.from_tensor(f.to(dtype))
                for f in factors
            ],
        )

    def init_from_tensor(self, tensor, l2_reg=1e-5, **kwargs):
        with paddle.no_grad():
            weights, factors = parafac(tensor, self.rank, l2_reg=l2_reg, **kwargs)

        self.weights = paddle.base.framework.EagerParamBase.from_tensor(weights)
        self.factors = FactorList(
            [paddle.base.framework.EagerParamBase.from_tensor(f) for f in factors]
        )
        return self

    @property
    def decomposition(self):
        return self.weights, self.factors

    def to_tensor(self):
        return tl.cp_to_tensor(self.decomposition)

    def normal_(self, mean=0, std=1):
        super().normal_(mean, std)
        std_factors = (std / math.sqrt(self.rank)) ** (1 / self.order)

        with paddle.no_grad():
            self.weights.fill_(1)
            for factor in self.factors:
                # must use develop branch!!!
                factor.data.normal_(0, std_factors)
        return self

    def __getitem__(self, indices):
        if isinstance(indices, int):
            # Select one dimension of one mode
            mixing_factor, *factors = self.factors
            weights = self.weights * mixing_factor[indices, :]
            return self.__class__(weights, factors)

        elif isinstance(indices, slice):
            # Index part of a factor
            mixing_factor, *factors = self.factors
            factors = [mixing_factor[indices, :], *factors]
            weights = self.weights
            return self.__class__(weights, factors)

        else:
            # Index multiple dimensions
            factors = self.factors
            index_factors = []
            weights = self.weights
            for index in indices:
                if index is Ellipsis:
                    raise ValueError(
                        f"Ellipsis is not yet supported, yet got indices={indices} which contains one."
                    )

                mixing_factor, *factors = factors
                if isinstance(index, (np.integer, int)):
                    if factors or index_factors:
                        weights = weights * mixing_factor[index, :]
                    else:
                        # No factors left
                        return tl.sum(weights * mixing_factor[index, :])
                else:
                    index_factors.append(mixing_factor[index, :])

            return self.__class__(weights, index_factors + factors)
        # return self.__class__(*tl.cp_indexing(self.weights, self.factors, indices))

    def transduct(self, new_dim, mode=0, new_factor=None):
        """Transduction adds a new dimension to the existing factorization

        Parameters
        ----------
        new_dim : int
            dimension of the new mode to add
        mode : where to insert the new dimension, after the channels, default is 0
            by default, insert the new dimensions before the existing ones
            (e.g. add time before height and width)

        Returns
        -------
        self
        """
        factors = self.factors
        # Important: don't increment the order before accessing factors which uses order!
        self.order += 1
        self.shape = self.shape[:mode] + (new_dim,) + self.shape[mode:]

        if new_factor is None:
            new_factor = paddle.ones([new_dim], self.rank)  # new_dim

        factors.insert(
            mode,
            paddle.base.framework.EagerParamBase.from_tensor(new_factor.to(factors[0])),
        )
        self.factors = FactorList(factors)

        return self


class TuckerTensor(FactorizedTensor, name="Tucker"):
    """Tucker Factorization

    Parameters
    ----------
    core
    factors
    shape
    rank
    """

    def __init__(self, core, factors, shape=None, rank=None):
        super().__init__()
        if shape is not None and rank is not None:
            self.shape, self.rank = shape, rank
        else:
            self.shape, self.rank = tl.tucker_tensor._validate_tucker_tensor(
                (core, factors)
            )

        self.order = len(self.shape)
        # self.core = core
        if isinstance(core, paddle.base.framework.EagerParamBase):
            self.add_parameter("core", core)
        else:
            self.register_buffer("core", core)

        self.factors = FactorList(factors)

    @classmethod
    def new(cls, shape, rank, fixed_rank_modes=None, device=None, dtype=None, **kwargs):
        rank = tl.tucker_tensor.validate_tucker_rank(
            shape, rank, fixed_modes=fixed_rank_modes
        )

        # Register the parameters
        core = paddle.base.framework.EagerParamBase.from_tensor(
            paddle.empty(rank, dtype=dtype)
        )
        # Avoid the issues with ParameterList
        factors = [
            paddle.base.framework.EagerParamBase.from_tensor(
                paddle.empty((s, r), dtype=dtype)
            )
            for (s, r) in zip(shape, rank)
        ]

        return cls(core, factors)

    @classmethod
    def from_tensor(cls, tensor, rank="same", fixed_rank_modes=None, **kwargs):
        shape = tensor.shape
        rank = tl.tucker_tensor.validate_tucker_rank(
            shape, rank, fixed_modes=fixed_rank_modes
        )

        with paddle.no_grad():
            core, factors = tucker(tensor, rank, **kwargs)

        return cls(
            paddle.base.framework.EagerParamBase.from_tensor(core),
            [paddle.base.framework.EagerParamBase.from_tensor(f) for f in factors],
        )

    def init_from_tensor(
        self, tensor, unsqueezed_modes=None, unsqueezed_init="average", **kwargs
    ):
        """Initialize the tensor factorization from a tensor

        Parameters
        ----------
        tensor : torch.Tensor
            full tensor to decompose
        unsqueezed_modes : int list
            list of modes for which the rank is 1 that don't correspond to a mode in the full tensor
            essentially we are adding a new dimension for which the core has dim 1,
            and that is not initialized through decomposition.
            Instead first `tensor` is decomposed into the other factors.
            The `unsqueezed factors` are then added and  initialized e.g. with 1/dim[i]
        unsqueezed_init : 'average' or float
            if unsqueezed_modes, this is how the added "unsqueezed" factors will be initialized
            if 'average', then unsqueezed_factor[i] will have value 1/tensor.shape[i]
        """
        if unsqueezed_modes is not None:
            unsqueezed_modes = sorted(unsqueezed_modes)
            for mode in unsqueezed_modes[::-1]:
                if self.rank[mode] != 1:
                    msg = "It is only possible to initialize by averagig over mode for which rank=1."
                    msg += f"However, got unsqueezed_modes={unsqueezed_modes} but rank[{mode}]={self.rank[mode]} != 1."
                    raise ValueError(msg)

            rank = tuple(
                r for (i, r) in enumerate(self.rank) if i not in unsqueezed_modes
            )
        else:
            rank = self.rank

        with paddle.no_grad():
            core, factors = tucker(tensor, rank, **kwargs)

            if unsqueezed_modes is not None:
                # Initialise with 1/shape[mode] or given value
                for mode in unsqueezed_modes:
                    size = self.shape[mode]
                    factor = paddle.ones(size, 1)
                    if unsqueezed_init == "average":
                        factor /= size
                    else:
                        factor *= unsqueezed_init
                    factors.insert(mode, factor)
                    core = core.unsqueeze(mode)

        self.core = paddle.base.framework.EagerParamBase.from_tensor(core)
        self.factors = FactorList(
            [paddle.base.framework.EagerParamBase.from_tensor(f) for f in factors]
        )
        return self

    @property
    def decomposition(self):
        return self.core, self.factors

    def to_tensor(self):
        return tl.tucker_to_tensor(self.decomposition)

    def normal_(self, mean=0, std=1):
        if mean != 0:
            raise ValueError(f"Currently only mean=0 is supported, but got mean={mean}")

        r = np.prod([math.sqrt(r) for r in self.rank])
        std_factors = (std / r) ** (1 / (self.order + 1))

        with paddle.no_grad():
            self.core.data.normal_(0, std_factors)
            for factor in self.factors:
                factor.data.normal_(0, std_factors)
        return self

    def __getitem__(self, indices):
        if isinstance(indices, int):
            # Select one dimension of one mode
            mixing_factor, *factors = self.factors
            core = tenalg.mode_dot(self.core, mixing_factor[indices, :], 0)
            return self.__class__(core, factors)

        elif isinstance(indices, slice):
            mixing_factor, *factors = self.factors
            factors = [mixing_factor[indices, :], *factors]
            return self.__class__(self.core, factors)

        else:
            # Index multiple dimensions
            modes = []
            factors = []
            factors_contract = []
            for i, (index, factor) in enumerate(zip(indices, self.factors)):
                if index is Ellipsis:
                    raise ValueError(
                        f"Ellipsis is not yet supported, yet got indices={indices}, indices[{i}]={index}."
                    )
                if isinstance(index, int):
                    modes.append(i)
                    factors_contract.append(factor[index, :])
                else:
                    factors.append(factor[index, :])

            if modes:
                core = tenalg.multi_mode_dot(self.core, factors_contract, modes=modes)
            else:
                core = self.core
            factors = factors + self.factors[i + 1 :]

            if factors:
                return self.__class__(core, factors)

            # Fully contracted tensor
            return core


class TTTensor(FactorizedTensor, name="TT"):
    """Tensor-Train (Matrix-Product-State) Factorization

    Parameters
    ----------
    factors
    shape
    rank
    """

    def __init__(self, factors, shape=None, rank=None):
        super().__init__()
        if shape is None or rank is None:
            self.shape, self.rank = tl.tt_tensor._validate_tt_tensor(factors)
        else:
            self.shape, self.rank = shape, rank

        self.order = len(self.shape)
        self.factors = FactorList(factors)

    @classmethod
    def new(cls, shape, rank, device=None, dtype=None, **kwargs):
        rank = tl.tt_tensor.validate_tt_rank(shape, rank)

        # Avoid the issues with ParameterList
        factors = [
            paddle.base.framework.EagerParamBase.from_tensor(
                paddle.empty((rank[i], s, rank[i + 1]), dtype=dtype)
            )
            for i, s in enumerate(shape)
        ]

        return cls(factors)

    @classmethod
    def from_tensor(cls, tensor, rank="same", **kwargs):
        shape = tensor.shape
        rank = tl.tt_tensor.validate_tt_rank(shape, rank)

        with paddle.no_grad():
            # TODO: deal properly with wrong kwargs
            factors = tensor_train(tensor, rank)

        return cls(
            [paddle.base.framework.EagerParamBase.from_tensor(f) for f in factors]
        )

    def init_from_tensor(self, tensor, **kwargs):
        with paddle.no_grad():
            # TODO: deal properly with wrong kwargs
            factors = tensor_train(tensor, self.rank)

        self.factors = FactorList(
            [paddle.base.framework.EagerParamBase.from_tensor(f) for f in factors]
        )
        self.rank = tuple([f.shape[0] for f in factors] + [1])
        return self

    @property
    def decomposition(self):
        return self.factors

    def to_tensor(self):
        return tl.tt_to_tensor(self.decomposition)

    def normal_(self, mean=0, std=1):
        if mean != 0:
            raise ValueError(f"Currently only mean=0 is supported, but got mean={mean}")

        r = np.prod(self.rank)
        std_factors = (std / r) ** (1 / self.order)
        with paddle.no_grad():
            for factor in self.factors:
                factor.data.normal_(0, std_factors)
        return self

    def __getitem__(self, indices):
        if isinstance(indices, int):
            # Select one dimension of one mode
            factor, next_factor, *factors = self.factors
            next_factor = tenalg.mode_dot(
                next_factor, factor[:, indices, :].squeeze(1), 0
            )
            return self.__class__([next_factor, *factors])

        elif isinstance(indices, slice):
            mixing_factor, *factors = self.factors
            factors = [mixing_factor[:, indices], *factors]
            return self.__class__(factors)

        else:
            factors = []
            all_contracted = True
            for i, index in enumerate(indices):
                if index is Ellipsis:
                    raise ValueError(
                        f"Ellipsis is not yet supported, yet got indices={indices}, indices[{i}]={index}."
                    )
                if isinstance(index, int):
                    if i:
                        factor = tenalg.mode_dot(
                            factor, self.factors[i][:, index, :].T, -1
                        )
                    else:
                        factor = self.factors[i][:, index, :]
                else:
                    if i:
                        if all_contracted:
                            factor = tenalg.mode_dot(
                                self.factors[i][:, index, :], factor, 0
                            )
                        else:
                            factors.append(factor)
                            factor = self.factors[i][:, index, :]
                    else:
                        factor = self.factors[i][:, index, :]
                    all_contracted = False

            if factor.ndim == 2:  # We have contracted all cores, so have a 2D matrix
                if self.order == (i + 1):
                    # No factors left
                    return factor.squeeze()
                else:
                    next_factor, *factors = self.factors[i + 1 :]
                    factor = tenalg.mode_dot(next_factor, factor, 0)
                    return self.__class__([factor, *factors])
            else:
                return self.__class__([*factors, factor, *self.factors[i + 1 :]])

    def transduct(self, new_dim, mode=0, new_factor=None):
        """Transduction adds a new dimension to the existing factorization

        Parameters
        ----------
        new_dim : int
            dimension of the new mode to add
        mode : where to insert the new dimension, after the channels, default is 0
            by default, insert the new dimensions before the existing ones
            (e.g. add time before height and width)

        Returns
        -------
        self
        """
        factors = self.factors

        # Important: don't increment the order before accessing factors which uses order!
        self.order += 1
        new_rank = self.rank[mode]
        self.rank = self.rank[:mode] + (new_rank,) + self.rank[mode:]
        self.shape = self.shape[:mode] + (new_dim,) + self.shape[mode:]

        # Init so the reconstruction is equivalent to concatenating the previous self new_dim times
        if new_factor is None:
            new_factor = paddle.zeros(new_rank, new_dim, new_rank)
            for i in range(new_dim):
                new_factor[:, i, :] = paddle.eye(new_rank)  # /new_dim
            # Below: <=> static prediciton
            # new_factor[:, new_dim//2, :] = torch.eye(new_rank)

        factors.insert(
            mode,
            paddle.base.framework.EagerParamBase.from_tensor(new_factor),
        )
        self.factors = FactorList(factors)

        return self
