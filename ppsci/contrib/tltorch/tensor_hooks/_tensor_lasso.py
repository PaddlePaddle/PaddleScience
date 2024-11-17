import warnings

import paddle
import tensorly as tl
from paddle.nn import functional as F

from ..factorized_tensors import CPTensor
from ..factorized_tensors import TTTensor
from ..factorized_tensors import TuckerTensor
from ..utils import ParameterList

tl.set_backend("paddle")


# Author: Jean Kossaifi
# License: BSD 3 clause


class TensorLasso:
    """Generalized Tensor Lasso on factorized tensors

        Applies a generalized Lasso (l1 regularization) on a factorized tensor.


    Parameters
    ----------
    penalty : float, default is 0.01
        scaling factor for the loss

    clamp_weights : bool, default is True
        if True, the lasso weights are clamp between -1 and 1

    threshold : float, default is 1e-6
        if a lasso weight is lower than the set threshold, it is set to 0

    normalize_loss : bool, default is True
        If True, the loss will be between 0 and 1.
        Otherwise, the raw sum of absolute weights will be returned.

    Examples
    --------

    First you need to create an instance of the regularizer:

    >>> regularizer = tensor_lasso(factorization='cp')

    You can apply the regularizer to one or several layers:

    >>> trl = TRL((5, 5), (5, 5), rank='same')
    >>> trl2 = TRL((5, 5), (2, ), rank='same')
    >>> regularizer.apply(trl.weight)
    >>> regularizer.apply(trl2.weight)

    The lasso is automatically applied:

    >>> x = trl(x)
    >>> pred = trl2(x)
    >>> loss = your_loss_function(pred)

    Add the Lasso loss:

    >>> loss = loss + regularizer.loss

    You can now backpropagate through your loss as usual:

    >>> loss.backwards()

    After you finish updating the weights, don't forget to reset the regularizer,
    otherwise it will keep accumulating values!

    >>> loss.reset()

    You can also remove the regularizer with `regularizer.remove(trl)`.
    """

    _factorizations = dict()

    def __init_subclass__(cls, factorization, **kwargs):
        """When a subclass is created, register it in _factorizations"""
        cls._factorizations[factorization.__name__] = cls

    def __init__(
        self, penalty=0.01, clamp_weights=True, threshold=1e-6, normalize_loss=True
    ):
        self.penalty = penalty
        self.clamp_weights = clamp_weights
        self.threshold = threshold
        self.normalize_loss = normalize_loss

        # Initialize the counters
        self.reset()

    def reset(self):
        """Reset the loss, should be called at the end of each iteration."""
        self._loss = 0
        self.n_element = 0

    @property
    def loss(self):
        """Returns the current Lasso (l1) loss for the layers that have been called so far.

        Returns
        -------
        float
            l1 regularization on the tensor layers the regularization has been applied to.
        """
        if self.n_element == 0:
            warnings.warn("The L1Regularization was not applied to any weights.")
            return 0
        elif self.normalize_loss:
            return self.penalty * self._loss / self.n_element
        else:
            return self.penalty * self._loss

    def __call__(self, module, input, tucker_tensor):
        raise NotImplementedError

    def apply_lasso(self, tucker_tensor, lasso_weights):
        """Applies the lasso to a decomposed tensor"""
        raise NotImplementedError

    @classmethod
    def from_factorization(
        cls,
        factorization,
        penalty=0.01,
        clamp_weights=True,
        threshold=1e-6,
        normalize_loss=True,
    ):
        return cls.from_factorization_name(
            factorization.__class__.__name__,
            penalty=penalty,
            clamp_weights=clamp_weights,
            threshold=threshold,
            normalize_loss=normalize_loss,
        )

    @classmethod
    def from_factorization_name(
        cls,
        factorization_name,
        penalty=0.01,
        clamp_weights=True,
        threshold=1e-6,
        normalize_loss=True,
    ):
        cls = cls._factorizations[factorization_name]
        lasso = cls(
            penalty=penalty,
            clamp_weights=clamp_weights,
            threshold=threshold,
            normalize_loss=normalize_loss,
        )
        return lasso

    def remove(self, module):
        raise NotImplementedError


class CPLasso(TensorLasso, factorization=CPTensor):
    """Decomposition Hook for Tensor Lasso on CP tensors

    Parameters
    ----------
    penalty : float, default is 0.01
        scaling factor for the loss

    clamp_weights : bool, default is True
        if True, the lasso weights are clamp between -1 and 1

    threshold : float, default is 1e-6
        if a lasso weight is lower than the set threshold, it is set to 0

    normalize_loss : bool, default is True
        If True, the loss will be between 0 and 1.
        Otherwise, the raw sum of absolute weights will be returned.
    """

    def __call__(self, module, input, cp_tensor):
        """CP already includes weights, we'll just take their l1 norm"""
        weights = getattr(module, "lasso_weights")

        with paddle.no_grad():
            if self.clamp_weights:
                weights.data = paddle.clamp(weights.data, -1, 1)
                setattr(module, "lasso_weights", weights)

            if self.threshold:
                weights.data = F.threshold(
                    weights.data, threshold=self.threshold, value=0, inplace=True
                )
                setattr(module, "lasso_weights", weights)

        self.n_element += weights.numel()
        self._loss = self._loss + self.penalty * paddle.norm(weights, 1)
        return cp_tensor

    def apply(self, module):
        """Apply an instance of the L1Regularizer to a tensor module

        Parameters
        ----------
        module : TensorModule
            module on which to add the regularization

        Returns
        -------
        TensorModule (with Regularization hook)
        """
        context = tl.context(module.factors[0])
        lasso_weights = paddle.base.framework.EagerParamBase.from_tensor(
            paddle.ones(module.rank, **context)
        )
        setattr(module, "lasso_weights", lasso_weights)

        module.register_forward_hook(self)
        return module

    def remove(self, module):
        delattr(module, "lasso_weights")

    def set_weights(self, module, value):
        with paddle.no_grad():
            module.lasso_weights.data.fill_(value)


class TuckerLasso(TensorLasso, factorization=TuckerTensor):
    """Decomposition Hook for Tensor Lasso on Tucker tensors

        Applies a generalized Lasso (l1 regularization) on the tensor layers the regularization it is applied to.


    Parameters
    ----------
    penalty : float, default is 0.01
        scaling factor for the loss

    clamp_weights : bool, default is True
        if True, the lasso weights are clamp between -1 and 1

    threshold : float, default is 1e-6
        if a lasso weight is lower than the set threshold, it is set to 0

    normalize_loss : bool, default is True
        If True, the loss will be between 0 and 1.
        Otherwise, the raw sum of absolute weights will be returned.
    """

    _log = []

    def __call__(self, module, input, tucker_tensor):
        lasso_weights = getattr(module, "lasso_weights")
        order = len(lasso_weights)

        with paddle.no_grad():
            for i in range(order):
                if self.clamp_weights:
                    lasso_weights[i].data = paddle.clamp(lasso_weights[i].data, -1, 1)

                if self.threshold:
                    lasso_weights[i] = F.threshold(
                        lasso_weights[i],
                        threshold=self.threshold,
                        value=0,
                        inplace=True,
                    )

            setattr(module, "lasso_weights", lasso_weights)

        for weight in lasso_weights:
            self.n_element += weight.numel()
            self._loss = self._loss + paddle.sum(paddle.abs(weight))

        return self.apply_lasso(tucker_tensor, lasso_weights)

    def apply_lasso(self, tucker_tensor, lasso_weights):
        """Applies the lasso to a decomposed tensor"""
        factors = tucker_tensor.factors
        factors = [factor * w for (factor, w) in zip(factors, lasso_weights)]
        return TuckerTensor(tucker_tensor.core, factors)

    def apply(self, module):
        """Apply an instance of the L1Regularizer to a tensor module

        Parameters
        ----------
        module : TensorModule
            module on which to add the regularization

        Returns
        -------
        TensorModule (with Regularization hook)
        """
        rank = module.rank
        context = tl.context(module.core)
        lasso_weights = ParameterList(
            [
                paddle.base.framework.EagerParamBase.from_tensor(
                    paddle.ones(r, **context)
                )
                for r in rank
            ]
        )
        setattr(module, "lasso_weights", lasso_weights)
        module.register_forward_hook(self)

        return module

    def remove(self, module):
        delattr(module, "lasso_weights")

    def set_weights(self, module, value):
        with paddle.no_grad():
            for weight in module.lasso_weights:
                weight.data.fill_(value)


class TTLasso(TensorLasso, factorization=TTTensor):
    """Decomposition Hook for Tensor Lasso on TT tensors

    Parameters
    ----------
    penalty : float, default is 0.01
        scaling factor for the loss

    clamp_weights : bool, default is True
        if True, the lasso weights are clamp between -1 and 1

    threshold : float, default is 1e-6
        if a lasso weight is lower than the set threshold, it is set to 0

    normalize_loss : bool, default is True
        If True, the loss will be between 0 and 1.
        Otherwise, the raw sum of absolute weights will be returned.
    """

    def __call__(self, module, input, tt_tensor):
        lasso_weights = getattr(module, "lasso_weights")
        order = len(lasso_weights)

        with paddle.no_grad():
            for i in range(order):
                if self.clamp_weights:
                    lasso_weights[i].data = paddle.clamp(lasso_weights[i].data, -1, 1)

                if self.threshold:
                    lasso_weights[i] = F.threshold(
                        lasso_weights[i],
                        threshold=self.threshold,
                        value=0,
                        inplace=True,
                    )

            setattr(module, "lasso_weights", lasso_weights)

        for weight in lasso_weights:
            self.n_element += weight.numel()
            self._loss = self._loss + paddle.sum(paddle.abs(weight))

        return self.apply_lasso(tt_tensor, lasso_weights)

    def apply_lasso(self, tt_tensor, lasso_weights):
        """Applies the lasso to a decomposed tensor"""
        factors = tt_tensor.factors
        factors = [factor * w for (factor, w) in zip(factors, lasso_weights)] + [
            factors[-1]
        ]
        return TTTensor(factors)

    def apply(self, module):
        """Apply an instance of the L1Regularizer to a tensor module

        Parameters
        ----------
        module : TensorModule
            module on which to add the regularization

        Returns
        -------
        TensorModule (with Regularization hook)
        """
        rank = module.rank[1:-1]
        lasso_weights = ParameterList(
            [
                paddle.base.framework.EagerParamBase.from_tensor(paddle.ones([1, 1, r]))
                for r in rank
            ]
        )
        setattr(module, "lasso_weights", lasso_weights)
        # handle = module.register_forward_hook(self)
        return module

    def remove(self, module):
        """Remove the Regularization from a module."""
        delattr(module, "lasso_weights")

    def set_weights(self, module, value):
        with paddle.no_grad():
            for weight in module.lasso_weights:
                weight.data.fill_(value)


def tensor_lasso(
    factorization="CP",
    penalty=0.01,
    clamp_weights=True,
    threshold=1e-6,
    normalize_loss=True,
):
    """Generalized Tensor Lasso from a factorized tensors

        Applies a generalized Lasso (l1 regularization) on a factorized tensor.


    Parameters
    ----------
    factorization : str

    penalty : float, default is 0.01
        scaling factor for the loss

    clamp_weights : bool, default is True
        if True, the lasso weights are clamp between -1 and 1

    threshold : float, default is 1e-6
        if a lasso weight is lower than the set threshold, it is set to 0

    normalize_loss : bool, default is True
        If True, the loss will be between 0 and 1.
        Otherwise, the raw sum of absolute weights will be returned.

    Examples
    --------

    Let's say you have a set of factorized (here, CP) tensors:

    >>> tensor = FactorizedTensor.new((3, 4, 2), rank='same', factorization='CP').normal_()
    >>> tensor2 = FactorizedTensor.new((5, 6, 7), rank=0.5, factorization='CP').normal_()

    First you need to create an instance of the regularizer:

    >>> regularizer = TensorLasso(factorization='cp', penalty=penalty)

    You can apply the regularizer to one or several layers:

    >>> regularizer.apply(tensor)
    >>> regularizer.apply(tensor2)

    The lasso is automatically applied:

    >>> sum = torch.sum(tensor() + tensor2())

    You can access the Lasso loss from your instance:

    >>> l1_loss = regularizer.loss

    You can optimize and backpropagate through your loss as usual.

    After you finish updating the weights, don't forget to reset the regularizer,
    otherwise it will keep accumulating values!

    >>> regularizer.reset()

    You can also remove the regularizer with `regularizer.remove(tensor)`,
    or `remove_tensor_lasso(tensor)`.
    """
    factorization = factorization.lower()
    mapping = dict(cp="CPTensor", tucker="TuckerTensor", tt="TTTensor")
    return TensorLasso.from_factorization_name(
        mapping[factorization],
        penalty=penalty,
        clamp_weights=clamp_weights,
        threshold=threshold,
        normalize_loss=normalize_loss,
    )


def remove_tensor_lasso(factorized_tensor):
    """Removes the tensor lasso from a TensorModule

    Parameters
    ----------
    factorized_tensor : tltorch.FactorizedTensor
        the tensor module parametrized by the tensor decomposition to which to apply tensor dropout

    Examples
    --------
    >>> tensor = FactorizedTensor.new((3, 4, 2), rank=0.5, factorization='CP').normal_()
    >>> tensor = tensor_lasso(tensor, p=0.5)
    >>> remove_tensor_lasso(tensor)
    """
    for key, hook in factorized_tensor._forward_hooks.items():
        if isinstance(hook, TensorLasso):
            hook.remove(factorized_tensor)
            del factorized_tensor._forward_hooks[key]
            return factorized_tensor

    raise ValueError(f"TensorLasso not found in factorized tensor {factorized_tensor}")
