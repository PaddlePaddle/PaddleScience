r"""Basic building blocks and (sub-)networks of learned image registration models.

For spatial transformation models used in both classic non-learning based registration and
learned image registration, where transformation parameters are inferred by a neural network
from the input data instead of being optimized directly, see the :mod:`.spatial` library.

Commonly, the neural network model infers the transformation parameters from the input data.
These parameters are then used to evaluate and apply the spatial transformation. For this, the
``params`` attribute of parameteric transformations can be set to a neural network instance of
type ``torch.nn.Module``. The input data of the so parametrized spatial transformation is then
set as :meth:`.SpatialTransform.condition`, which constitutes the input of the neural network.

"""
