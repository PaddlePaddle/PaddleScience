import re
from collections import defaultdict
from typing import Dict
from typing import Generator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

import paddle
from deepali.core import Grid
from deepali.core import PaddingMode
from deepali.core import Sampling
from deepali.core import functional as U
from deepali.losses import BSplineLoss
from deepali.losses import DisplacementLoss
from deepali.losses import LandmarkPointDistance
from deepali.losses import PairwiseImageLoss
from deepali.losses import ParamsLoss
from deepali.losses import PointSetDistance
from deepali.losses import RegistrationLoss
from deepali.losses import RegistrationLosses
from deepali.losses import RegistrationResult
from deepali.modules import SampleImage
from deepali.spatial import BSplineTransform
from deepali.spatial import CompositeTransform
from deepali.spatial import SequentialTransform
from deepali.spatial import SpatialTransform
from paddle import Tensor
from paddle.nn import Layer

RE_WEIGHT = re.compile(
    "^((?P<mul>[0-9]+(\\.[0-9]+)?)\\s*[\\* ])?\\s*(?P<chn>[a-zA-Z0-9_-]+)\\s*(\\+\\s*(?P<add>[0-9]+(\\.[0-9]+)?))?$"
)
RE_TERM_VAR = re.compile("^[a-zA-Z0-9_-]+\\((?P<var>[a-zA-Z0-9_]+)\\)$")
TLayer = TypeVar("TLayer", bound=Layer)
TSpatialTransform = TypeVar("TSpatialTransform", bound=SpatialTransform)


class PairwiseImageRegistrationLoss(RegistrationLoss):
    """Loss function for pairwise multi-channel image registration."""

    def __init__(
        self,
        source_data: paddle.Tensor,
        target_data: paddle.Tensor,
        source_grid: Grid,
        target_grid: Grid,
        source_chns: Mapping[str, Union[int, Tuple[int, int]]],
        target_chns: Mapping[str, Union[int, Tuple[int, int]]],
        source_pset: Optional[paddle.Tensor] = None,
        target_pset: Optional[paddle.Tensor] = None,
        source_landmarks: Optional[paddle.Tensor] = None,
        target_landmarks: Optional[paddle.Tensor] = None,
        losses: Optional[RegistrationLosses] = None,
        weights: Mapping[str, Union[float, str]] = None,
        transform: Optional[Union[CompositeTransform, SpatialTransform]] = None,
        sampling: Union[Sampling, str] = Sampling.LINEAR,
    ):
        """Initialize multi-channel registration loss function.

        Args:
            source_data: Moving normalized multi-channel source image batch tensor.
            source_data: Fixed normalized multi-channel target image batch tensor.
            source_grid: Sampling grid of source image.
            source_grid: Sampling grid of target image.
            source_chns: Mapping from channel (loss, weight) name to index or range.
            target_chns: Mapping from channel (loss, weight) name to index or range.
            source_pset: Point sets defined with respect to source image grid.
            target_pset: Point sets defined with respect to target image grid.
            source_landmarks: Landmark points defined with respect to source image grid.
            target_landmarks: Landmark points defined with respect to target image grid.
            losses: Dictionary of named loss terms. Loss terms must be either a subclass of
                ``PairwiseImageLoss``, ``DisplacementLoss``, ``PointSetDistance``, ``ParamsLoss``,
                or ``paddle.nn.Layer``. In case of a ``PairwiseImageLoss``, the key (name) of the
                loss term must be found in ``channels`` which identifies the corresponding ``target``
                and ``source`` data channels that this loss term relates to. If the name is not found
                in the ``channels`` mapping, the loss term is called with all image channels as input.
                If a loss term is not an instance of a known registration loss type, it is assumed to be a
                regularization term without arguments, e.g., a ``paddle.nn.Layer`` which itself has a reference
                to the parameters of the transformation that it is based on.
            weights: Scalar weights of loss terms or name of channel with locally adaptive weights.
            transform: Spatial transformation to apply to ``source`` image.
            sampling: Image interpolation mode.

        """
        super().__init__()
        self.register_buffer(name="_source_data", tensor=source_data)
        self.register_buffer(name="_target_data", tensor=target_data)
        self.source_grid = source_grid
        self.target_grid = target_grid
        self.source_chns = dict(source_chns or {})
        self.target_chns = dict(target_chns or {})
        self.source_pset = source_pset
        self.target_pset = target_pset
        self.source_landmarks = source_landmarks
        self.target_landmarks = target_landmarks
        if transform is None:
            transform = SequentialTransform(self.target_grid)
        elif isinstance(transform, SpatialTransform):
            transform = SequentialTransform(transform)
        elif not isinstance(transform, CompositeTransform):
            raise TypeError(
                "PairwiseImageRegistrationLoss() 'transform' must be of type CompositeTransform"
            )
        self.transform = transform
        self._sample_image = SampleImage(
            target=self.target_grid,
            source=self.source_grid,
            sampling=sampling,
            padding=PaddingMode.ZEROS,
            align_centers=False,
        )
        points = self.target_grid.coords(device=self._target_data.place)
        self.register_buffer(name="grid_points", tensor=points.unsqueeze(axis=0))
        self.loss_terms = self.as_module_dict(losses)
        self.weights = dict(weights or {})

    @property
    def device(self) -> str:
        """Device on which loss is evaluated."""
        device = self._target_data.place
        assert isinstance(device, paddle.base.libpaddle.Place)
        return device

    def loss_terms_of_type(self, loss_type: Type[TLayer]) -> Dict[str, TLayer]:
        """Get dictionary of loss terms of a specifictype."""
        return {
            name: module
            for name, module in self.loss_terms.items()
            if isinstance(module, loss_type)
        }

    def transforms_of_type(
        self, transform_type: Type[TSpatialTransform]
    ) -> List[TSpatialTransform]:
        """Get list of spatial transformations of a specific type."""

        def _iter_transforms(transform) -> Generator[SpatialTransform, None, None]:
            if isinstance(transform, transform_type):
                yield transform
            elif isinstance(transform, CompositeTransform):
                for t in transform.transforms():
                    yield from _iter_transforms(t)

        transforms = list(_iter_transforms(self.transform))
        return transforms

    @property
    def has_transform(self) -> bool:
        """Whether a spatial transformation is set."""
        return len(self.transform) > 0

    def target_data(self) -> paddle.Tensor:
        """Target image tensor."""
        data = self._target_data
        assert isinstance(data, paddle.Tensor)
        return data

    def source_data(self, grid: Optional[paddle.Tensor] = None) -> paddle.Tensor:
        """Sample source image at transformed target grid points."""
        data = self._source_data
        assert isinstance(data, paddle.Tensor)
        if grid is None:
            return data
        return self._sample_image(grid, data)

    def data_mask(
        self, data: paddle.Tensor, channels: Dict[str, Union[int, Tuple[int, int]]]
    ) -> paddle.Tensor:
        """Get boolean mask from data tensor."""
        slice_ = self.as_slice(channels["msk"])
        start, stop = slice_.start, slice_.stop
        start_0 = data.shape[1] + start if start < 0 else start
        mask = paddle.slice(data, [1], [start_0], [start_0 + (stop - start)])
        return mask > 0.9

    def overlap_mask(
        self, source: paddle.Tensor, target: paddle.Tensor
    ) -> Optional[paddle.Tensor]:
        """Overlap mask at which to evaluate pairwise data term."""
        mask = self.data_mask(source, self.source_chns)
        mask &= self.data_mask(target, self.target_chns)
        return mask

    @classmethod
    def as_slice(cls, arg: Union[int, Sequence[int]]) -> slice:
        """Slice of image data channels associated with the specified name."""
        if isinstance(arg, int):
            arg = (arg,)
        if len(arg) == 1:
            arg = arg[0], arg[0] + 1
        if len(arg) == 2:
            arg = arg[0], arg[1], 1
        if len(arg) != 3:
            raise ValueError(
                f"{cls.__name__}.as_slice() 'arg' must be int or sequence of length 1, 2, or 3"
            )
        return slice(*arg)

    @classmethod
    def data_channels(cls, data: paddle.Tensor, c: slice) -> paddle.Tensor:
        """Get subimage data tensor of named channel."""
        i = (slice(0, tuple(data.shape)[0]), c) + tuple(
            slice(0, tuple(data.shape)[dim]) for dim in range(2, data.ndim)
        )
        return data[i]

    def loss_input(
        self,
        name: str,
        data: paddle.Tensor,
        channels: Dict[str, Union[int, Tuple[int, int]]],
    ) -> paddle.Tensor:
        """Get input for named loss term."""
        if name in channels:
            c = channels[name]
        elif "img" not in channels:
            raise RuntimeError(
                f"Channels map contains neither entry for '{name}' nor 'img'"
            )
        else:
            c = channels["img"]
        i: slice = self.as_slice(c)
        return self.data_channels(data, i)

    def loss_mask(
        self,
        name: str,
        data: paddle.Tensor,
        channels: Dict[str, Union[int, Tuple[int, int]]],
        mask: paddle.Tensor,
    ) -> paddle.Tensor:
        """Get mask for named loss term."""
        weight = self.weights.get(name, 1.0)
        if not isinstance(weight, str):
            return mask
        match = RE_WEIGHT.match(weight)
        if match is None:
            raise RuntimeError(
                f"Invalid weight string ('{weight}') for loss term '{name}'"
            )
        chn = match.group("chn")
        mul = match.group("mul")
        add = match.group("add")
        c = channels.get(chn)
        if c is None:
            raise RuntimeError(
                f"Channels map contains no entry for '{name}' weight string '{weight}'"
            )
        i = self.as_slice(c)
        w = self.data_channels(data, i)
        if mul is not None:
            w = w * float(mul)
        if add is not None:
            w = w + float(add)
        return w * mask

    def eval(self) -> RegistrationResult:
        """Evaluate pairwise image registration loss."""
        result = {}
        losses = {}
        misc_excl = set()
        x: Tensor = self.grid_points
        y: Tensor = self.transform(x, grid=True)
        variables = defaultdict(list)
        for name, buf in self.transform.named_buffers():
            if not buf.stop_gradient:
                var = name.rsplit(".", 1)[-1]
                variables[var].append(buf)
        variables["w"] = [U.move_dim(y - x, -1, 1)]
        data_terms = self.loss_terms_of_type(PairwiseImageLoss)
        misc_excl |= set(data_terms.keys())
        if data_terms:
            source = self.source_data(y)
            target = self.target_data()
            mask = self.overlap_mask(source, target)
            for name, term in data_terms.items():
                s = self.loss_input(name, source, self.source_chns)
                t = self.loss_input(name, target, self.target_chns)
                m = self.loss_mask(name, target, self.target_chns, mask)
                losses[name] = term(s, t, mask=m)
            result["source"] = source
            result["target"] = target
            result["mask"] = mask
        dist_terms = self.loss_terms_of_type(PointSetDistance)
        misc_excl |= set(dist_terms.keys())
        ldist_terms = {
            k: v for k, v in dist_terms.items() if isinstance(v, LandmarkPointDistance)
        }
        dist_terms = {k: v for k, v in dist_terms.items() if k not in ldist_terms}
        if dist_terms:
            if self.source_pset is None:
                raise RuntimeError(f"{type(self).__name__}() missing source point set")
            if self.target_pset is None:
                raise RuntimeError(f"{type(self).__name__}() missing target point set")
            s = self.source_pset
            t = self.transform(self.target_pset)
            for name, term in dist_terms.items():
                losses[name] = term(t, s)
        if ldist_terms:
            if self.source_landmarks is None:
                raise RuntimeError(f"{type(self).__name__}() missing source landmarks")
            if self.target_landmarks is None:
                raise RuntimeError(f"{type(self).__name__}() missing target landmarks")
            s = self.source_landmarks
            t = self.transform(self.target_landmarks)
            for name, term in ldist_terms.items():
                losses[name] = term(t, s)
        disp_terms = self.loss_terms_of_type(DisplacementLoss)
        misc_excl |= set(disp_terms.keys())
        for name, term in disp_terms.items():
            match = RE_TERM_VAR.match(name)
            if match:
                var = match.group("var")
            elif "v" in variables:
                var = "v"
            elif "u" in variables:
                var = "u"
            else:
                var = "w"
            bufs = variables.get(var)
            if not bufs:
                raise RuntimeError(f"Unknown variable in loss term name '{name}'")
            value = paddle.to_tensor(data=0, dtype="float32", place=self.device)
            for buf in bufs:
                value += term(buf)
            losses[name] = value
        bspline_transforms = self.transforms_of_type(BSplineTransform)
        bspline_terms = self.loss_terms_of_type(BSplineLoss)
        misc_excl |= set(bspline_terms.keys())
        for name, term in bspline_terms.items():
            value = paddle.to_tensor(data=0, dtype="float32", place=self.device)
            for bspline_transform in bspline_transforms:
                value += term(bspline_transform.data())
            losses[name] = value
        params_terms = self.loss_terms_of_type(ParamsLoss)
        misc_excl |= set(params_terms.keys())
        for name, term in params_terms.items():
            value = paddle.to_tensor(data=0, dtype="float32", place=self.device)
            count = 0
            for params in self.transform.parameters():
                value += term(params)
                count += 1
            if count > 1:
                value /= count
            losses[name] = value
        misc_terms = {k: v for k, v in self.loss_terms.items() if k not in misc_excl}
        for name, term in misc_terms.items():
            losses[name] = term()
        result["losses"] = losses
        result["weights"] = self.weights
        result["loss"] = self._weighted_sum(losses)
        return result

    def _weighted_sum(self, losses: Mapping[str, paddle.Tensor]) -> paddle.Tensor:
        """Compute weighted sum of loss terms."""
        loss = paddle.to_tensor(data=0, dtype="float32", place=self.device)
        weights = self.weights
        for name, value in losses.items():
            w = weights.get(name, 1.0)
            if not isinstance(w, str):
                value = w * value
            loss += value.sum()
        return loss


def weight_channel_names(weights: Mapping[str, Union[float, str]]) -> Dict[str, str]:
    """Get names of channels that are used to weight loss term of another channel."""
    names = {}
    for term, weight in weights.items():
        if not isinstance(weight, str):
            continue
        match = RE_WEIGHT.match(weight)
        if match is None:
            continue
        names[term] = match.group("chn")
    return names
