from typing import List, Union

from paddle import nn
from paddle.nn import functional as F

from ..utils import validate_scaling_factor


class DomainPadding(nn.Layer):
    """Applies domain padding scaled automatically to the input's resolution

    Parameters
    ----------
    domain_padding : float or list
        typically, between zero and one, percentage of padding to use
        if a list, make sure if matches the dim of (d1, ..., dN)
    padding_mode : {'symmetric', 'one-sided'}, optional
        whether to pad on both sides, by default 'one-sided'
    output_scaling_factor : int ; default is 1

    Notes
    -----
    This class works for any input resolution, as long as it is in the form
    `(batch-size, channels, d1, ...., dN)`
    """

    def __init__(
        self,
        domain_padding,
        padding_mode="one-sided",
        output_scaling_factor: Union[int, List[int]] = 1,
    ):
        super().__init__()
        self.domain_padding = domain_padding
        self.padding_mode = padding_mode.lower()
        if output_scaling_factor is None:
            output_scaling_factor = 1
        self.output_scaling_factor: Union[int, List[int]] = output_scaling_factor

        # dict(f'{resolution}'=padding) such that padded = F.pad(x, indices)
        self._padding = dict()

        # dict(f'{resolution}'=indices_to_unpad) such that unpadded = x[indices]
        self._unpad_indices = dict()

    def forward(self, x):
        """forward pass: pad the input"""
        self.pad(x)

    def pad(self, x, verbose=False):
        """Take an input and pad it by the desired fraction

        The amount of padding will be automatically scaled with the resolution
        """
        resolution = x.shape[2:]

        # if domain_padding is list, then to pass on
        if isinstance(self.domain_padding, (float, int)):
            self.domain_padding = [float(self.domain_padding)] * len(resolution)

        assert len(self.domain_padding) == len(resolution), (
            "domain_padding length must match the number of spatial/time dimensions "
            "(excluding batch, ch)"
        )

        output_scaling_factor = self.output_scaling_factor
        if not isinstance(self.output_scaling_factor, list):
            # if unset by the user, scaling_factor will be 1 be default,
            # so `output_scaling_factor` should never be None.
            output_scaling_factor: List[float] = validate_scaling_factor(
                self.output_scaling_factor, len(resolution), n_layers=None
            )

        try:
            padding = self._padding[f"{resolution}"]
            return F.pad(x, padding, mode="constant")

        except KeyError:
            padding = [round(p * r) for (p, r) in zip(self.domain_padding, resolution)]

            if verbose:
                print(
                    f"Padding inputs of resolution={resolution} with "
                    f"padding={padding}, {self.padding_mode}"
                )

            output_pad = padding

            output_pad = [
                round(i * j) for (i, j) in zip(output_scaling_factor, output_pad)
            ]

            # padding is being applied in reverse order
            # (so we must reverse the padding list)
            padding = padding[::-1]

            # the F.pad(x, padding) funtion pads the tensor 'x' in reverse order
            # of the "padding" list i.e. the last axis of tensor 'x' will be
            # padded by the amount mention at the first position of the
            # 'padding' vector. The details about F.pad can be found here:
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html

            if self.padding_mode == "symmetric":
                # Pad both sides
                unpad_list = list()
                for p in output_pad:
                    if p == 0:
                        padding_end = None
                        padding_start = None
                    else:
                        padding_end = p
                        padding_start = -p
                    unpad_list.append(slice(padding_end, padding_start, None))
                unpad_indices = (Ellipsis,) + tuple(unpad_list)

                padding = [i for p in padding for i in (p, p)]

            elif self.padding_mode == "one-sided":
                # One-side padding
                unpad_list = list()
                for p in output_pad:
                    if p == 0:
                        padding_start = None
                    else:
                        padding_start = -p
                    unpad_list.append(slice(None, padding_start, None))
                unpad_indices = (Ellipsis,) + tuple(unpad_list)
                padding = [i for p in padding for i in (0, p)]
            else:
                raise ValueError(f"Got padding_mode={self.padding_mode}")

            self._padding[f"{resolution}"] = padding

            padded = F.pad(x, padding, mode="constant")

            output_shape = padded.shape[2:]

            output_shape = [
                round(i * j) for (i, j) in zip(output_scaling_factor, output_shape)
            ]

            self._unpad_indices[f"{[i for i in output_shape]}"] = unpad_indices

            return padded

    def unpad(self, x):
        """Remove the padding from padding inputs"""
        unpad_indices = self._unpad_indices[f"{list(x.shape[2:])}"]
        return x[unpad_indices]
