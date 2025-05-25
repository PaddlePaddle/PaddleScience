from typing import Any

import paddle
from paddle import Tensor

from paddle_geometric.data import Data
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.transforms import BaseTransform
from paddle_geometric.utils import scatter


@functional_transform('to_slic')
class ToSLIC(BaseTransform):
    r"""Converts an image to a superpixel representation using the
    :meth:`skimage.segmentation.slic` algorithm, resulting in a
    :obj:`paddle_geometric.data.Data` object holding the centroids of
    superpixels in :obj:`data.pos` and their mean color in :obj:`data.x`
    (functional name: :obj:`to_slic`).

    Args:
        add_seg (bool, optional): If set to `True`, will add the segmentation
            result to the data object. (default: :obj:`False`)
        add_img (bool, optional): If set to `True`, will add the input image
            to the data object. (default: :obj:`False`)
        **kwargs (optional): Arguments to adjust the output of the SLIC
            algorithm.
    """
    def __init__(
        self,
        add_seg: bool = False,
        add_img: bool = False,
        **kwargs: Any,
    ) -> None:
        self.add_seg = add_seg
        self.add_img = add_img
        self.kwargs = kwargs

    def forward(self, img: Tensor) -> Data:
        from skimage.segmentation import slic

        img = img.transpose([1, 2, 0])  # Permute dimensions to HWC
        h, w, c = img.shape

        seg = slic(img.astype('float64').numpy(), start_label=0, **self.kwargs)
        seg = paddle.to_tensor(seg)

        x = scatter(img.reshape([h * w, c]), seg.reshape([h * w]), dim=0, reduce='mean')

        pos_y = paddle.arange(h, dtype='float32')
        pos_y = pos_y.unsqueeze(1).expand([h, w]).reshape([h * w])
        pos_x = paddle.arange(w, dtype='float32')
        pos_x = pos_x.unsqueeze(0).expand([h, w]).reshape([h * w])

        pos = paddle.stack([pos_x, pos_y], axis=-1)
        pos = scatter(pos, seg.reshape([h * w]), dim=0, reduce='mean')

        data = Data(x=x, pos=pos)

        if self.add_seg:
            data.seg = seg.reshape([1, h, w])

        if self.add_img:
            data.img = img.transpose([2, 0, 1]).reshape([1, c, h, w])

        return data
