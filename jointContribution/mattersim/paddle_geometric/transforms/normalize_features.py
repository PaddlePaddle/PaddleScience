from typing import List, Union

from paddle_geometric.data import Data, HeteroData
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.transforms import BaseTransform


@functional_transform('normalize_features')
class NormalizeFeatures(BaseTransform):
    r"""Row-normalizes the attributes given in :obj:`attrs` to sum-up to one
    (functional name: :obj:`normalize_features`).

    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["x"]`)
    """
    def __init__(self, attrs: List[str] = ["x"]):
        self.attrs = attrs

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                if value.numel() > 0:
                    value = value - value.min()
                    value.divide_(value.sum(axis=-1, keepdim=True).clip_(min=1.))
                    store[key] = value
        return data
