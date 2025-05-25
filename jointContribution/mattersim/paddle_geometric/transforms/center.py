from typing import Union

from paddle_geometric.data import Data, HeteroData
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.transforms import BaseTransform


@functional_transform('center')
class Center(BaseTransform):
    r"""Centers node positions :obj:`data.pos` around the origin
    (functional name: :obj:`center`).
    """
    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.node_stores:
            if hasattr(store, 'pos'):
                store.pos = store.pos - store.pos.mean(axis=-2, keepdim=True)
        return data
