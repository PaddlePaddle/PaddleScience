import os
from typing import Callable, Optional

import numpy as np
import paddle

from paddle_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class Taobao(InMemoryDataset):
    r"""The Taobao dataset, a user behavior dataset from Taobao provided by Alibaba,
    available via the `Tianchi Alicloud platform
    <https://tianchi.aliyun.com/dataset/649>`_.

    The Taobao dataset is a heterogeneous graph for recommendation tasks.
    Nodes represent users (user IDs), items (item IDs), and categories (category IDs).
    Edges between users and items represent different types of user behaviors towards items,
    and edges between items and categories assign each item to a set of categories.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in a
            :obj:`paddle_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in a
            :obj:`paddle_geometric.data.HeteroData` object and returns a transformed
            version before saving to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    """
    url = ('https://alicloud-dev.oss-cn-hangzhou.aliyuncs.com/'
           'UserBehavior.csv.zip')

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> str:
        return 'UserBehavior.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def process(self) -> None:
        import pandas as pd

        # Define columns and load data
        cols = ['userId', 'itemId', 'categoryId', 'behaviorType', 'timestamp']
        df = pd.read_csv(self.raw_paths[0], names=cols)

        # Filter data by time range
        start = 1511539200  # Start timestamp: 2017.11.25-00:00:00
        end = 1512316799    # End timestamp: 2017.12.03-23:59:59
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

        # Remove duplicate entries
        df = df.drop_duplicates()

        # Map behavior types to integers
        behavior_dict = {'pv': 0, 'cart': 1, 'buy': 2, 'fav': 3}
        df['behaviorType'] = df['behaviorType'].map(behavior_dict)

        num_entries = {}
        for name in ['userId', 'itemId', 'categoryId']:
            # Map IDs to consecutive integers
            value, df[name] = np.unique(df[[name]].values, return_inverse=True)
            num_entries[name] = value.shape[0]

        data = HeteroData()

        data['user'].num_nodes = num_entries['userId']
        data['item'].num_nodes = num_entries['itemId']
        data['category'].num_nodes = num_entries['categoryId']

        # Set up user-item edges with timestamp and behavior type as edge attributes
        row = paddle.to_tensor(df['userId'].values, dtype='int64')
        col = paddle.to_tensor(df['itemId'].values, dtype='int64')
        data['user', 'item'].edge_index = paddle.stack([row, col], axis=0)
        data['user', 'item'].time = paddle.to_tensor(df['timestamp'].values, dtype='int64')
        behavior = paddle.to_tensor(df['behaviorType'].values, dtype='int64')
        data['user', 'item'].behavior = behavior

        # Set up item-category edges
        df = df[['itemId', 'categoryId']].drop_duplicates()
        row = paddle.to_tensor(df['itemId'].values, dtype='int64')
        col = paddle.to_tensor(df['categoryId'].values, dtype='int64')
        data['item', 'category'].edge_index = paddle.stack([row, col], axis=0)

        # Apply any pre-transformations if specified
        data = data if self.pre_transform is None else self.pre_transform(data)

        # Save the processed data
        self.save([data], self.processed_paths[0])
