import os
import os.path as osp
from typing import Callable, List, Optional

import paddle
import numpy as np
from paddle_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from paddle_geometric.io import fs

MOVIE_HEADERS = ["movieId", "title", "genres"]
USER_HEADERS = ["userId", "gender", "age", "occupation", "zipCode"]
RATING_HEADERS = ['userId', 'movieId', 'rating', 'timestamp']


class MovieLens1M(InMemoryDataset):
    r"""The MovieLens 1M heterogeneous rating dataset, assembled by GroupLens
    Research from the `MovieLens web site <https://movielens.org>`__.
    """

    url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'

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
    def raw_file_names(self) -> List[str]:
        return ['movies.dat', 'users.dat', 'ratings.dat']

    @property
    def processed_file_names(self) -> str:
        return 'data.pdparams'

    def download(self) -> None:
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.remove(path)
        folder = osp.join(self.root, 'ml-1m')
        fs.rm(self.raw_dir)
        os.rename(folder, self.raw_dir)

    def process(self) -> None:
        import pandas as pd

        data = HeteroData()

        # Process movie data
        df = pd.read_csv(
            self.raw_paths[0],
            sep='::',
            header=None,
            index_col='movieId',
            names=MOVIE_HEADERS,
            encoding='ISO-8859-1',
            engine='python',
        )
        movie_mapping = {idx: i for i, idx in enumerate(df.index)}

        genres = paddle.to_tensor(np.array(df['genres'].str.get_dummies('|').values, dtype='float32'))
        data['movie'].x = genres

        # Process user data
        df = pd.read_csv(
            self.raw_paths[1],
            sep='::',
            header=None,
            index_col='userId',
            names=USER_HEADERS,
            dtype='str',
            encoding='ISO-8859-1',
            engine='python',
        )
        user_mapping = {idx: i for i, idx in enumerate(df.index)}

        age = paddle.to_tensor(np.array(df['age'].str.get_dummies().values, dtype='float32'))
        gender = paddle.to_tensor(np.array(df['gender'].str.get_dummies().values, dtype='float32'))
        occupation = paddle.to_tensor(np.array(df['occupation'].str.get_dummies().values, dtype='float32'))
        data['user'].x = paddle.concat([age, gender, occupation], axis=-1)

        # Process rating data
        df = pd.read_csv(
            self.raw_paths[2],
            sep='::',
            header=None,
            names=RATING_HEADERS,
            encoding='ISO-8859-1',
            engine='python',
        )

        src = [user_mapping[idx] for idx in df['userId']]
        dst = [movie_mapping[idx] for idx in df['movieId']]
        edge_index = paddle.to_tensor([src, dst], dtype='int64')
        data['user', 'rates', 'movie'].edge_index = edge_index

        rating = paddle.to_tensor(np.array(df['rating'].values, dtype='int64'))
        data['user', 'rates', 'movie'].rating = rating

        time = paddle.to_tensor(np.array(df['timestamp'].values, dtype='int64'))
        data['user', 'rates', 'movie'].time = time

        # Reverse edge for rated_by relation
        data['movie', 'rated_by', 'user'].edge_index = edge_index.flip([0])
        data['movie', 'rated_by', 'user'].rating = rating
        data['movie', 'rated_by', 'user'].time = time

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'MovieLens1M({len(self)})'
