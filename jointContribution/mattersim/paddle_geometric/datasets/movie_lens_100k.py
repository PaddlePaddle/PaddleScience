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

MOVIE_HEADERS = [
    "movieId", "title", "releaseDate", "videoReleaseDate", "IMDb URL",
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
USER_HEADERS = ["userId", "age", "gender", "occupation", "zipCode"]
RATING_HEADERS = ["userId", "movieId", "rating", "timestamp"]


class MovieLens100K(InMemoryDataset):
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'

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
        return ['u.item', 'u.user', 'u1.base', 'u1.test']

    @property
    def processed_file_names(self) -> str:
        return 'data.pdparams'

    def download(self) -> None:
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.remove(path)
        folder = osp.join(self.root, 'ml-100k')
        fs.rm(self.raw_dir)
        os.rename(folder, self.raw_dir)

    def process(self) -> None:
        import pandas as pd

        data = HeteroData()

        # Process movie data
        df = pd.read_csv(
            self.raw_paths[0],
            sep='|',
            header=None,
            names=MOVIE_HEADERS,
            index_col='movieId',
            encoding='ISO-8859-1',
        )
        movie_mapping = {idx: i for i, idx in enumerate(df.index)}

        x = np.array(df[MOVIE_HEADERS[6:]].values, dtype='float32')
        data['movie'].x = paddle.to_tensor(x)

        # Process user data
        df = pd.read_csv(
            self.raw_paths[1],
            sep='|',
            header=None,
            names=USER_HEADERS,
            index_col='userId',
            encoding='ISO-8859-1',
        )
        user_mapping = {idx: i for i, idx in enumerate(df.index)}

        age = paddle.to_tensor(df['age'].values / df['age'].values.max(), dtype='float32').reshape([-1, 1])
        gender = paddle.to_tensor(df['gender'].str.get_dummies().values, dtype='float32')
        occupation = paddle.to_tensor(df['occupation'].str.get_dummies().values, dtype='float32')

        data['user'].x = paddle.concat([age, gender, occupation], axis=-1)

        # Process rating data for training
        df = pd.read_csv(
            self.raw_paths[2],
            sep='\t',
            header=None,
            names=RATING_HEADERS,
        )

        src = [user_mapping[idx] for idx in df['userId']]
        dst = [movie_mapping[idx] for idx in df['movieId']]
        edge_index = paddle.to_tensor([src, dst], dtype='int64')
        data['user', 'rates', 'movie'].edge_index = edge_index

        rating = paddle.to_tensor(df['rating'].values, dtype='int64')
        data['user', 'rates', 'movie'].rating = rating

        time = paddle.to_tensor(df['timestamp'].values, dtype='int64')
        data['user', 'rates', 'movie'].time = time

        data['movie', 'rated_by', 'user'].edge_index = paddle.flip(edge_index, [0])
        data['movie', 'rated_by', 'user'].rating = rating
        data['movie', 'rated_by', 'user'].time = time

        # Process rating data for testing
        df = pd.read_csv(
            self.raw_paths[3],
            sep='\t',
            header=None,
            names=RATING_HEADERS,
        )

        src = [user_mapping[idx] for idx in df['userId']]
        dst = [movie_mapping[idx] for idx in df['movieId']]
        edge_label_index = paddle.to_tensor([src, dst], dtype='int64')
        data['user', 'rates', 'movie'].edge_label_index = edge_label_index

        edge_label = paddle.to_tensor(df['rating'].values, dtype='float32')
        data['user', 'rates', 'movie'].edge_label = edge_label

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'MovieLens100K({len(self)})'
