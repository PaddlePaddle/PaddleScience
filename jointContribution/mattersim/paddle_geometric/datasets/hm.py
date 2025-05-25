from typing import Callable, List, Optional

import pandas as pd
import paddle
from paddle_geometric.data import HeteroData, InMemoryDataset


class HM(InMemoryDataset):
    r"""The heterogeneous H&M dataset from the `Kaggle H&M Personalized Fashion
    Recommendations
    <https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations>`_
    challenge.
    The task is to develop product recommendations based on data from previous
    transactions, as well as from customer and product meta data.

    Args:
        root (str): Root directory where the dataset should be saved.
        use_all_tables_as_node_types (bool, optional): If set to :obj:`True`,
            will use the transaction table as a distinct node type.
            (default: :obj:`False`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`paddle_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`paddle_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """
    url = ('https://www.kaggle.com/competitions/'
           'h-and-m-personalized-fashion-recommendations/data')

    def __init__(
        self,
        root: str,
        use_all_tables_as_node_types: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.use_all_tables_as_node_types = use_all_tables_as_node_types
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'customers.csv.zip', 'articles.csv.zip',
            'transactions_train.csv.zip'
        ]

    @property
    def processed_file_names(self) -> str:
        if self.use_all_tables_as_node_types:
            return 'data.pdparams'
        else:
            return 'data_merged.pdparams'

    def download(self) -> None:
        raise RuntimeError(
            f"Dataset not found. Please download {self.raw_file_names} from "
            f"'{self.url}' and move it to '{self.raw_dir}'")

    def process(self) -> None:
        data = HeteroData()

        # Process customer data ###############################################
        df = pd.read_csv(self.raw_paths[0], index_col='customer_id')
        customer_map = {idx: i for i, idx in enumerate(df.index)}

        xs = []
        for name in [
                'Active', 'FN', 'club_member_status', 'fashion_news_frequency'
        ]:
            x = pd.get_dummies(df[name]).values
            xs.append(paddle.to_tensor(x, dtype='float32'))

        x = paddle.to_tensor(df['age'].values, dtype='float32').reshape([-1, 1])
        x = paddle.where(paddle.isnan(x), paddle.full_like(x, x.mean()), x)
        xs.append(x / x.max())

        data['customer'].x = paddle.concat(xs, axis=-1)

        # Process article data ################################################
        df = pd.read_csv(self.raw_paths[1], index_col='article_id')
        article_map = {idx: i for i, idx in enumerate(df.index)}

        xs = []
        for name in [
                'product_type_no', 'product_type_name', 'product_group_name',
                'graphical_appearance_no', 'graphical_appearance_name',
                'colour_group_code', 'colour_group_name',
                'perceived_colour_value_id', 'perceived_colour_value_name',
                'perceived_colour_master_id', 'perceived_colour_master_name',
                'index_code', 'index_name', 'index_group_no',
                'index_group_name', 'section_no', 'section_name',
                'garment_group_no', 'garment_group_name'
        ]:
            x = pd.get_dummies(df[name]).values
            xs.append(paddle.to_tensor(x, dtype='float32'))

        data['article'].x = paddle.concat(xs, axis=-1)

        # Process transaction data ############################################
        df = pd.read_csv(self.raw_paths[2], parse_dates=['t_dat'])

        x1 = pd.get_dummies(df['sales_channel_id']).values
        x1 = paddle.to_tensor(x1, dtype='float32')
        x2 = paddle.to_tensor(df['price'].values, dtype='float32').reshape([-1, 1])
        x = paddle.concat([x1, x2], axis=-1)

        time = paddle.to_tensor(df['t_dat'].values.astype('int64'))
        time = time // (60 * 60 * 24 * 10**9)  # Convert nanoseconds to days.

        src = paddle.to_tensor([customer_map[idx] for idx in df['customer_id']])
        dst = paddle.to_tensor([article_map[idx] for idx in df['article_id']])

        if self.use_all_tables_as_node_types:
            data['transaction'].x = x
            data['transaction'].time = time

            edge_index = paddle.stack([src, paddle.arange(len(df))], axis=0)
            data['customer', 'to', 'transaction'].edge_index = edge_index
            data['transaction', 'rev_to', 'customer'].edge_index = edge_index[::-1]

            edge_index = paddle.stack([dst, paddle.arange(len(df))], axis=0)
            data['article', 'to', 'transaction'].edge_index = edge_index
            data['transaction', 'rev_to', 'article'].edge_index = edge_index[::-1]
        else:
            edge_index = paddle.stack([src, dst], axis=0)
            data['customer', 'to', 'article'].edge_index = edge_index
            data['customer', 'to', 'article'].time = time
            data['customer', 'to', 'article'].edge_attr = x

            data['article', 'rev_to', 'customer'].edge_index = edge_index[::-1]
            data['article', 'rev_to', 'customer'].time = time
            data['article', 'rev_to', 'customer'].edge_attr = x

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
