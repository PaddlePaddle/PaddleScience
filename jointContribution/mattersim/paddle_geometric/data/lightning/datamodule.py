import copy
import inspect
import warnings
from typing import Any, Dict, Optional, Tuple, Type, Union

import paddle
from paddle.io import DataLoader
from paddle_geometric.data import Data, HeteroData
from paddle_geometric.loader import NeighborSampler

from paddle_geometric.data import Data, Dataset, HeteroData
from paddle_geometric.loader import DataLoader, LinkLoader, NodeLoader
from paddle_geometric.sampler import BaseSampler, NeighborSampler
from paddle_geometric.typing import InputEdges, InputNodes, OptTensor



class LightningDataModule:
    def __init__(self, has_val: bool, has_test: bool, **kwargs: Any) -> None:
        self.has_val = has_val
        self.has_test = has_test

        kwargs.setdefault('batch_size', 1)
        kwargs.setdefault('num_workers', 0)
        kwargs.setdefault('pin_memory', True)
        kwargs.setdefault('persistent_workers', kwargs.get('num_workers', 0) > 0)

        if 'shuffle' in kwargs:
            warnings.warn(
                f"The 'shuffle={kwargs['shuffle']}' option is ignored in '{self.__class__.__name__}'."
                " Remove it from the argument list to disable this warning"
            )
            del kwargs['shuffle']

        self.kwargs = kwargs

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.kwargs})'


class LightningData(LightningDataModule):
    def __init__(
        self,
        data: Union[Data, HeteroData],
        has_val: bool,
        has_test: bool,
        loader: str = 'neighbor',
        graph_sampler: Optional[NeighborSampler] = None,
        eval_loader_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault('batch_size', 1)
        kwargs.setdefault('num_workers', 0)

        if graph_sampler is not None:
            loader = 'custom'

        if loader not in ['full', 'neighbor', 'custom']:
            raise ValueError(f"Undefined 'loader' option (got '{loader}')")

        if loader == 'full' and kwargs['batch_size'] != 1:
            warnings.warn(f"Re-setting 'batch_size' to 1 for loader='full'")
            kwargs['batch_size'] = 1

        if loader == 'full' and kwargs['num_workers'] != 0:
            warnings.warn(f"Re-setting 'num_workers' to 0 for loader='full'")
            kwargs['num_workers'] = 0

        if loader == 'full' and kwargs.get('sampler') is not None:
            warnings.warn("'sampler' option is not supported for loader='full'")
            kwargs.pop('sampler', None)

        if loader == 'full' and kwargs.get('batch_sampler') is not None:
            warnings.warn("'batch_sampler' option is not supported for loader='full'")
            kwargs.pop('batch_sampler', None)

        super().__init__(has_val, has_test, **kwargs)

        if loader == 'full':
            if kwargs.get('pin_memory', False):
                warnings.warn(f"Re-setting 'pin_memory' to 'False' for loader='full'")
            self.kwargs['pin_memory'] = False

        self.data = data
        self.loader = loader

        if loader in ['neighbor']:
            sampler_kwargs = {k: v for k, v in kwargs.items() if k in NeighborSampler.__init__.__annotations__}
            sampler_kwargs.setdefault('share_memory', kwargs['num_workers'] > 0)
            self.graph_sampler = NeighborSampler(data, **sampler_kwargs)
            self.loader_kwargs = {k: v for k, v in kwargs.items() if k not in sampler_kwargs}

        elif graph_sampler is not None:
            self.graph_sampler = graph_sampler
            self.loader_kwargs = kwargs

        else:
            assert loader == 'full'
            self.loader_kwargs = kwargs

        self.eval_loader_kwargs = copy.copy(self.loader_kwargs)
        if eval_loader_kwargs is not None:
            if hasattr(self, 'graph_sampler'):
                self.eval_graph_sampler = copy.copy(self.graph_sampler)

                eval_sampler_kwargs = {
                    k: v for k, v in eval_loader_kwargs.items() if k in NeighborSampler.__init__.__annotations__
                }
                for key, value in eval_sampler_kwargs.items():
                    setattr(self.eval_graph_sampler, key, value)

            self.eval_loader_kwargs.update(eval_loader_kwargs)

        elif hasattr(self, 'graph_sampler'):
            self.eval_graph_sampler = self.graph_sampler

        self.eval_loader_kwargs.pop('sampler', None)
        self.eval_loader_kwargs.pop('batch_sampler', None)

        if 'batch_sampler' in self.loader_kwargs:
            self.loader_kwargs.pop('batch_size', None)

    @property
    def train_shuffle(self) -> bool:
        shuffle = self.loader_kwargs.get('sampler', None) is None
        shuffle &= self.loader_kwargs.get('batch_sampler', None) is None
        return shuffle

    def prepare_data(self) -> None:
        if self.loader == 'full':
            raise ValueError(
                f"'{self.__class__.__name__}' with loader='full' requires training on a single device"
            )

    def full_dataloader(self, **kwargs: Any) -> DataLoader:
        warnings.filterwarnings('ignore', '.*does not have many workers.*')
        warnings.filterwarnings('ignore', '.*data loading bottlenecks.*')

        return DataLoader(
            [self.data],  # type: ignore
            batch_size=1,
            collate_fn=lambda xs: xs[0],
            **kwargs,
        )

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(data={self.data}, loader={self.loader}, kwargs={self.kwargs})'

class LightningDataset(LightningDataModule):
    r"""
    Converts a set of `paddle_geometric.data.Dataset` objects into a compatible module
    for multi-GPU graph-level training.

    This class simplifies the integration of datasets with PaddlePaddle's DataLoader
    for training, validation, testing, and prediction.
    """

    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        pred_dataset: Optional[Dataset] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the LightningDataset.

        Args:
            train_dataset (Dataset): The training dataset.
            val_dataset (Dataset, optional): The validation dataset. Defaults to None.
            test_dataset (Dataset, optional): The test dataset. Defaults to None.
            pred_dataset (Dataset, optional): The prediction dataset. Defaults to None.
            **kwargs (optional): Additional arguments for the DataLoader.
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.pred_dataset = pred_dataset
        self.kwargs = kwargs

    def dataloader(self, dataset: Dataset, **kwargs: Any) -> DataLoader:
        """
        Create a DataLoader for the given dataset.

        Args:
            dataset (Dataset): The dataset to load.
            **kwargs (optional): Additional arguments for the DataLoader.

        Returns:
            DataLoader: A DataLoader instance for the dataset.
        """
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for training.

        Returns:
            DataLoader: The DataLoader for training data.
        """
        shuffle = self.kwargs.get('sampler', None) is None and \
                  self.kwargs.get('batch_sampler', None) is None
        return self.dataloader(
            self.train_dataset,
            shuffle=shuffle,
            **self.kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for validation.

        Returns:
            DataLoader: The DataLoader for validation data.
        """
        assert self.val_dataset is not None, "Validation dataset cannot be None"

        kwargs = copy.copy(self.kwargs)
        kwargs.pop('sampler', None)
        kwargs.pop('batch_sampler', None)

        return self.dataloader(self.val_dataset, shuffle=False, **kwargs)

    def test_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for testing.

        Returns:
            DataLoader: The DataLoader for test data.
        """
        assert self.test_dataset is not None, "Test dataset cannot be None"

        kwargs = copy.copy(self.kwargs)
        kwargs.pop('sampler', None)
        kwargs.pop('batch_sampler', None)

        return self.dataloader(self.test_dataset, shuffle=False, **kwargs)

    def predict_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for predictions.

        Returns:
            DataLoader: The DataLoader for prediction data.
        """
        assert self.pred_dataset is not None, "Prediction dataset cannot be None"

        kwargs = copy.copy(self.kwargs)
        kwargs.pop('sampler', None)
        kwargs.pop('batch_sampler', None)

        return self.dataloader(self.pred_dataset, shuffle=False, **kwargs)

    def __repr__(self) -> str:
        """
        Return a string representation of the object.

        Returns:
            str: A string representation of the object.
        """
        kwargs = {
            "train_dataset": self.train_dataset,
            "val_dataset": self.val_dataset,
            "test_dataset": self.test_dataset,
            "pred_dataset": self.pred_dataset,
            **self.kwargs,
        }
        return f'{self.__class__.__name__}({kwargs})'


class LightningNodeData(LightningData):
    """
    Converts a `paddle_geometric.data.Data` or `paddle_geometric.data.HeteroData` object
    into a node-level DataLoader for multi-GPU training using Paddle.

    This class simplifies the process of preparing data for training, validation,
    testing, and prediction at the node level.
    """

    def __init__(
        self,
        data: Union[Data, HeteroData],
        input_train_nodes: InputNodes = None,
        input_train_time: OptTensor = None,
        input_val_nodes: InputNodes = None,
        input_val_time: OptTensor = None,
        input_test_nodes: InputNodes = None,
        input_test_time: OptTensor = None,
        input_pred_nodes: InputNodes = None,
        input_pred_time: OptTensor = None,
        loader: str = 'neighbor',
        node_sampler: Optional[BaseSampler] = None,
        eval_loader_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # Automatically infer node splits if not provided.
        if input_train_nodes is None:
            input_train_nodes = self.infer_input_nodes(data, split='train')

        if input_val_nodes is None:
            input_val_nodes = self.infer_input_nodes(data, split='val')

        if input_test_nodes is None:
            input_test_nodes = self.infer_input_nodes(data, split='test')

        if input_pred_nodes is None:
            input_pred_nodes = self.infer_input_nodes(data, split='pred')

        super().__init__(
            data=data,
            has_val=input_val_nodes is not None,
            has_test=input_test_nodes is not None,
            loader=loader,
            graph_sampler=node_sampler,
            eval_loader_kwargs=eval_loader_kwargs,
            **kwargs,
        )

        # self.data = data
        # self.loader = loader
        # self.node_sampler = node_sampler
        # self.eval_loader_kwargs = eval_loader_kwargs or {}
        # self.kwargs = kwargs

        self.input_train_nodes = input_train_nodes
        self.input_train_time = input_train_time
        self.input_train_id: OptTensor = None

        self.input_val_nodes = input_val_nodes
        self.input_val_time = input_val_time
        self.input_val_id: OptTensor = None

        self.input_test_nodes = input_test_nodes
        self.input_test_time = input_test_time
        self.input_test_id: OptTensor = None

        self.input_pred_nodes = input_pred_nodes
        self.input_pred_time = input_pred_time
        self.input_pred_id: OptTensor = None

    def infer_input_nodes(self, data, split: str) -> InputNodes:
        """
        Infers the input nodes for a given split (train, val, test, pred) based on
        attributes in the data object.
        """
        for attr in [f'{split}_mask', f'{split}_idx', f'{split}_index']:
            if hasattr(data, attr):
                return getattr(data, attr)
        return None

    def dataloader(
        self,
        input_nodes: InputNodes,
        input_time: OptTensor = None,
        input_id: OptTensor = None,
        node_sampler: Optional[BaseSampler] = None,
        **kwargs: Any,
    ) -> DataLoader:
        """
        Creates a DataLoader for the given input nodes.

        Args:
            input_nodes: The nodes to sample.
            input_time: Optional time information for temporal graphs.
            input_id: Optional IDs for additional filtering.
            node_sampler: The sampler object to use for batching.
            **kwargs: Additional DataLoader arguments.
        """
        if self.loader == 'full':
            return DataLoader([self.data], batch_size=1, shuffle=False, **kwargs)

        assert node_sampler is not None, "Node sampler is required for neighbor sampling."

        return NodeLoader(
            data=self.data,
            node_sampler=node_sampler,
            input_nodes=input_nodes,
            input_time=input_time,
            input_id=input_id,
            **kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        """
        Creates a DataLoader for training.

        Returns:
            DataLoader: A DataLoader for the training nodes.
        """
        return self.dataloader(
            self.input_train_nodes,
            self.input_train_time,
            self.input_train_id,
            node_sampler=self.node_sampler,
            shuffle=True,
            **self.kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Creates a DataLoader for validation.

        Returns:
            DataLoader: A DataLoader for the validation nodes.
        """
        return self.dataloader(
            self.input_val_nodes,
            self.input_val_time,
            self.input_val_id,
            node_sampler=self.node_sampler,
            shuffle=False,
            **self.eval_loader_kwargs,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Creates a DataLoader for testing.

        Returns:
            DataLoader: A DataLoader for the test nodes.
        """
        return self.dataloader(
            self.input_test_nodes,
            self.input_test_time,
            self.input_test_id,
            node_sampler=self.node_sampler,
            shuffle=False,
            **self.eval_loader_kwargs,
        )

    def predict_dataloader(self) -> DataLoader:
        """
        Creates a DataLoader for prediction.

        Returns:
            DataLoader: A DataLoader for the prediction nodes.
        """
        return self.dataloader(
            self.input_pred_nodes,
            self.input_pred_time,
            self.input_pred_id,
            node_sampler=self.node_sampler,
            shuffle=False,
            **self.eval_loader_kwargs,
        )
class LightningLinkData(LightningData):
    """
    Converts a `paddle_geometric.data.Data` or `paddle_geometric.data.HeteroData`
    object into a link-level DataLoader for multi-GPU training using Paddle.

    This class supports both full-batch and neighbor-based mini-batch loading.

    Args:
        data (Data or HeteroData): The graph data object.
        input_train_edges (Tensor, optional): The edges used for training.
        input_train_labels (Tensor, optional): Labels for the training edges.
        input_train_time (Tensor, optional): Timestamps for the training edges.
        input_val_edges (Tensor, optional): The edges used for validation.
        input_val_labels (Tensor, optional): Labels for the validation edges.
        input_val_time (Tensor, optional): Timestamps for the validation edges.
        input_test_edges (Tensor, optional): The edges used for testing.
        input_test_labels (Tensor, optional): Labels for the test edges.
        input_test_time (Tensor, optional): Timestamps for the test edges.
        input_pred_edges (Tensor, optional): The edges used for prediction.
        input_pred_labels (Tensor, optional): Labels for the prediction edges.
        input_pred_time (Tensor, optional): Timestamps for the prediction edges.
        loader (str): Loading strategy ('full' or 'neighbor'). Default is 'neighbor'.
        link_sampler (BaseSampler, optional): Custom sampler for mini-batches.
        eval_loader_kwargs (dict, optional): Additional arguments for evaluation loaders.
        **kwargs (optional): Additional arguments for `LinkNeighborLoader`.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData],
        input_train_edges: InputEdges = None,
        input_train_labels: OptTensor = None,
        input_train_time: OptTensor = None,
        input_val_edges: InputEdges = None,
        input_val_labels: OptTensor = None,
        input_val_time: OptTensor = None,
        input_test_edges: InputEdges = None,
        input_test_labels: OptTensor = None,
        input_test_time: OptTensor = None,
        input_pred_edges: InputEdges = None,
        input_pred_labels: OptTensor = None,
        input_pred_time: OptTensor = None,
        loader: str = 'neighbor',
        link_sampler: Optional[BaseSampler] = None,
        eval_loader_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:

        super().__init__(
            data=data,
            has_val=input_val_edges is not None,
            has_test=input_test_edges is not None,
            loader=loader,
            graph_sampler=link_sampler,
            eval_loader_kwargs=eval_loader_kwargs,
            **kwargs,
        )
        # self.data = data
        # self.loader = loader
        # self.link_sampler = link_sampler
        # self.eval_loader_kwargs = eval_loader_kwargs or {}
        # self.kwargs = kwargs

        self.input_train_edges = input_train_edges
        self.input_train_labels = input_train_labels
        self.input_train_time = input_train_time
        self.input_train_id: OptTensor = None

        self.input_val_edges = input_val_edges
        self.input_val_labels = input_val_labels
        self.input_val_time = input_val_time
        self.input_val_id: OptTensor = None

        self.input_test_edges = input_test_edges
        self.input_test_labels = input_test_labels
        self.input_test_time = input_test_time
        self.input_test_id: OptTensor = None

        self.input_pred_edges = input_pred_edges
        self.input_pred_labels = input_pred_labels
        self.input_pred_time = input_pred_time
        self.input_pred_id: OptTensor = None

    def dataloader(
            self,
            input_edges=None,
            input_labels=None,
            input_time=None,
            link_sampler=None,
            **kwargs: Any,
    ) -> DataLoader:
        if self.loader == 'full':
            return DataLoader([self.data], batch_size=1, shuffle=False, **kwargs)

        assert link_sampler is not None, "Link sampler is required for neighbor sampling."

        return LinkLoader(
            data=self.data,
            link_sampler=link_sampler,
            edge_label_index=input_edges,
            edge_label=input_labels,
            edge_label_time=input_time,
            **kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(
            self.input_train_edges,
            self.input_train_labels,
            self.input_train_time,
            link_sampler=self.link_sampler,
            shuffle=True,
            **self.kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(
            self.input_val_edges,
            self.input_val_labels,
            self.input_val_time,
            link_sampler=self.link_sampler,
            shuffle=False,
            **self.eval_loader_kwargs,
        )

    def test_dataloader(self) -> DataLoader:
        return self.dataloader(
            self.input_test_edges,
            self.input_test_labels,
            self.input_test_time,
            link_sampler=self.link_sampler,
            shuffle=False,
            **self.eval_loader_kwargs,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.dataloader(
            self.input_pred_edges,
            self.input_pred_labels,
            self.input_pred_time,
            link_sampler=self.link_sampler,
            shuffle=False,
            **self.eval_loader_kwargs,
        )

# Supporting Functions
def infer_input_nodes(data: Union[Data, HeteroData], split: str):
    attr_name: Optional[str] = None
    if f'{split}_mask' in data:
        attr_name = f'{split}_mask'
    elif f'{split}_idx' in data:
        attr_name = f'{split}_idx'
    elif f'{split}_index' in data:
        attr_name = f'{split}_index'

    if attr_name is None:
        return None

    if isinstance(data, Data):
        return data[attr_name]
    if isinstance(data, HeteroData):
        input_nodes_dict = {
            node_type: store[attr_name]
            for node_type, store in data.node_items() if attr_name in store
        }
        if len(input_nodes_dict) != 1:
            raise ValueError(f"Could not automatically determine the input "
                             f"nodes of {data} since there exist multiple "
                             f"types with attribute '{attr_name}'")
        return list(input_nodes_dict.items())[0]
    return None

def kwargs_repr(**kwargs: Any) -> str:
    return ', '.join([f'{k}={v}' for k, v in kwargs.items() if v is not None])

def split_kwargs(
        kwargs: Dict[str, Any],
        sampler_cls: Type,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Splits keyword arguments into sampler and loader arguments.
    """
    sampler_args = sampler_cls.__init__.__code__.co_varnames

    sampler_kwargs: Dict[str, Any] = {}
    loader_kwargs: Dict[str, Any] = {}

    for key, value in kwargs.items():
        if key in sampler_args:
            sampler_kwargs[key] = value
        else:
            loader_kwargs[key] = value

    return sampler_kwargs, loader_kwargs