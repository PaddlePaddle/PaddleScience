import json
import os
import os.path as osp
from typing import Callable, Dict, List, Literal, Optional

import paddle
import tqdm
from paddle import Tensor
from paddle_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_tar,
)



class OPFDataset(InMemoryDataset):
    r"""The heterogeneous OPF data for PaddlePaddle.

    Args:
        root (str): Root directory where the dataset should be saved.
        split (str, optional): Dataset split ('train', 'val', 'test').
        case_name (str, optional): The name of the original pglib-opf case.
        num_groups (int, optional): Number of dataset groups.
        topological_perturbations (bool, optional): Use perturbed data.
        transform (callable, optional): Transformation function.
        pre_transform (callable, optional): Pre-processing transformation.
        pre_filter (callable, optional): Pre-filter function.
        force_reload (bool, optional): Whether to force re-process.
    """
    url = 'https://storage.googleapis.com/gridopt-dataset'

    def __init__(
        self,
        root: str,
        split: Literal['train', 'val', 'test'] = 'train',
        case_name: Literal[
            'pglib_opf_case14_ieee',
            'pglib_opf_case30_ieee',
            'pglib_opf_case57_ieee',
            'pglib_opf_case118_ieee',
            'pglib_opf_case500_goc',
            'pglib_opf_case2000_goc',
            'pglib_opf_case6470_rte',
            'pglib_opf_case4661_sdet',
            'pglib_opf_case10000_goc',
            'pglib_opf_case13659_pegase',
        ] = 'pglib_opf_case14_ieee',
        num_groups: int = 20,
        topological_perturbations: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:

        self.split = split
        self.case_name = case_name
        self.num_groups = num_groups
        self.topological_perturbations = topological_perturbations

        self._release = 'dataset_release_1'
        if topological_perturbations:
            self._release += '_nminusone'

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        idx = self.processed_file_names.index(f'{split}.pkl')
        self.load(self.processed_paths[idx])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self._release, self.case_name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self._release, self.case_name,
                        f'processed_{self.num_groups}')

    @property
    def raw_file_names(self) -> List[str]:
        return [f'{self.case_name}_{i}.tar.gz' for i in range(self.num_groups)]

    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pkl', 'val.pkl', 'test.pkl']

    def download(self) -> None:
        for name in self.raw_file_names:
            url = f'{self.url}/{self._release}/{name}'
            path = download_url(url, self.raw_dir)
            extract_tar(path, self.raw_dir)

    def process(self) -> None:
        train_data_list = []
        val_data_list = []
        test_data_list = []

        for group in tqdm.tqdm(range(self.num_groups)):
            tmp_dir = osp.join(
                self.raw_dir,
                'gridopt-dataset-tmp',
                self._release,
                self.case_name,
                f'group_{group}',
            )

            for name in os.listdir(tmp_dir):
                with open(osp.join(tmp_dir, name)) as f:
                    obj = json.load(f)

                grid = obj['grid']
                solution = obj['solution']
                metadata = obj['metadata']

                # Create graph data:
                data = HeteroData()
                data['global'] = paddle.to_tensor(grid['context'], dtype='float32')
                data['global_objective'] = paddle.to_tensor(metadata['objective'], dtype='float32')

                # Nodes:
                data['bus'] = paddle.to_tensor(grid['nodes']['bus'], dtype='float32')
                data['bus_label'] = paddle.to_tensor(solution['nodes']['bus'], dtype='float32')

                data['generator'] = paddle.to_tensor(grid['nodes']['generator'], dtype='float32')
                data['generator_label'] = paddle.to_tensor(solution['nodes']['generator'], dtype='float32')

                data['load'] = paddle.to_tensor(grid['nodes']['load'], dtype='float32')
                data['shunt'] = paddle.to_tensor(grid['nodes']['shunt'], dtype='float32')

                # Edges:
                data['ac_line'] = self.extract_edge_features(grid, solution, 'ac_line')
                data['transformer'] = self.extract_edge_features(grid, solution, 'transformer')

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                i = int(name.split('.')[0].split('_')[1])
                train_limit = int(15_000 * self.num_groups * 0.9)
                val_limit = train_limit + int(15_000 * self.num_groups * 0.05)
                if i < train_limit:
                    train_data_list.append(data)
                elif i < val_limit:
                    val_data_list.append(data)
                else:
                    test_data_list.append(data)

        self.save(train_data_list, self.processed_paths[0])
        self.save(val_data_list, self.processed_paths[1])
        self.save(test_data_list, self.processed_paths[2])

    def extract_edge_features(self, grid: Dict, solution: Dict, edge_name: str) -> Dict:
        edge_data = {
            'index': paddle.to_tensor([
                grid['edges'][edge_name]['senders'],
                grid['edges'][edge_name]['receivers'],
            ], dtype='int64'),
            'features': paddle.to_tensor(grid['edges'][edge_name]['features'], dtype='float32'),
            'labels': paddle.to_tensor(solution['edges'][edge_name]['features'], dtype='float32'),
        }
        return edge_data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'split={self.split}, '
                f'case_name={self.case_name}, '
                f'topological_perturbations={self.topological_perturbations})')


def extract_edge_index(obj: Dict, edge_name: str) -> Tensor:
    return paddle.to_tensor([
        obj['grid']['edges'][edge_name]['senders'],
        obj['grid']['edges'][edge_name]['receivers'],
    ], dtype='int64')


def extract_edge_index_rev(obj: Dict, edge_name: str) -> Tensor:
    return paddle.to_tensor([
        obj['grid']['edges'][edge_name]['receivers'],
        obj['grid']['edges'][edge_name]['senders'],
    ], dtype='int64')
