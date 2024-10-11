from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from copy import copy as shallowcopy
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import TypeVar
from typing import Union
from typing import overload

import paddle
import pandas as pd
from paddle.io import Subset
from paddle.nn import Sequential

from ..core.config import DataclassConfig
from ..core.types import PathStr
from ..core.types import Sample
from ..core.types import is_namedtuple
from ..core.types import is_path_str
from .transforms import Transform
from .transforms.image import ImageTransformConfig
from .transforms.image import image_transforms
from .transforms.image import prepend_read_image_transform

__all__ = (
    "Dataset",
    "MetaDataset",
    "ImageDataset",
    "ImageDatasetConfig",
    "GroupDataset",
    "JoinDataset",
    "read_table",
)
TDataset = TypeVar("TDataset", bound="Dataset")


class Dataset(paddle.io.Dataset, metaclass=ABCMeta):
    """Base class of datasets with optionally on-the-fly pre-processed samples.

    This map-style dataset base class is convenient for attaching data transformations to
    a given dataset. Otherwise, datasets may also derive directly from the respective
    ``paddle.utils.data`` dataset classes or simply implement the expected interfaces.


    """

    def __init__(
        self, transforms: Optional[Union[Transform, Sequence[Transform]]] = None
    ):
        """Initialize dataset.

        If a dataset produces samples (i.e., a dictionary, named tuple, or custom dataclass)
        which contain fields with ``None`` values, ``collate_fn=collate_samples`` must be
        passed to ``paddle.utils.data.DataLoader``. This custom collate function will ignore
        ``None`` values and pass these on to the respective batch entry. Auxiliary function
        ``prepare_batch()`` can be used to transfer the batch data retrieved by the data
        loader to the execution device.

        Args:
            transforms: Data preprocessing and augmentation transforms.
                If more than one transformation is given, these will be composed
                in the given order, where the first transformation in the sequence
                is applied first. When the data samples are passed directly to
                ``paddle.utils.data.DataLoader``, the transformed sample data must
                be of type ``np.ndarray``, ``paddle.Tensor``, or ``None``.

        """
        super().__init__()
        if transforms is None:
            transform = Sequential()
        elif isinstance(transforms, Sequential):
            transform = transforms
        else:
            if not isinstance(transforms, (list, tuple)):
                transforms = [transforms]
            transform = Sequential(*transforms)
        self._transform: Sequential = transform

    @abstractmethod
    def __len__(self) -> int:
        """Number of samples in dataset."""
        raise NotImplementedError

    def __getitem__(self, index: int) -> Sample:
        """Processed data of i-th dataset sample.

        Args:
            index: Index of dataset sample.

        Returns:
            Sample data.

        """
        sample = self.sample(index)
        sample = self._transform(sample)
        return sample

    @abstractmethod
    def sample(self, index: int) -> Sample:
        """Data of i-th dataset sample."""
        raise NotImplementedError

    def samples(self) -> Iterable[Sample]:
        """Get iterable over untransformed dataset samples."""

        class SampleIterator(object):
            def __init__(self, dataset: Dataset):
                self.dataset = dataset
                self.index = -1

            def __iter__(self) -> Iterator[Sample]:
                self.index = 0
                return self

            def __next__(self) -> Sample:
                if self.index >= len(self.dataset):
                    raise StopIteration
                sample = self.dataset.sample(self.index)
                self.index += 1
                return sample

        return SampleIterator(self)

    @overload
    def transform(self) -> Sequential:
        ...

    @overload
    def transform(
        self: TDataset,
        arg0: Union[Transform, Sequence[Transform], None],
        *args: Union[Transform, Sequence[Transform], None],
    ) -> TDataset:
        ...

    def transform(
        self: TDataset, *args: Union[Transform, Sequence[Transform], None]
    ) -> Union[Sequential, TDataset]:
        """Get composite data preprocessing and augmentation transform, or new dataset with specified transform."""
        if not args:
            return self._transform
        return shallowcopy(self).transform_(*args)

    def transform_(
        self: TDataset,
        arg0: Union[Transform, Sequence[Transform], None],
        *args: Union[Transform, Sequence[Transform], None],
    ) -> TDataset:
        """Set data preprocessing and augmentation transform of this dataset."""
        transforms = []
        for arg in [arg0, *args]:
            if arg is None:
                continue
            if isinstance(arg, (list, tuple)):
                transforms.extend(arg)
            else:
                transforms.append(arg)
        if not transforms:
            self._transform = None
        elif len(transforms) == 1 and isinstance(transforms[0], Sequential):
            self._transform = transforms[0]
        else:
            self._transform = Sequential(*transforms)
        return self

    @overload
    def transforms(self) -> List[Transform]:
        ...

    @overload
    def transforms(
        self: TDataset,
        arg0: Union[Transform, Sequence[Transform], None],
        *args: Union[Transform, Sequence[Transform], None],
    ) -> TDataset:
        ...

    def transforms(
        self: TDataset, *args: Union[Transform, Sequence[Transform], None]
    ) -> Union[List[Transform], TDataset]:
        """Get or set dataset transforms."""
        if not args:
            return [transform for transform in self._transform]
        return shallowcopy(self).transform_(*args)

    def transforms_(
        self,
        arg0: Union[Transform, Sequence[Transform], None],
        *args: Union[Transform, Sequence[Transform], None],
    ) -> Dataset:
        """Set data transforms of this dataset."""
        return self.transform_(arg0, *args)


class MetaDataset(Dataset):
    """Dataset of file path template strings and sample meta-data given by Pandas DataFrame.

    This dataset can be used in conjunction with data reader transforms to load the data from
    configured input file paths. For example, use the :class:`deepali.data.transforms.ReadImage`
    transform followed by image data preprocessing and augmentation functions for image data.
    The specified file path strings are Python format strings, where keywords are replaced by the
    respective column entries for the sample in the dataset index table (`pandas.DataFrame`).

    """

    def __init__(
        self,
        table: Union[Path, str, pd.DataFrame],
        paths: Optional[Mapping[str, Union[PathStr, Callable[..., PathStr]]]] = None,
        prefix: Optional[PathStr] = None,
        transforms: Optional[Union[Transform, Sequence[Transform]]] = None,
        **kwargs,
    ):
        """Initialize dataset.

        Args:
            table: Table with sample IDs, optionally sample specific input file path template
                strings (cf. ``paths``), and additional sample meta data.
            paths: File path template strings of input data files. The format string may contain keys ``prefix``,
                when a ``prefix`` path has been specified, and ``table`` column names. The dictionary keys of this
                argument are used as sample data dictionary keys for the respective file paths. When the path value
                is a string which matches exactly the name of a ``table`` column, the value of this column is used
                without configuring a file path template string. This is useful when the input ``table`` already
                specifies the file paths for each sample. Instead of a string, the dictionary value can be a
                callable function instead, which takes the ``table`` row values as keyword arguments, and must return
                the respectively formatted input file path string. When no ``paths`` are given, the dataset samples
                only contain the meta-data from the input ``table`` columns.
            prefix: Root directory of input file paths starting with ``"{prefix}/"``.
                If ``None`` and ``table`` is a file path, it is set to the directory containing the index table.
                Otherwise, template file path strings may not contain a ``{prefix}`` key if ``None``.
            transforms: Data preprocessing and augmentation transforms.
            kwargs: Additional format arguments used in addition to ``prefix`` and ``table`` column values.

        """
        if isinstance(table, (str, Path)):
            if prefix is None:
                path = Path(table).absolute()
                prefix = path.parent
            elif prefix:
                prefix = Path(prefix).absolute()
                path = prefix / Path(table)
            else:
                path = Path(table).absolute()
            table = read_table(path)
        if not isinstance(table, pd.DataFrame):
            raise TypeError(
                f"{type(self).__name__}() 'table' must be pandas.DataFrame or file path"
            )
        df: pd.DataFrame = table
        paths = {} if paths is None else dict(paths)
        if "meta" in df.columns:
            raise ValueError(
                f"{type(self).__name__} 'table' contains column with reserved name 'meta'"
            )
        if "meta" in paths:
            raise ValueError(
                f"{type(self).__name__} 'paths' contains reserved 'meta' key"
            )
        prefix = Path(prefix).absolute() if prefix else None
        self.table = df
        self.paths = paths
        self.prefix = prefix
        self.kwargs = kwargs
        super().__init__(transforms=transforms)

    def __len__(self) -> int:
        """Number of samples in dataset."""
        return len(self.table)

    def row(self, index: int) -> Dict[str, Any]:
        """Get i-th table row values."""
        return self.table.iloc[index].to_dict()

    def sample(self, index: int) -> Dict[str, Any]:
        """Input file paths and/or meta-data of i-th sample in dataset."""
        meta = self.row(index)
        if not self.paths:
            return meta
        data = {}
        args = {"prefix": str(self.prefix)} if self.prefix else {}
        args["index"] = index
        args["index+1"] = index + 1
        args["index + 1"] = index + 1
        args.update(self.kwargs)
        args.update(meta)
        for name, path in self.paths.items():
            if callable(path):
                path = path(**args)
            elif path in meta:
                path = meta[path]
            else:
                path = path.format(**args)
            if not path:
                continue
            path = str(path)
            data[name] = path
            meta[name] = path
        data["meta"] = meta
        return data

    def samples(self) -> Iterable[Dict[str, Any]]:
        """Get iterable over untransformed dataset samples."""

        class DatasetSampleIterator(object):
            def __init__(self, dataset: MetaDataset):
                self.dataset = dataset
                self.index = -1

            def __iter__(self) -> Iterator[Dict[str, Any]]:
                self.index = 0
                return self

            def __next__(self) -> Dict[str, Any]:
                if self.index >= len(self.dataset):
                    raise StopIteration
                sample = self.dataset.sample(self.index)
                self.index += 1
                return sample

        return DatasetSampleIterator(self)


@dataclass
class ImageDatasetConfig(DataclassConfig):
    """Configuration of image dataset."""

    table: PathStr
    prefix: Optional[PathStr] = None
    images: Mapping[str, PathStr] = field(default_factory=dict)
    transforms: Mapping[str, ImageTransformConfig] = field(default_factory=dict)

    @classmethod
    def _from_dict(
        cls, arg: Mapping[str, Any], parent: Optional[Path] = None
    ) -> ImageDatasetConfig:
        """Create configuration from dictionary.

        This function optionally re-organizes the dictionary entries to conform to the dataclass layout.
        It allows the image data transforms to be specified as separate "transforms" entry for each image.
        In this case, the image file path template string must given by the "path" dictionary entry.
        Additionally, a "read" image transform is added when a "dtype" or "device" is specified on which
        the image data is loaded and preprocessed can also be specified alongside the file "path".
        Any image "transforms" specified at the top-level are applied after any "transforms" specified
        underneath the "images" key.

        """
        arg = dict(arg)
        images = arg.pop("images", None) or {}
        transforms = arg.pop("transforms", None) or {}
        image_paths = {}
        for name, value in images.items():
            dtype = None
            device = None
            if isinstance(value, Mapping):
                if "path" not in value:
                    raise ValueError(
                        f"{cls.__name__}.from_dict() 'images' key '{name}' dict must contain 'path' entry"
                    )
                path = value["path"]
                dtype = value.get("dtype", dtype)
                device = value.get("device", device)
                image_transforms = value.get("transforms") or []
                if not isinstance(image_transforms, Sequence):
                    raise TypeError(
                        f"{cls.__name__}.from_dict() image 'transforms' value must be Sequence"
                    )
            elif is_path_str(value):
                path = Path(value).as_posix()
                image_transforms = []
            else:
                raise ValueError(
                    f"{cls.__name__}.from_dict() 'images' key '{name}' must be PathStr or dict with 'path' entry"
                )
            if name in transforms:
                item_transforms = transforms[name]
                if not isinstance(item_transforms, Sequence):
                    raise TypeError(
                        f"{cls.__name__}.from_dict() 'transforms' dict value must be Sequence"
                    )
                item_transforms = list(item_transforms)
            else:
                item_transforms = []
            image_transforms = image_transforms + item_transforms
            if dtype or device:
                image_transforms = prepend_read_image_transform(
                    image_transforms, dtype=dtype, device=device
                )
            transforms[name] = image_transforms
            image_paths[name] = path
        arg["images"] = {k: v for k, v in image_paths.items() if v}
        arg["transforms"] = {k: v for k, v in transforms.items() if v}
        return super()._from_dict(arg, parent)


class ImageDataset(MetaDataset):
    """Configurable image dataset."""

    @classmethod
    def from_config(cls, config: ImageDatasetConfig) -> ImageDataset:
        transforms = []
        for image_name in config.images:
            image_transforms_config = config.transforms.get(image_name, [])
            image_transforms_config = prepend_read_image_transform(
                image_transforms_config
            )
            item_transforms = image_transforms(image_transforms_config, key=image_name)
            transforms.extend(item_transforms)
        return cls(
            config.table,
            paths=config.images,
            prefix=config.prefix,
            transforms=transforms,
        )


class GroupDataset(paddle.io.Dataset):
    """Group samples in dataset."""

    def __init__(
        self,
        dataset: MetaDataset,
        groupby: Union[Sequence[str], str],
        sortby: Optional[Union[Sequence[str], str]] = None,
        ascending: bool = True,
    ) -> None:
        super().__init__()
        indices = []
        df = dataset.table
        if sortby:
            df = df.sort_values(sortby, ascending=ascending)
        groups = df.groupby(groupby)
        for _, group in groups:
            assert isinstance(group, pd.DataFrame)
            ilocs = [row[0] for row in group.itertuples(index=True)]
            indices.append(ilocs)
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Subset[Dict[str, Any]]:
        indices = self.indices[index]
        return Subset(dataset=self.dataset, indices=indices)


class JoinDataset(Dataset):
    """Join dict entries from one or more datasets in a single dict."""

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__()
        datasets = list(datasets)
        if not all(len(dataset) == len(datasets[0]) for dataset in datasets):
            raise ValueError("JoinDataset() 'datasets' must have the same size")
        self.datasets = datasets

    def __len__(self) -> int:
        datasets = self.datasets
        return len(datasets[0]) if datasets else 0

    def sample(self, index: int) -> Sample:
        sample = {}
        for i, dataset in enumerate(self.datasets):
            data = dataset[index]
            if not isinstance(data, dict):
                if is_namedtuple(data):
                    data = data._asdict()
                else:
                    data = {str(i): data}
            for key, value in data.items():
                current = sample.get(key, None)
                if current is not None and current != value:
                    raise ValueError(
                        "JoinDataset() encountered ambiguous duplicate key '{key}'"
                    )
                sample[key] = value
        return sample


def read_table(path: PathStr) -> pd.DataFrame:
    """Read dataset index table."""
    path = Path(path).absolute()
    if path.suffix.lower() == ".h5":
        return pd.read_hdf(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, comment="#", skip_blank_lines=True, delimiter="\t")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, comment="#", skip_blank_lines=True)
    raise NotImplementedError(
        f"read_table() does not support {path.suffix} file format"
    )
