import copy
import inspect
import typing
from collections import defaultdict
from dataclasses import dataclass, field, make_dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import paddle

EXCLUDE = {'self', 'args', 'kwargs'}

MAPPING = {
    paddle.nn.Layer: Any,
    paddle.Tensor: Any,
}

try:
    from omegaconf import MISSING
except Exception:
    MISSING = '???'

try:
    import hydra  # noqa
    WITH_HYDRA = True
except Exception:
    WITH_HYDRA = False

if not typing.TYPE_CHECKING and WITH_HYDRA:
    from hydra.core.config_store import ConfigStore

    def get_node(cls: Union[str, Any]) -> Optional[Any]:
        if (not isinstance(cls, str)
                and cls.__module__ in {'builtins', 'typing'}):
            return None

        def _get_candidates(repo: Dict[str, Any]) -> List[Any]:
            outs: List[Any] = []
            for key, value in repo.items():
                if isinstance(value, dict):
                    outs.extend(_get_candidates(value))
                elif getattr(value.node._metadata, 'object_type', None) == cls:
                    outs.append(value.node)
                elif getattr(value.node._metadata, 'orig_type', None) == cls:
                    outs.append(value.node)
                elif isinstance(cls, str) and key == f'{cls}.yaml':
                    outs.append(value.node)

            return outs

        candidates = _get_candidates(get_config_store().repo)

        if len(candidates) > 1:
            raise ValueError(f"Found multiple entries in the configuration "
                             f"store for the same node '{candidates[0].name}'")

        return candidates[0] if len(candidates) == 1 else None

    def dataclass_from_class(cls: Union[str, Any]) -> Optional[Any]:
        node = get_node(cls)
        return node._metadata.object_type if node is not None else None

    def class_from_dataclass(cls: Union[str, Any]) -> Optional[Any]:
        node = get_node(cls)
        return node._metadata.orig_type if node is not None else None

else:

    class Singleton(type):
        _instances: Dict[type, Any] = {}

        def __call__(cls, *args: Any, **kwargs: Any) -> Any:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
                return instance
            return cls._instances[cls]

    @dataclass
    class Metadata:
        orig_type: Optional[Any] = None

    @dataclass
    class ConfigNode:
        name: str
        node: Any
        group: Optional[str] = None
        _metadata: Metadata = field(default_factory=Metadata)

    class ConfigStore(metaclass=Singleton):
        def __init__(self) -> None:
            self.repo: Dict[str, Any] = defaultdict(dict)

        @classmethod
        def instance(cls, *args: Any, **kwargs: Any) -> 'ConfigStore':
            return cls(*args, **kwargs)

        def store(
            self,
            name: str,
            node: Any,
            group: Optional[str] = None,
            orig_type: Optional[Any] = None,
        ) -> None:
            cur = self.repo
            if group is not None:
                cur = cur[group]
            if name in cur:
                raise KeyError(f"Configuration '{name}' already registered. "
                               f"Please store it under a different group.")
            metadata = Metadata(orig_type=orig_type)
            cur[name] = ConfigNode(name, node, group, metadata)

    def get_node(cls: Union[str, Any]) -> Optional[ConfigNode]:
        if (not isinstance(cls, str)
                and cls.__module__ in {'builtins', 'typing'}):
            return None

        def _get_candidates(repo: Dict[str, Any]) -> List[ConfigNode]:
            outs: List[ConfigNode] = []
            for key, value in repo.items():
                if isinstance(value, dict):
                    outs.extend(_get_candidates(value))
                elif value.node == cls:
                    outs.append(value)
                elif value._metadata.orig_type == cls:
                    outs.append(value)
                elif isinstance(cls, str) and key == cls:
                    outs.append(value)

            return outs

        candidates = _get_candidates(get_config_store().repo)

        if len(candidates) > 1:
            raise ValueError(f"Found multiple entries in the configuration "
                             f"store for the same node '{candidates[0].name}'")

        return candidates[0] if len(candidates) == 1 else None

    def dataclass_from_class(cls: Union[str, Any]) -> Optional[Any]:
        node = get_node(cls)
        return node.node if node is not None else None

    def class_from_dataclass(cls: Union[str, Any]) -> Optional[Any]:
        node = get_node(cls)
        return node._metadata.orig_type if node is not None else None


def map_annotation(
    annotation: Any,
    mapping: Optional[Dict[Any, Any]] = None,
) -> Any:
    origin = getattr(annotation, '__origin__', None)
    args: Tuple[Any, ...] = getattr(annotation, '__args__', tuple())
    if origin in {Union, list, dict, tuple}:
        assert origin is not None
        args = tuple(map_annotation(a, mapping) for a in args)
        if type(annotation).__name__ == 'GenericAlias':
            annotation = origin[args]
        else:
            annotation = copy.copy(annotation)
            annotation.__args__ = args

        return annotation

    if mapping is not None and annotation in mapping:
        return mapping[annotation]

    out = dataclass_from_class(annotation)
    if out is not None:
        return out

    return annotation


def to_dataclass(
    cls: Any,
    base_cls: Optional[Any] = None,
    with_target: Optional[bool] = None,
    map_args: Optional[Dict[str, Tuple]] = None,
    exclude_args: Optional[List[str]] = None,
    strict: bool = False,
) -> Any:
    fields = []

    params = inspect.signature(cls.__init__).parameters

    if strict:
        keys = set() if map_args is None else set(map_args.keys())
        if exclude_args is not None:
            keys |= {arg for arg in exclude_args if isinstance(arg, str)}
        diff = keys - set(params.keys())
        if len(diff) > 0:
            raise ValueError(f"Expected input argument(s) {diff} in "
                             f"'{cls.__name__}'")

    for i, (name, arg) in enumerate(params.items()):
        if name in EXCLUDE:
            continue
        if exclude_args is not None:
            if name in exclude_args or i in exclude_args:
                continue
        if base_cls is not None:
            if name in base_cls.__dataclass_fields__:
                continue

        if map_args is not None and name in map_args:
            fields.append((name, ) + map_args[name])
            continue

        annotation, default = arg.annotation, arg.default
        annotation = map_annotation(annotation, mapping=MAPPING)

        if annotation != inspect.Parameter.empty:
            origin = getattr(annotation, '__origin__', None)
            args = getattr(annotation, '__args__', [])
            if origin == Union and type(None) in args and len(args) > 2:
                annotation = Optional[Any]
            elif origin == Union and type(None) not in args:
                annotation = Any
            elif origin == list:
                if getattr(args[0], '__origin__', None) == Union:
                    annotation = List[Any]
            elif origin == dict:
                if getattr(args[1], '__origin__', None) == Union:
                    annotation = Dict[args[0], Any]
        else:
            annotation = Any

        if str(default) == "<required parameter>":
            default = field(default=MISSING)
        elif default != inspect.Parameter.empty:
            if isinstance(default, (list, dict)):
                def wrapper(default: Any) -> Callable[[], Any]:
                    return lambda: default

                default = field(default_factory=wrapper(default))
        else:
            default = field(default=MISSING)

        fields.append((name, annotation, default))

    with_target = base_cls is not None if with_target is None else with_target
    if with_target:
        full_cls_name = f'{cls.__module__}.{cls.__qualname__}'
        fields.append(('_target_', str, field(default=full_cls_name)))

    return make_dataclass(cls.__qualname__, fields=fields,
                          bases=() if base_cls is None else (base_cls, ))


def get_config_store() -> ConfigStore:
    return ConfigStore.instance()


def clear_config_store() -> ConfigStore:
    config_store = get_config_store()
    for key in list(config_store.repo.keys()):
        if key != 'hydra' and not key.endswith('.yaml'):
            del config_store.repo[key]
    return config_store


def register(
    cls: Optional[Any] = None,
    data_cls: Optional[Any] = None,
    group: Optional[str] = None,
    **kwargs: Any,
) -> Union[Any, Callable]:
    if cls is not None:
        name = cls.__name__

        if get_node(cls):
            raise ValueError(f"The class '{name}' is already registered in "
                             "the global configuration store")

        if data_cls is None:
            data_cls = to_dataclass(cls, **kwargs)
        elif get_node(data_cls):
            raise ValueError(
                f"The data class '{data_cls.__name__}' is already registered "
                f"in the global configuration store")

        if not typing.TYPE_CHECKING and WITH_HYDRA:
            get_config_store().store(name, data_cls, group)
            get_node(name)._metadata.orig_type = cls
        else:
            get_config_store().store(name, data_cls, group, cls)

        return data_cls

    def bounded_register(cls: Any) -> Any:
        register(cls=cls, data_cls=data_cls, group=group, **kwargs)
        return cls

    return bounded_register


@dataclass
class Transform:
    pass


@dataclass
class Dataset:
    pass


@dataclass
class Model:
    pass


@dataclass
class Optimizer:
    pass


@dataclass
class LRScheduler:
    pass


@dataclass
class Config:
    dataset: Dataset = MISSING
    model: Model = MISSING
    optim: Optimizer = MISSING
    lr_scheduler: Optional[LRScheduler] = None


def fill_config_store() -> None:
    config_store = get_config_store()

    # Example of PaddlePaddle transform registration:
    # Replace this with paddle transforms as needed.
    # Example:
    # config_store.store('NormalizeFeatures', group='transform', node=Transform())

    # Example of registering Paddle datasets:
    # Replace this with paddle dataset registrations as needed.
    # config_store.store('CIFAR10', group='dataset', node=Dataset())

    # Example of registering Paddle models:
    # Replace this with paddle model registrations as needed.
    # config_store.store('ResNet50', group='model', node=Model())

    # Example of registering Paddle optimizers:
    for optimizer_name in dir(paddle.optimizer):
        if not optimizer_name.startswith('_'):
            cls = getattr(paddle.optimizer, optimizer_name)
            if inspect.isclass(cls):
                data_cls = to_dataclass(cls, base_cls=Optimizer, exclude_args=['parameters'])
                config_store.store(optimizer_name, group='optimizer', node=data_cls)

    # Example of registering Paddle learning rate schedulers:
    for scheduler_name in dir(paddle.optimizer.lr):
        if not scheduler_name.startswith('_'):
            cls = getattr(paddle.optimizer.lr, scheduler_name)
            if inspect.isclass(cls):
                data_cls = to_dataclass(cls, base_cls=LRScheduler)
                config_store.store(scheduler_name, group='lr_scheduler', node=data_cls)

    config_store.store('config', node=Config)
