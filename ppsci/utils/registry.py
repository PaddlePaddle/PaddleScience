# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

import logging
from typing import Dict
from typing import Optional
from typing import Type

import paddle

import ppsci.data.dataset as dataset

_REGISTRY: Dict[str, Dict[str, Type]] = {
    "dataset": {},
    "model": {},
    "loss": {},
    "constraint": {},
    "validator": {},
    "visualizer": {},
    "optimizer": {},
    "lr_scheduler": {},
    "transform": {},
    "geometry": {},
    "solver": {},
    "equation": {},
}


def register_cls_to_module(
    module_name: str, cls: Type, category: Optional[str] = None
) -> None:
    """
    Register a class to specified module in registry.

    Args:
        module_name (str): Module name to register class to.
        cls (Type): Class to be registered.
        category (str, optional): Category for the class. If None, inferred from class.
    """
    cls_name = cls.__name__

    if category is None:
        category = _infer_category(cls)

    if category in _REGISTRY:
        if cls_name in _REGISTRY[category]:
            logging.warning(
                f"{cls_name} is already registered in {category}. Overwriting..."
            )
        _REGISTRY[category][cls_name] = cls
        setattr(dataset, cls_name, cls)
    else:
        raise ValueError(f"Category {category} is not supported for registry.")


def _infer_category(cls: Type) -> str:
    """
    Infer the category of a class based on its inheritance.

    Args:
        cls (Type): Class to infer category for.

    Returns:
        str: Category name.
    """
    if issubclass(cls, paddle.io.Dataset):
        return "dataset"
    elif hasattr(cls, "__module__") and "model" in cls.__module__:
        return "model"
    elif hasattr(cls, "__module__") and "loss" in cls.__module__:
        return "loss"
    elif hasattr(cls, "__module__") and "constraint" in cls.__module__:
        return "constraint"
    elif hasattr(cls, "__module__") and "validator" in cls.__module__:
        return "validator"
    elif hasattr(cls, "__module__") and "visualizer" in cls.__module__:
        return "visualizer"
    else:
        raise ValueError(f"Class {cls} is not supported for registry.")


def get_registry(category: str) -> Dict[str, Type]:
    """
    Get registry for specified category.

    Args:
        category (str): Category name.

    Returns:
        Dict[str, Type]: Registry dictionary for the category.
    """
    if category not in _REGISTRY:
        raise ValueError(f"Category {category} is not registered.")
    return _REGISTRY[category]


def get_class(category: str, name: str) -> Type:
    """
    Get class from registry by category and name.

    Args:
        category (str): Category name.
        name (str): Class name.

    Returns:
        Type: Registered class.
    """
    registry = get_registry(category)
    if name not in registry:
        available_classes = list(registry.keys())
        raise KeyError(
            f"Class {name} is not found in category {category}. "
            f"Available classes: {available_classes}"
        )
    return registry[name]


def register_to_dataset(cls: Type) -> Type:
    """
    Decorator for registering dataset classes.

    Args:
        cls (Type): Dataset class to register.

    Returns:
        Type: The same class.
    """
    if not issubclass(cls, paddle.io.Dataset):
        raise ValueError(
            f"The registered class '{cls.__name__}' should be inherited from `paddle.io.Dataset`, "
            f"but got {cls.__bases__}."
        )

    register_cls_to_module("dataset", cls, "dataset")
    return cls
