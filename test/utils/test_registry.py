# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import patch

import paddle
import pytest

from ppsci.utils import registry


def test_register_cls_to_module():
    """Test register_cls_to_module function."""
    # Clear registry to avoid test interference
    original_registry = registry._REGISTRY.copy()

    class DummyDataset(paddle.io.Dataset):
        def __init__(self):
            pass

        def __len__(self):
            return 0

        def __getitem__(self, idx):
            return {}

    # Test normal registration
    registry.register_cls_to_module("test_module", DummyDataset, "dataset")

    # Verify class is registered
    assert DummyDataset.__name__ in registry._REGISTRY["dataset"]
    assert registry._REGISTRY["dataset"][DummyDataset.__name__] is DummyDataset

    # Test override warning
    with patch("ppsci.utils.registry.logging.warning") as mock_warning:
        registry.register_cls_to_module("test_module", DummyDataset, "dataset")
        mock_warning.assert_called_once()

    # Restore original registry
    registry._REGISTRY = original_registry


def test_infer_category():
    """Test _infer_category function."""
    original_registry = registry._REGISTRY.copy()

    class TestDataset(paddle.io.Dataset):
        def __init__(self):
            pass

        def __len__(self):
            return 0

        def __getitem__(self, idx):
            return {}

    # Test dataset category inference
    category = registry._infer_category(TestDataset)
    assert category == "dataset"
    registry._REGISTRY = original_registry


def test_infer_category_non_paddle_dataset():
    """Test _infer_category function with non-paddle.io.Dataset class."""
    original_registry = registry._REGISTRY.copy()

    # Test category inference for non-paddle.io.Dataset subclass (should raise exception)
    class NonDatasetClass:
        pass

    # Verify that non-dataset class without specific module identifier raises exception
    with pytest.raises(ValueError, match="is not supported for registry"):
        registry._infer_category(NonDatasetClass)

    # Test decorator with non-dataset class raises exception
    with pytest.raises(
        ValueError, match="should be inherited from `paddle.io.Dataset`"
    ):

        @registry.register_to_dataset
        class NonDatasetClass:
            pass

    registry._REGISTRY = original_registry


def test_get_registry():
    """Test get_registry function."""
    original_registry = registry._REGISTRY.copy()

    # Test getting existing category
    dataset_registry = registry.get_registry("dataset")
    assert isinstance(dataset_registry, dict)

    # Test getting non-existent category should raise exception
    with pytest.raises(ValueError):
        registry.get_registry("nonexistent_category")

    registry._REGISTRY = original_registry


def test_get_class():
    """Test get_class function."""
    # Clear registry to avoid test interference
    original_registry = registry._REGISTRY.copy()

    class TestDataset(paddle.io.Dataset):
        def __init__(self):
            pass

        def __len__(self):
            return 0

        def __getitem__(self, idx):
            return {}

    # Register test class
    registry.register_cls_to_module("test_module", TestDataset, "dataset")

    # Test getting registered class
    retrieved_class = registry.get_class("dataset", "TestDataset")
    assert retrieved_class is TestDataset

    # Test getting non-existent class should raise exception
    with pytest.raises(KeyError):
        registry.get_class("dataset", "NonExistentClass")

    # Restore original registry
    registry._REGISTRY = original_registry


def test_register_to_dataset_decorator():
    """Test register_to_dataset decorator."""
    # Clear registry to avoid test interference
    original_registry = registry._REGISTRY.copy()

    # Test correct dataset class decoration
    @registry.register_to_dataset
    class MyDataset(paddle.io.Dataset):
        def __init__(self):
            pass

        def __len__(self):
            return 0

        def __getitem__(self, idx):
            return {}

    # Verify class is correctly registered
    assert "MyDataset" in registry._REGISTRY["dataset"]
    assert registry._REGISTRY["dataset"]["MyDataset"] is MyDataset

    # Test non-dataset class decoration should raise exception
    with pytest.raises(
        ValueError, match="should be inherited from `paddle.io.Dataset`"
    ):

        @registry.register_to_dataset
        class NotADataset:
            pass

    registry._REGISTRY = original_registry


def test_register_cls_to_module_adds_to_dataset_module():
    """Test that register_cls_to_module adds class to ppsci.data.dataset module for eval access."""
    original_registry = registry._REGISTRY.copy()

    class TestDataset(paddle.io.Dataset):
        def __init__(self):
            pass

        def __len__(self):
            return 0

        def __getitem__(self, idx):
            return {}

    # Register the class to dataset category
    registry.register_cls_to_module("test_module", TestDataset, "dataset")

    # Verify class is added to ppsci.data.dataset module
    import ppsci.data.dataset as dataset_module

    assert hasattr(dataset_module, "TestDataset")
    assert getattr(dataset_module, "TestDataset") is TestDataset

    # Verify eval can access the class from dataset module
    import ppsci.data.dataset as dataset_module

    eval_result = eval("TestDataset", dataset_module.__dict__)
    assert eval_result is TestDataset

    # Restore original registry
    registry._REGISTRY = original_registry


if __name__ == "__main__":
    pytest.main()
