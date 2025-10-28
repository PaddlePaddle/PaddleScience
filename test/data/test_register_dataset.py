import paddle
import pytest


def test_register_new_dataset(caplog):
    from ppsci.data import register_to_dataset

    class NewDataset(paddle.io.Dataset):
        pass

    register_to_dataset(NewDataset)


def test_register_new_dataset_but_not_inherit(caplog):
    from ppsci.data import register_to_dataset

    class NotInheritDataset:
        pass

    with pytest.raises(ValueError, match="is not supported for registry"):
        register_to_dataset(NotInheritDataset)


if __name__ == "__main__":
    pytest.main()
