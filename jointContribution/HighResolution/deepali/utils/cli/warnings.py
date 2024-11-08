import warnings


def filter_warning_of_experimental_named_tensors_feature() -> None:
    """Filter out warning reminding that named tensors are still an experimental feature."""
    warnings.filterwarnings(
        "ignore",
        message="Named tensors and all their associated APIs are an experimental feature and subject to change.",
        category=UserWarning,
    )
