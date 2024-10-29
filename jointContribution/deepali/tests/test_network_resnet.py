from deepali.networks.resnet import ResNet


def _test_resnet() -> None:
    r"""Test construction of ResNet model."""
    # ResNet-34 for ImageNet classification
    model = ResNet.from_depth(model_depth=34, spatial_dims=2, in_channels=3, num_classes=1000)
    assert type(model) is ResNet
    # summary(model, input_size=(1, 3, 256, 256), depth=10, device="cpu")
