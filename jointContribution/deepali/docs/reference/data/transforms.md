# Data transforms

```{eval-rst}
.. automodule:: deepali.data.transforms
    :noindex:
```

## Item transforms

```{eval-rst}
.. autoapisummary::

    deepali.data.transforms.item.ItemTransform
    deepali.data.transforms.item.ItemwiseTransform

```

## Image transforms

```{eval-rst}
.. autoapisummary::

    deepali.data.transforms.image.AvgPoolImage
    deepali.data.transforms.image.CastImage
    deepali.data.transforms.image.CenterCropImage
    deepali.data.transforms.image.CenterPadImage
    deepali.data.transforms.image.ClampImage
    deepali.data.transforms.image.ImageToTensor
    deepali.data.transforms.image.NarrowImage
    deepali.data.transforms.image.NormalizeImage
    deepali.data.transforms.image.ReadImage
    deepali.data.transforms.image.ResampleImage
    deepali.data.transforms.image.RescaleImage
    deepali.data.transforms.image.ResizeImage

```

Image transforms can also be instantiated based on a configuration, e.g., in YAML format:

```{eval-rst}
.. autoapifunction:: deepali.data.transforms.image.image_transforms
    :noindex:
```
