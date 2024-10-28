# Composite transforms

Spatial transforms which are composed of other spatial transforms are referred to as composite
transforms with common type {class}`.CompositeTransform`. Two main composite types are distinguished
based on how these compose the individual spatial transforms: {class}`.MultiLevelTransform` and
{class}`.SequentialTransform`. A special kind of sequential transform is the {class}`.GenericSpatialTransform`,
which can be configured to represent most common spatial transformation models.

## Multi-level transform

```{eval-rst}
.. automodule:: deepali.spatial.MultiLevelTransform
    :noindex:
```

## Sequential transform

```{eval-rst}
.. automodule:: deepali.spatial.SequentialTransform
    :noindex:
```

## Generic transform

A {class}`.GenericSpatialTransform` sequentially applies a configured set of transformations.

### Configuration

The types of transformations and their order of application can be configured via a {class}`.TransformConfig`,
which can be programmatically created or initialized from a configuration file in JSON or YAML format
(cf. {meth}`.DataclassConfig.read`).

```{eval-rst}
.. autoclass:: deepali.spatial.TransformConfig
    :noindex:
    :members:
```

### Parameters

The parameters of a {class}`.GenericSpatialTransform` are either the parameters specified as ``params``, or
a dictionary of parameters inferred by a neural network. The keys of the ``params`` dictionary must match
the names of the configured spatial transforms making up the {class}`.SequentialTransform`.

Keys for affine component parameters:

- ``A``: ``"affine"``
- ``K``: ``"shearing"``
- ``T``: ``"translation"`` or ``"offset"``
- ``R``: ``"rotation"`` or ``"angles"``
- ``S``: ``"scaling"`` or ``"scales"``
- ``Q``: ``"quaternion"``

Keys of non-rigid component parameters:

- ``"nonrigid"`` or ``"vfield"``.
