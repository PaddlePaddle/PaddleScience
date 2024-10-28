# Common transforms

## Parametric transforms

Spatial transforms which are not simply composed of other transformations are called parametric.
These are derived from one of the base classes ({class}`.SpatialTransform`, {class}`.LinearTransform`,
{class}`.NonRigidTransform`) and one of the following mix-ins depending on whether the transformation
can be readily inverted during evaluation of its {meth}`.SpatialTransform.forward` method or not.

- {class}`.ParametricTransform`
- {class}`.InvertibleParametricTransform`

```{eval-rst}
.. automodule:: deepali.spatial.parametric
    :noindex:
```

## Linear transforms

```{eval-rst}
.. automodule:: deepali.spatial.linear
    :noindex:
```

The most generic linear transform is a {class}`.HomogeneousTransform`, which is parameterized by
a homogeneous coordinate transformation matrix. This parameterization is less suitable for direct
optimization, however, because there is no control over the individual elementary transformations
making up a linear homogeneous coordinate transformation. But it can be used when the matrix
is derived from the prediction of a neural network or computed by other means.

### Elementary linear transforms

Elementary homogeneous coordinate transformations.

- {class}`.Translation`
- {class}`.EulerRotation`
- {class}`.QuaternionRotation`
- {class}`.IsotropicScaling`
- {class}`.AnisotropicScaling`
- {class}`.Shearing`

### Composite linear transforms

Spatial transforms composed of two or more elementary linear transforms.

- {class}`.RigidTransform`
- {class}`.RigidQuaternionTransform`
- {class}`.SimilarityTransform`
- {class}`.AffineTransform`
- {class}`.FullAffineTransform`

## Non-rigid transforms

```{eval-rst}
.. automodule:: deepali.spatial.nonrigid
    :noindex:
```

### Displacement fields

Spatial transforms which directly represent a displacement vector field.

- {class}`.DisplacementFieldTransform`
- {class}`.FreeFormDeformation`

### Diffeomorphic transforms

Spatial transforms based on the integration of a velocity vector field.

- {class}`.StationaryVelocityFieldTransform`
- {class}`.StationaryVelocityFreeFormDeformation`
