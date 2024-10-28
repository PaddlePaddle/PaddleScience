import paddle
from deepali.core import functional as U
from deepali.data import ImageBatch
from deepali.spatial import ImageTransformer
from deepali.spatial import Translation


# There is a loss of accuracy in the calculation, so the atol parameter of paddle.allclose is used.
def test_spatial_image_transformer() -> None:
    # Generate sample image
    image = ImageBatch(U.cshape_image(size=(65, 33), center=(32, 16), sigma=1, dtype="float32"))
    assert tuple(image.shape) == (1, 1, 33, 65)

    # Apply identity transformation
    offset = paddle.to_tensor(data=[0, 0], dtype="float32")
    translation = Translation(image.grid(), params=offset.unsqueeze(axis=0))
    transformer = ImageTransformer(translation)
    warped = transformer.forward(image)
    # assert warped.equal(y=image).astype("bool").all()
    assert paddle.allclose(warped, image, atol=1e-07)

    # Shift image in world (and image) space to the left
    offset = paddle.to_tensor(data=[0.5, 0], dtype="float32")
    translation = Translation(image.grid(), params=offset.unsqueeze(axis=0))
    transformer = ImageTransformer(translation)
    warped = transformer.forward(image)
    expected = U.cshape_image(size=(65, 33), center=(32 - 16, 16), sigma=1, dtype=image.dtype)
    # assert paddle.allclose(x=warped, y=expected).item()
    assert paddle.allclose(warped, expected, atol=1e-07)

    # Shift image in world (and image) space to the right
    offset = paddle.to_tensor(data=[-0.5, 0], dtype="float32")
    translation = Translation(image.grid(), params=offset.unsqueeze(axis=0))
    transformer = ImageTransformer(translation)
    warped = transformer.forward(image)
    expected = U.cshape_image(size=(65, 33), center=(32 + 16, 16), sigma=1, dtype=image.dtype)
    # assert paddle.allclose(x=warped, y=expected).item()
    assert paddle.allclose(warped, expected, atol=1e-07)

    # Shift target sampling grid in world space and sample input with identity transform.
    # This results in a shift of the image in the image space though world positions are unchanged.
    target = image.grid().center(32 - 16, 0)
    assert not target.same_domain_as(image.grid())

    assert paddle.allclose(
        x=image.grid().index_to_world([32, 16]), y=target.index_to_world([16, 16])
    ).item()

    offset = paddle.to_tensor(data=[0, 0], dtype="float32")
    translation = Translation(image.grid(), params=offset.unsqueeze(axis=0))
    transformer = ImageTransformer(translation, target=target, source=image.grid())
    warped = transformer.forward(image)
    expected = U.cshape_image(size=(65, 33), center=(32 - 16, 16), sigma=1, dtype=image.dtype)
    # assert paddle.allclose(x=warped, y=expected).item()
    assert paddle.allclose(warped, expected, atol=1e-07)
