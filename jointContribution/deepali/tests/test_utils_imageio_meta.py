import numpy as np
import pytest
import SimpleITK as sitk
from deepali.core.affine import euler_rotation_matrix
from deepali.core.tempfile import temp_dir
from deepali.utils.imageio.meta import meta_image_bytes
from deepali.utils.imageio.meta import read_meta_image_from_fileobj


@pytest.fixture
def rand_image() -> sitk.Image:
    rng = np.random.default_rng(123456789)
    data = (255 * rng.random((32, 64, 128))).astype(np.uint8)
    im = sitk.GetImageFromArray(data)
    rot = euler_rotation_matrix((0.1, 0.6, -0.4))
    im.SetDirection(rot.flatten().tolist())
    im.SetSpacing((1.2, 0.6, 0.25))
    return im


def test_read_meta_image(rand_image: sitk.Image) -> None:
    r"""Test reading image from MetaIO file bytes."""
    image_data = sitk.GetArrayViewFromImage(rand_image)
    with temp_dir(prefix="test_read_meta_image_") as temp:
        path = temp / "rand_image.mha"
        compress = True
        sitk.WriteImage(rand_image, str(path), compress)
        with path.open("rb") as f:
            data, meta = read_meta_image_from_fileobj(f)
        assert meta.get("NDims") == 3
        assert meta.get("ObjectType") == "Image"
        assert np.all(meta.get("DimSize") == rand_image.GetSize())
        assert meta.get("ElementNumberOfChannels") == 1
        assert np.all(meta.get("ElementSpacing") == rand_image.GetSpacing())
        assert np.all(meta.get("Offset") == rand_image.GetOrigin())
        assert np.all(meta.get("TransformMatrix").flatten() == rand_image.GetDirection())
        assert np.all(meta.get("CenterOfRotation") == (0, 0, 0))
        assert meta.get("ElementType") == image_data.dtype
        assert np.allclose(image_data, data)


def test_meta_image_bytes(rand_image: sitk.Image) -> None:
    r"""Test serialization of image to MetaIO bytes."""
    image_data = sitk.GetArrayViewFromImage(rand_image)
    with temp_dir(prefix="test_meta_image_bytes_") as temp:
        path = temp / "rand_image.mha"
        compress = True
        sitk.WriteImage(rand_image, str(path), compress)
        assert path.is_file()
        with path.open("rb") as f:
            data, meta = read_meta_image_from_fileobj(f)
        path.unlink()
        assert not path.is_file()
        blob = meta_image_bytes(data, meta)
        assert path.write_bytes(blob) == len(blob)
        assert path.is_file()
        im = sitk.ReadImage(str(path))
        assert np.all(np.equal(im.GetSize(), rand_image.GetSize()))
        assert np.all(np.equal(im.GetSpacing(), rand_image.GetSpacing()))
        assert np.all(np.equal(im.GetDirection(), rand_image.GetDirection()))
        im_data = sitk.GetArrayViewFromImage(im)
        assert image_data.dtype == im_data.dtype
        assert np.allclose(image_data, im_data)
