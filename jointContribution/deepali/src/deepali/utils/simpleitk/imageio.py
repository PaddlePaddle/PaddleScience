import SimpleITK as sitk
from deepali.core.pathlib import PathUri
from deepali.core.pathlib import unlink_or_mkdir
from deepali.core.storage import StorageObject


def read_image(path: PathUri) -> sitk.Image:
    r"""Read image from file."""
    with StorageObject.from_path(path) as obj:
        if not obj.is_file():
            raise FileNotFoundError(f"No such file or no access: '{path}'")
        obj.pull(force=True)
        image = sitk.ReadImage(str(obj.path))
    return image


def write_image(image: sitk.Image, path: PathUri, compress: bool = True):
    r"""Write image to file."""
    with StorageObject.from_path(path) as obj:
        path = unlink_or_mkdir(obj.path)
        sitk.WriteImage(image, str(path), compress)
        obj.push(force=True)
