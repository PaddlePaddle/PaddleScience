r"""Auxiliary functions for reading and writing MetaImage files."""
import io
import zlib
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle
from deepali.core.grid import Grid
from deepali.core.pathlib import PathUri
from deepali.core.pathlib import path_suffix
from deepali.core.storage import StorageObject
from deepali.utils import paddle_aux  # noqa

META_IMAGE_SUFFIXES = ".mha", ".mhd"


def has_meta_image_suffix(path: PathUri) -> bool:
    r"""Check whether filename ends in a MetaImage extension."""
    return path_suffix(path).lower() in META_IMAGE_SUFFIXES


def read_meta_image(arg: Union[PathUri, bytes, io.BufferedReader]) -> Tuple[paddle.Tensor, Grid]:
    r"""Read image data and grid attributes from MetaImage file."""
    if isinstance(arg, (Path, str)):
        suffix = path_suffix(arg).lower()
        if suffix == ".mha":
            if isinstance(arg, Path):
                with Path(arg).open("rb") as f:
                    data, meta = read_meta_image_from_fileobj(f)
            else:
                with StorageObject.from_path(arg) as obj:
                    with io.BytesIO(obj.read_bytes()) as f:
                        data, meta = read_meta_image_from_fileobj(f)
        elif suffix == ".mhd":
            try:
                from .sitk import read_sitk_image
            except ImportError:
                raise RuntimeError("SimpleITK is required for reading .mhd MetaImage files")
            return read_sitk_image(arg)
        else:
            raise ValueError(f"MetaImage filename suffix must be one of {META_IMAGE_SUFFIXES}")
    elif isinstance(arg, bytes):
        with io.BytesIO(arg) as f:
            data, meta = read_meta_image_from_fileobj(f)
    elif isinstance(arg, io.BufferedReader):
        data, meta = read_meta_image_from_fileobj(arg)
    else:
        raise TypeError("read_meta_image() 'arg' must be path, bytes, or io.BufferedReader")
    if meta.get("ElementNumberOfChannels", 1) == 1:
        data = np.expand_dims(data, 0)
    else:
        data = np.squeeze(np.swapaxes(np.expand_dims(data, 0), 0, -1), -1)
    size = tuple(meta.get("DimSize", tuple(data.shape)[:0:-1]))
    assert size[::-1] == tuple(data.shape)[1:]
    origin = meta.get("Position", meta.get("Origin", meta.get("Offset")))
    matrix = meta.get("Rotation", meta.get("Orientation", meta.get("TransformMatrix")))
    spacing = meta.get("ElementSpacing")
    grid = Grid(size=size, origin=origin, spacing=spacing, direction=matrix)
    if data.dtype == np.uint16:
        data = data.astype(np.int32)
    elif data.dtype == np.uint32:
        data = data.astype(np.int64)
    return paddle.to_tensor(data=data), grid


def write_meta_image(data: paddle.Tensor, grid: Grid, path: PathUri, compress: bool = True) -> None:
    r"""Write image data and grid attributes to MetaImage file."""
    suffix = path_suffix(path).lower()
    if suffix == ".mha":
        if data.ndim > grid.ndim + 1:
            raise ValueError("write_image() data.ndim must be equal to grid.ndim or grid.ndim + 1")
        meta = {
            "CompressedData": compress,
            "ElementNumberOfChannels": tuple(data.shape)[0],
            "ElementSpacing": grid.spacing().cpu().numpy(),
            "Offset": grid.origin().cpu().numpy(),
            "TransformMatrix": grid.direction().cpu().numpy(),
        }
        data = data.detach().cpu()
        if data.ndim == grid.ndim + 1:
            if tuple(data.shape)[0] > 1:
                data = data.unsqueeze(axis=-1).transpose_(
                    perm=paddle_aux.transpose_aux_func(data.unsqueeze(axis=-1).ndim, 0, -1)
                )
            data = data.squeeze(axis=0)
        blob = meta_image_bytes(data.numpy(), meta)
        with StorageObject.from_path(path) as obj:
            obj.write_bytes(blob)
    elif suffix == ".mhd":
        try:
            from .sitk import write_sitk_image
        except ImportError:
            raise RuntimeError("SimpleITK is required for writing .mhd MetaImage files")
        return write_sitk_image(data, grid, path, compress=compress)
    else:
        raise ValueError(f"MetaImage filename suffix must be one of {META_IMAGE_SUFFIXES}")


# ------------------------------------------------------------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2021 Ali Uneri
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
# IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------------------------------------

MetaData = Dict[str, Any]

# https://itk.org/Wiki/ITK/MetaIO/Documentation#Reference:_Tags_of_MetaImage
META_IMAGE_TAGS = (
    "Comment",  # MET_STRING
    "ObjectType",  # MET_STRING (Image)
    "ObjectSubType",  # MET_STRING
    "TransformType",  # MET_STRING (Rigid)
    "NDims",  # MET_INT
    "Name",  # MET_STRING
    "ID",  # MET_INT
    "ParentID",  # MET_INT
    "CompressedData",  # MET_STRING (boolean)
    "CompressedDataSize",  # MET_INT
    "BinaryData",  # MET_STRING (boolean)
    "BinaryDataByteOrderMSB",  # MET_STRING (boolean)
    "ElementByteOrderMSB",  # MET_STRING (boolean)
    "Color",  # MET_FLOAT_ARRAY[4]
    "Position",  # MET_FLOAT_ARRAY[NDims]
    "Offset",  # == Position
    "Origin",  # == Position
    "Orientation",  # MET_FLOAT_MATRIX[NDims][NDims]
    "Rotation",  # == Orientation
    "TransformMatrix",  # == Orientation
    "CenterOfRotation",  # MET_FLOAT_ARRAY[NDims]
    "AnatomicalOrientation",  # MET_STRING (RAS)
    "ElementSpacing",  # MET_FLOAT_ARRAY[NDims]
    "DimSize",  # MET_INT_ARRAY[NDims]
    "HeaderSize",  # MET_INT
    "HeaderSizePerSlice",  # MET_INT (non-standard tag for handling per slice header)
    "Modality",  # MET_STRING (MET_MOD_CT)
    "SequenceID",  # MET_INT_ARRAY[4]
    "ElementMin",  # MET_FLOAT
    "ElementMax",  # MET_FLOAT
    "ElementNumberOfChannels",  # MET_INT
    "ElementSize",  # MET_FLOAT_ARRAY[NDims]
    "ElementType",  # MET_STRING (MET_UINT)
    "ElementDataFile",  # MET_STRING
)

META_IMAGE_TYPES = {
    "MET_CHAR": np.int8,
    "MET_UCHAR": np.uint8,
    "MET_SHORT": np.int16,
    "MET_USHORT": np.uint16,
    "MET_INT": np.int32,
    "MET_UINT": np.uint32,
    "MET_LONG": np.int64,
    "MET_ULONG": np.uint64,
    "MET_FLOAT": np.float32,
    "MET_DOUBLE": np.float64,
}


# Adapted from https://github.com/auneri/MetaImageIO/blob/079914fdb55e0720fdb7db53baa5ab20af2fe556/metaimageio/read.py
def read_meta_image_from_fileobj(f: io.BufferedReader) -> Tuple[np.ndarray, MetaData]:
    # read metadata
    meta_in = {}
    meta_size = 0
    islocal = False
    f.seek(0)
    for line in f:
        line = line.decode()
        meta_size += len(line)
        # skip empty and commented lines
        if not line or line.startswith("#"):
            continue
        key = line.split("=", 1)[0].strip()
        value = line.split("=", 1)[-1].strip()
        # handle case variations
        try:
            key = META_IMAGE_TAGS[[x.upper() for x in META_IMAGE_TAGS].index(key.upper())]
        except ValueError:
            pass
        meta_in[key] = value
        # handle supported ElementDataFile formats
        if key == "ElementDataFile":
            if value.upper() != "LOCAL":
                raise NotImplementedError("ElementDataFile must be 'LOCAL'")
            islocal = True
            break
    if not islocal:
        raise ValueError("Missing ElementDataFile header key")

    # typecast metadata to native types
    meta = dict.fromkeys(META_IMAGE_TAGS, None)
    for key, value in meta_in.items():
        if key in (
            "Comment",
            "ObjectType",
            "ObjectSubType",
            "TransformType",
            "Name",
            "AnatomicalOrientation",
            "Modality",
            "ElementDataFile",
        ):
            meta[key] = value
        elif key in (
            "NDims",
            "ID",
            "ParentID",
            "CompressedDataSize",
            "HeaderSize",
            "HeaderSizePerSlice",
            "ElementNumberOfChannels",
        ):
            meta[key] = np.uintp(value)
        elif key in (
            "CompressedData",
            "BinaryData",
            "BinaryDataByteOrderMSB",
            "ElementByteOrderMSB",
        ):
            meta[key] = value.upper() == "TRUE"
        elif key in (
            "Color",
            "Position",
            "Offset",
            "Origin",
            "CenterOfRotation",
            "ElementSpacing",
            "ElementSize",
        ):
            meta[key] = np.array(value.split(), dtype=float)
        elif key in ("Orientation", "Rotation", "TransformMatrix"):
            meta[key] = np.array(value.split(), dtype=float).reshape(3, 3).transpose()
        elif key in ("DimSize", "SequenceID"):
            meta[key] = np.array(value.split(), dtype=int)
        elif key in ("ElementMin", "ElementMax"):
            meta[key] = float(value)
        elif key == "ElementType":
            try:
                meta[key] = [x[1] for x in META_IMAGE_TYPES.items() if x[0] == value.upper()][0]
            except IndexError as exception:
                raise ValueError(f'ElementType "{value}" is not supported') from exception
        else:
            meta[key] = value

    # read image from file
    shape = np.asarray(meta["DimSize"]).copy()[::-1]
    if (meta.get("ElementNumberOfChannels") or 1) > 1:
        shape = np.r_[shape, meta["ElementNumberOfChannels"]]
    else:
        meta["ElementNumberOfChannels"] = 1
    element_size = np.dtype(meta["ElementType"]).itemsize
    increment = np.prod(shape[1:], dtype=int) * element_size

    f.seek(meta_size, 0)
    f.seek(meta.get("HeaderSize") or 0, 1)
    if meta.get("CompressedData"):
        if meta["CompressedDataSize"] is None:
            raise ValueError("CompressedDataSize needs to be specified when using CompressedData")
        if meta["HeaderSizePerSlice"] is not None:
            raise ValueError("HeaderSizePerSlice is not supported with compressed images")
        buffer = zlib.decompress(f.read(meta["CompressedDataSize"]))
    else:
        data = io.BytesIO()
        read, seek = 0, 0
        for _ in range(shape[0]):
            if meta["HeaderSizePerSlice"] is not None:
                data.write(f.read(read))
                read = 0
                seek += meta["HeaderSizePerSlice"]
            f.seek(int(seek), 1)
            seek = 0
            read += increment
            if read > np.iinfo(int).max - increment:
                data.write(f.read(read))
                read = 0
        data.write(f.read(read))
        buffer = data.getbuffer()

    image = np.frombuffer(buffer, dtype=meta["ElementType"]).reshape(shape)
    if meta.get("BinaryDataByteOrderMSB") or meta.get("ElementByteOrderMSB"):
        image.byteswap(inplace=True)
    image = image.copy()

    # remove unused metadata
    meta["ElementDataFile"] = None
    meta = {x: y for x, y in meta.items() if y is not None}

    return image, meta


# Adapted from https://github.com/auneri/MetaImageIO/blob/57cc445afe0560fa50e245d6875b7efff7a1d880/metaimageio/writer.py
def meta_image_bytes(data: np.ndarray, meta: Optional[MetaData] = None) -> bytes:
    r"""Serialize image to bytes object in MetaImage .mha format.

    Args:
        data: NumPy array of (uncompressed) image data.
        meta: Metadata dictionary.

    Returns:
        blob: Uncompressed or compressed MHA file data.

    """

    data = np.asanyarray(data)

    # handle ElementNumberOfChannels
    if meta and meta["ElementNumberOfChannels"] is not None and meta["ElementNumberOfChannels"] > 1:
        size = np.array(tuple(data.shape)[:-1][::-1])
        ndim = np.ndim(data) - 1
    else:
        size = np.array(tuple(data.shape)[::-1])
        ndim = np.ndim(data)

    # initialize metadata
    meta_out: MetaData = dict.fromkeys(META_IMAGE_TAGS, None)
    meta_out["ObjectType"] = "Image"
    meta_out["NDims"] = ndim
    meta_out["BinaryData"] = True
    meta_out["BinaryDataByteOrderMSB"] = False
    meta_out["ElementSpacing"] = np.ones(ndim)
    meta_out["DimSize"] = size
    meta_out["ElementType"] = data.dtype

    # input metadata (case incensitive)
    if meta:
        for key, value in meta.items():
            try:
                key = META_IMAGE_TAGS[[x.upper() for x in META_IMAGE_TAGS].index(key.upper())]
            except ValueError:
                pass
            meta_out[key] = value
    meta = meta_out

    # define ElementDataFile
    meta["ElementDataFile"] = "LOCAL"

    # prepare image for saving
    if meta.get("BinaryDataByteOrderMSB") or meta.get("ElementByteOrderMSB"):
        data = data.byteswap()
    blob = data.astype(meta["ElementType"]).tobytes()
    if meta.get("CompressedData", True):
        blob = zlib.compress(blob, level=2)
        meta["CompressedDataSize"] = len(blob)

    # typecast metadata to string
    meta_out: MetaData = {}
    for key, value in meta.items():
        if value is None:
            continue
        elif key in (
            "Comment",
            "ObjectType",
            "ObjectSubType",
            "TransformType",
            "Name",
            "AnatomicalOrientation",
            "Modality",
        ):
            meta_out[key] = value
        elif key in (
            "NDims",
            "ID",
            "ParentID",
            "CompressedData",
            "CompressedDataSize",
            "BinaryData",
            "BinaryDataByteOrderMSB",
            "ElementByteOrderMSB",
            "HeaderSize",
            "HeaderSizePerSlice",
            "ElementMin",
            "ElementMax",
            "ElementNumberOfChannels",
        ):
            meta_out[key] = str(value)
        elif key in (
            "Color",
            "Position",
            "Offset",
            "Origin",
            "CenterOfRotation",
            "ElementSpacing",
            "DimSize",
            "SequenceID",
            "ElementSize",
        ):
            meta_out[key] = " ".join(str(x) for x in np.ravel(value))
        elif key in ("Orientation", "Rotation", "TransformMatrix"):
            meta_out[key] = " ".join(str(x) for x in np.ravel(np.transpose(value)))
        elif key == "ElementType":
            try:
                meta_out[key] = [
                    x[0] for x in META_IMAGE_TYPES.items() if np.issubdtype(value, x[1])
                ][0]
            except IndexError as exception:
                raise ValueError(
                    f"meta_image_bytes() ElementType '{value}' is not supported"
                ) from exception
        elif key == "ElementDataFile":
            if not isinstance(value, str) or value.upper() != "LOCAL":
                raise ValueError("meta_image_bytes() ElementDataFile must be 'LOCAL'")
            meta_out[key] = value
        else:
            meta_out[key] = value
    meta = meta_out

    # serialize meta and image data
    stream = io.BytesIO()
    stream.writelines(f"{key} = {value}\n".encode("ascii") for key, value in meta.items())
    stream.write(blob)
    return stream.getvalue()
