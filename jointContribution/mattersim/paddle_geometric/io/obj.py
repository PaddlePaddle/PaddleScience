from typing import Iterator, List, Optional, Tuple, Union

import paddle
from paddle_geometric.data import Data


def yield_file(in_file: str) -> Iterator[Tuple[str, List[Union[int, float]]]]:
    with open(in_file, 'r') as f:
        buf = f.read()
    for b in buf.split('\n'):
        if b.startswith('v '):
            yield 'v', [float(x) for x in b.split(" ")[1:]]
        elif b.startswith('f '):
            triangles = b.split(' ')[1:]
            # -1 as .obj is base 1 but the Data class expects base 0 indices
            yield 'f', [int(t.split("/")[0]) - 1 for t in triangles]
        else:
            yield '', []


def read_obj(in_file: str) -> Optional[Data]:
    vertices = []
    faces = []

    for k, v in yield_file(in_file):
        if k == 'v':
            vertices.append(v)
        elif k == 'f':
            faces.append(v)

    if not faces or not vertices:
        return None

    pos = paddle.to_tensor(vertices, dtype='float32')
    face = paddle.to_tensor(faces, dtype='int64').transpose([1, 0])

    data = Data(pos=pos, face=face)

    return data
