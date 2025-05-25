import paddle

from paddle_geometric.data import Data

try:
    import openmesh
except ImportError:
    openmesh = None


def read_ply(path: str) -> Data:
    if openmesh is None:
        raise ImportError('`read_ply` requires the `openmesh` package.')

    mesh = openmesh.read_trimesh(path)
    pos = paddle.to_tensor(mesh.points(), dtype='float32')
    face = paddle.to_tensor(mesh.face_vertex_indices(), dtype='int64').transpose([1, 0]).contiguous()
    return Data(pos=pos, face=face)
