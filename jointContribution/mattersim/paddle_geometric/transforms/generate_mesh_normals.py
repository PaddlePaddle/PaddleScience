import paddle.nn.functional as F
import paddle
from paddle_geometric.data import Data
from paddle_geometric.data.datapipes import functional_transform
from paddle_geometric.transforms import BaseTransform
from paddle_geometric.utils import scatter


@functional_transform('generate_mesh_normals')
class GenerateMeshNormals(BaseTransform):
    r"""Generate normal vectors for each mesh node based on neighboring
    faces (functional name: :obj:`generate_mesh_normals`).
    """
    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        assert data.face is not None
        pos, face = data.pos, data.face

        vec1 = pos[face[1]] - pos[face[0]]
        vec2 = pos[face[2]] - pos[face[0]]
        face_norm = F.normalize(paddle.cross(vec1, vec2, axis=1), p=2, axis=-1)  # [F, 3]

        face_norm = paddle.repeat_interleave(face_norm, repeats=3, axis=0)
        idx = face.flatten()

        norm = scatter(face_norm, idx, dim=0, reduce='sum', dim_size=pos.shape[0])
        norm = F.normalize(norm, p=2, axis=-1)  # [N, 3]

        data.norm = norm

        return data
