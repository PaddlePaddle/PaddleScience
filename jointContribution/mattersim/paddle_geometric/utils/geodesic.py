import multiprocessing as mp
import warnings
from typing import Optional

import numpy as np
import paddle
from paddle import Tensor


def geodesic_distance(
    pos: Tensor,
    face: Tensor,
    src: Optional[Tensor] = None,
    dst: Optional[Tensor] = None,
    norm: bool = True,
    max_distance: Optional[float] = None,
    num_workers: int = 0,
    **kwargs: Optional[Tensor],
) -> Tensor:
    import gdist

    if 'dest' in kwargs:
        dst = kwargs['dest']
        warnings.warn("'dest' attribute in 'geodesic_distance' is deprecated "
                      "and will be removed in a future release. Use the 'dst' "
                      "argument instead.")

    max_distance = float('inf') if max_distance is None else max_distance

    if norm:
        area = paddle.cross(pos[face[1]] - pos[face[0]], pos[face[2]] - pos[face[0]], axis=1)
        scale = float((paddle.norm(area, p=2, axis=1) / 2).sum().sqrt())
    else:
        scale = 1.0

    dtype = pos.dtype

    pos_np = pos.astype('float64').numpy()
    face_np = face.t().astype('int32').numpy()

    if src is None and dst is None:
        out = gdist.local_gdist_matrix(pos_np, face_np, max_distance * scale).toarray() / scale
        return paddle.to_tensor(out, dtype=dtype)

    if src is None:
        src_np = np.arange(pos.shape[0], dtype=np.int32)
    else:
        src_np = src.astype('int32').numpy()

    dst_np = None if dst is None else dst.astype('int32').numpy()

    def _parallel_loop(
        pos_np: np.ndarray,
        face_np: np.ndarray,
        src_np: np.ndarray,
        dst_np: Optional[np.ndarray],
        max_distance: float,
        scale: float,
        i: int,
        dtype: paddle.dtype,
    ) -> Tensor:
        s = src_np[i:i + 1]
        d = None if dst_np is None else dst_np[i:i + 1]
        out = gdist.compute_gdist(pos_np, face_np, s, d, max_distance * scale)
        out = out / scale
        return paddle.to_tensor(out, dtype=dtype)

    num_workers = mp.cpu_count() if num_workers <= -1 else num_workers
    if num_workers > 0:
        with mp.Pool(num_workers) as pool:
            data = [(pos_np, face_np, src_np, dst_np, max_distance, scale, i, dtype) for i in range(len(src_np))]
            outs = pool.starmap(_parallel_loop, data)
    else:
        outs = [_parallel_loop(pos_np, face_np, src_np, dst_np, max_distance, scale, i, dtype) for i in range(len(src_np))]

    out = paddle.concat(outs, axis=0)

    if dst is None:
        out = out.reshape([-1, pos.shape[0]])

    return out
