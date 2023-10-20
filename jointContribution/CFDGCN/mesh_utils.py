import pickle
from os import PathLike
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import paddle
from scipy.spatial import Delaunay

from ppsci.utils import logger

UnionTensor = Union[paddle.Tensor, np.ndarray]


SU2_SHAPE_IDS = {
    "line": 3,
    "triangle": 5,
    "quad": 9,
}


def get_mesh_graph(
    mesh_filename: Union[str, PathLike], dtype: np.dtype = np.float32
) -> Tuple[np.ndarray, np.ndarray, List[List[List[int]]], Dict[str, List[List[int]]]]:
    def get_rhs(s: str) -> str:
        return s.split("=")[-1]

    marker_dict = {}
    with open(mesh_filename) as f:
        for line in f:
            if line.startswith("NPOIN"):
                num_points = int(get_rhs(line))
                mesh_points = [
                    [float(p) for p in f.readline().split()[:2]]
                    for _ in range(num_points)
                ]
                nodes = np.array(mesh_points, dtype=dtype)

            if line.startswith("NMARK"):
                num_markers = int(get_rhs(line))
                for _ in range(num_markers):
                    line = f.readline()
                    assert line.startswith("MARKER_TAG")
                    marker_tag = get_rhs(line).strip()
                    num_elems = int(get_rhs(f.readline()))
                    marker_elems = [
                        [int(e) for e in f.readline().split()[-2:]]
                        for _ in range(num_elems)
                    ]
                    marker_dict[marker_tag] = marker_elems

            if line.startswith("NELEM"):
                edges = []
                triangles = []
                quads = []
                num_edges = int(get_rhs(line))
                for _ in range(num_edges):
                    elem = [int(p) for p in f.readline().split()]
                    if elem[0] == SU2_SHAPE_IDS["triangle"]:
                        n = 3
                        triangles.append(elem[1 : 1 + n])
                    elif elem[0] == SU2_SHAPE_IDS["quad"]:
                        n = 4
                        quads.append(elem[1 : 1 + n])
                    else:
                        raise NotImplementedError
                    elem = elem[1 : 1 + n]
                    edges += [[elem[i], elem[(i + 1) % n]] for i in range(n)]
                edges = np.array(edges, dtype=np.compat.long).transpose()
                elems = [triangles, quads]

    return nodes, edges, elems, marker_dict


def write_graph_mesh(
    output_filename: Union[str, PathLike],
    points: UnionTensor,
    elems_list: Sequence[Sequence[Sequence[int]]],
    marker_dict: Dict[str, Sequence[Sequence[int]]],
    dims: int = 2,
) -> None:
    def seq2str(s: Sequence[int]) -> str:
        return " ".join(str(x) for x in s)

    with open(output_filename, "w") as f:
        f.write(f"NDIME={dims}\n")

        num_points = points.shape[0]
        f.write(f"NPOIN={num_points}\n")
        for i, p in enumerate(points):
            f.write(f"{seq2str(p.tolist())} {i}\n")
        f.write("\n")

        num_elems = sum([len(elems) for elems in elems_list])
        f.write(f"NELEM={num_elems}\n")
        for elems in elems_list:
            for e in elems:
                if len(e) != 3 and len(e) != 4:
                    raise ValueError(
                        f"Meshes only support triangles and quadrilaterals, "
                        f"passed element had {len(e)} vertices."
                    )
                elem_id = (
                    SU2_SHAPE_IDS["triangle"] if len(e) == 3 else SU2_SHAPE_IDS["quad"]
                )
                f.write(f"{elem_id} {seq2str(e)}\n")
        f.write("\n")

        num_markers = len(marker_dict)
        f.write(f"NMARK={num_markers}\n")
        for marker_tag in marker_dict:
            f.write(f"MARKER_TAG={marker_tag}\n")
            marker_elems = marker_dict[marker_tag]
            f.write(f"MARKER_ELEMS={len(marker_elems)}\n")
            for m in marker_elems:
                f.write(f'{SU2_SHAPE_IDS["line"]} {seq2str(m)}\n')
        f.write("\n")


def generate_mesh(
    mesh_type="regular",
    airfoil_nodes=None,
    farfield_nodes=None,
    num_x=21,
    num_y=21,
    min_x=-20,
    max_x=20,
    min_y=-20,
    max_y=20,
):
    if mesh_type == "regular":
        num_nodes = num_x * num_y
        inds = np.arange(num_nodes).reshape(num_x, num_y)
        x_pos = np.linspace(min_x, max_x, num_x)
        y_pos = np.linspace(max_y, min_y, num_y)
        grid = np.stack(np.meshgrid(x_pos, y_pos))
        nodes = grid.transpose().reshape(num_nodes, 2)
    elif mesh_type == "random":
        num_nodes = num_x * num_y
        x_pos = np.random.uniform(min_x, max_x, num_nodes)
        y_pos = np.random.uniform(min_y, max_y, num_nodes)
        grid = np.stack([x_pos, y_pos], axis=1)
        nodes = grid.transpose().reshape(num_nodes, 2)
    elif mesh_type == "normal":
        num_nodes = num_x * num_y
        # set distance between min and max to be equal to 4 std devs, to have ~95% of points inside
        x_pos = np.random.normal(scale=(max_x - min_x) / 4, size=num_nodes)
        y_pos = np.random.normal(scale=(max_y - min_y) / 4, size=num_nodes)
        grid = np.stack([x_pos, y_pos], axis=1)
        nodes = grid.transpose().reshape(num_nodes, 2)
    else:
        raise NotImplementedError

    if airfoil_nodes is not None:
        # remove nodes that are repeated
        non_repeated_inds = []
        airfoil_list = (
            airfoil_nodes.tolist()
        )  # have to convert to list to check containment
        for i, n in enumerate(nodes):
            if n.tolist() in airfoil_list:
                logger.info(f"Removed node {i}: {n} because its already in airfoil.")
            else:
                non_repeated_inds.append(i)
        nodes = nodes[non_repeated_inds]

        # add airfoil nodes and remove nodes that are inside the airfoil
        nodes_with_airfoil = paddle.to_tensor(
            np.concatenate([nodes, airfoil_nodes], axis=0)
        )
        airfoil_inds = np.arange(nodes.shape[0], nodes_with_airfoil.shape[0])
        airfoil_signed_dists = signed_dist_graph(
            nodes_with_airfoil, airfoil_inds, with_sign=True
        ).numpy()
        is_inside_airfoil = airfoil_signed_dists < 0
        nodes_outside_airfoil = nodes_with_airfoil[~is_inside_airfoil]

        # adjust indices to account for removed nodes
        num_nodes_removed = is_inside_airfoil.sum()
        airfoil_inds = airfoil_inds - num_nodes_removed
        nodes = nodes_outside_airfoil.numpy()

    if farfield_nodes is not None:
        # remove nodes that are repeated
        num_nodes_removed = 0
        non_repeated_inds = []
        farfield_list = (
            farfield_nodes.tolist()
        )  # have to convert to list to check containment
        for i, n in enumerate(nodes):
            if n.tolist() in farfield_list:
                logger.info(f"Removed node {i}: {n} because its already in farfield.")
                num_nodes_removed += 1
            else:
                non_repeated_inds.append(i)
        if airfoil_nodes is not None:
            airfoil_inds -= num_nodes_removed
        nodes = nodes[non_repeated_inds]

        # add airfoil nodes and remove nodes that are inside the airfoil
        nodes_with_farfield = paddle.to_tensor(
            np.concatenate([nodes, farfield_nodes], axis=0)
        )
        farfield_inds = np.arange(nodes.shape[0], nodes_with_farfield.shape[0])
        farfield_signed_dists = signed_dist_graph(
            nodes_with_farfield, farfield_inds, with_sign=True
        ).numpy()
        is_outside_farfield = farfield_signed_dists > 0
        nodes_inside_farfield = nodes_with_farfield[~is_outside_farfield]

        # adjust indices to account for removed nodes
        num_nodes_removed = is_outside_farfield.sum()
        airfoil_inds = airfoil_inds - num_nodes_removed
        farfield_inds = farfield_inds - num_nodes_removed
        nodes = nodes_inside_farfield.numpy()

    elems = delauney(nodes).tolist()
    if airfoil_nodes is not None:
        # keep only elems that are outside airfoil
        elems = [e for e in elems if len([i for i in e if i in airfoil_inds]) < 3]

    marker_dict = {}
    if airfoil_nodes is not None:
        num_airfoil = airfoil_nodes.shape[0]
        marker_dict["airfoil"] = [
            [airfoil_inds[i], airfoil_inds[(i + 1) % num_airfoil]]
            for i in range(num_airfoil)
        ]

    if farfield_nodes is not None:
        num_farfield = farfield_nodes.shape[0]
        marker_dict["farfield"] = [
            [farfield_inds[i], farfield_inds[(i + 1) % num_farfield]]
            for i in range(num_farfield)
        ]
    else:
        marker_dict["farfield"] = []
        marker_dict["farfield"] += [
            [inds[0, j], inds[0, j + 1]] for j in range(num_x - 1)
        ]
        marker_dict["farfield"] += [
            [inds[-1, j], inds[-1, j + 1]] for j in range(num_x - 1)
        ]
        marker_dict["farfield"] += [
            [inds[i, 0], inds[i + 1, 0]] for i in range(num_y - 1)
        ]
        marker_dict["farfield"] += [
            [inds[i, -1], inds[i + 1, -1]] for i in range(num_y - 1)
        ]

    # write_graph_mesh(output_filename, nodes, [elems], marker_dict)
    return nodes, elems, marker_dict


def delauney(x):
    """Adapted from torch_geometric.transforms.delaunay.Delaunay."""
    pos = x[:, :2]
    if pos.shape[0] > 3:
        tri = Delaunay(pos, qhull_options="QJ")
        face = tri.simplices
    elif pos.size(0) == 3:
        face = np.array([[0, 1, 2]])
    else:
        raise ValueError(
            "Not enough points to contruct Delaunay triangulation, got {} "
            "but expected at least 3".format(pos.size(0))
        )

    elems = face.astype(np.compat.long)
    return elems


def get_dists(edge_index, pos, norm=True, max=None):
    """Adapted from torch_geometric.transforms.Distance"""
    (row, col), pos = edge_index, pos
    dist = paddle.norm(pos[col] - pos[row], p=2, axis=-1).view(-1, 1)
    if norm and dist.numel() > 0:
        dist = dist / dist.max() if max is None else max
    return dist


def is_ccw(points, ret_val=False):
    """From: https://stackoverflow.com/questions/1165647#1180256"""
    n = points.shape[0]
    a = paddle.argmin(points[:, 1])
    b = (a - 1) % n
    c = (a + 1) % n

    ab = points[a] - points[b]
    ac = points[a] - points[c]
    cross = ab[0] * ac[1] - ab[1] * ac[0]

    if not ret_val:
        return cross <= 0
    else:
        return cross


def is_cw(points, triangles, ret_val=False):
    tri_pts = points[triangles]
    a = tri_pts[:, 0] - tri_pts[:, 1]
    b = tri_pts[:, 1] - tri_pts[:, 2]
    cross = b[:, 0] * a[:, 1] - b[:, 1] * a[:, 0]

    if not ret_val:
        return cross > 0
    else:
        return cross


def quad2tri(elems):
    new_elems = []
    new_edges = []
    for e in elems:
        if len(e) <= 3:
            new_elems.append(e)
        else:
            new_elems.append([e[0], e[1], e[2]])
            new_elems.append([e[0], e[2], e[3]])
            new_edges.append(paddle.to_tensor(([[e[0]], [e[2]]]), dtype=paddle.int64))
    new_edges = (
        paddle.concat(new_edges, axis=1)
        if new_edges
        else paddle.to_tensor([], dtype=paddle.int64)
    )
    return new_elems, new_edges


def left_orthogonal(v):
    return paddle.stack([-v[..., 1], v[..., 0]], axis=-1)


def signed_dist_graph(nodes, marker_inds, with_sign=False):
    # assumes shape is convex
    # approximate signed distance by distance to closest point on surface
    signed_dists = paddle.zeros([nodes.shape[0]], dtype=paddle.float32)
    marker_nodes = nodes[marker_inds]
    if type(marker_inds) is paddle.Tensor:
        marker_inds = marker_inds.tolist()
    marker_inds = set(marker_inds)

    if with_sign:
        marker_surfaces = marker_nodes[:-1] - marker_nodes[1:]
        last_surface = marker_nodes[-1] - marker_nodes[0]
        marker_surfaces = paddle.concat([marker_surfaces, last_surface.unsqueeze(0)])
        normals = left_orthogonal(marker_surfaces) / marker_surfaces.norm(
            dim=1
        ).unsqueeze(1)

    for i, x in enumerate(nodes):
        if i not in marker_inds:
            vecs = marker_nodes - x
            dists = paddle.linalg.norm(vecs, axis=1)
            min_dist = dists.min()

            if with_sign:
                # if sign is requested, check if inside marker shape
                # dot product with normals to find if inside shape
                surface_dists = (vecs * normals).sum(dim=1)
                if (surface_dists < 0).unique().shape[0] == 1:
                    # if all point in same direction it is inside
                    min_dist *= -1

            signed_dists[i] = min_dist
    return signed_dists


def plot_field(
    nodes,
    elems_list,
    field,
    contour=False,
    clim=None,
    zoom=True,
    get_array=True,
    out_file=None,
    show=False,
    title="",
):
    elems_list = sum(elems_list, [])
    tris, _ = quad2tri(elems_list)
    tris = np.array(tris)
    x, y = nodes[:, :2].t().detach().cpu().numpy()
    field = field.detach().cpu().numpy()
    fig = plt.figure()
    if contour:
        plt.tricontourf(x, y, tris, field)
    else:
        plt.tripcolor(x, y, tris, field)
    if clim:
        plt.clim(*clim)
    plt.colorbar()
    if zoom:
        plt.xlim(left=-0.5, right=1.5)
        plt.ylim(bottom=-1, top=1)
    if title:
        plt.title(title)

    if out_file is not None:
        plt.savefig(out_file)
        plt.close()

    if show:
        # plt.show()
        raise NotImplementedError

    if get_array:
        fig.canvas.draw()
        a = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        a = a.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return a


def write_tecplot(graph, fields, elems_list, filename="flow.dat"):
    x = graph.x
    edge_index = graph.edge_index
    num_nodes = x.shape[0]
    num_edges = edge_index.shape[1]
    with open(filename, "w") as f:
        f.write('TITLE = "Visualization of the volumetric solution"\n')
        f.write(
            'VARIABLES = "x","y","Density","Momentum_x","Momentum_y",'
            '"Energy","Pressure","Temperature","Mach","C<sub>p</sub>"\n'
        )
        f.write(
            f"ZONE NODES = {num_nodes}, ELEMENTS = {num_edges}, "
            f"DATAPACKING = POINT, ZONETYPE = FEQUADRILATERAL\n"
        )
        for node, field in zip(x, fields):
            f.write(
                f"{node[0].item()}\t{node[1].item()}\t0.0\t"
                f"{field[0].item()}\t{field[1].item()}\t0.0\t"
                f"{field[2].item()}\t0.0\t0.0\t0.0\n"
            )
        elems_list = sum(elems_list, [])
        for elem in elems_list:
            f.write("\t".join(str(x + 1) for x in elem))
            if len(elem) == 3:
                # repeat last vertex if triangle
                f.write(f"\t{elem[-1]+1}")
            f.write("\n")


if __name__ == "__main__":
    import time

    mesh = "mesh_NACA0012_fine.su2"
    start = time.time()
    x, edge_index, _, marker_dict = get_mesh_graph(f"meshes/{mesh}")

    x = paddle.to_tensor(x, dtype=paddle.float32)
    edge_index = paddle.to_tensor(edge_index)

    triangulation = Delaunay(x)
    airfoil_markers = set(marker_dict["airfoil"][0])
    elems = triangulation.simplices
    keep_inds = [
        i
        for i in range(elems.shape[1])
        if not (
            elems[0, i].item() in airfoil_markers
            and elems[1, i].item() in airfoil_markers
            and elems[2, i].item() in airfoil_markers
        )
    ]
    elems = elems[:, keep_inds]

    write_graph_mesh("test_mesh.su2", x, [elems], marker_dict)
    logger.info(f"Took: {time.time() - start}")

    with open(f"meshes/graph_{mesh}.pkl", "wb") as f:
        pickle.dump([x, edge_index], f)


def visualize_mesh(
    nodes, elements, xlims=None, ylims=None, marker=".", plot_inds=False
):
    """Modified from: https://stackoverflow.com/questions/52202014"""

    x = nodes[:, 0]
    y = nodes[:, 1]

    # https://stackoverflow.com/questions/49640311/
    def plot_elems(x, y, elems, ax=None, **kwargs):
        if not ax:
            ax = plt.gca()
        xy = np.c_[x, y]
        verts = xy[elems]
        pc = matplotlib.collections.PolyCollection(verts, **kwargs)
        ax.add_collection(pc)
        ax.autoscale()

    plt.figure()
    plt.gca().set_aspect("equal")

    plot_elems(x, y, np.asarray(elements), ax=None, color="crimson", facecolor="None")
    plt.plot(x, y, marker=marker, ls="", color="crimson")

    if plot_inds:
        for i, pos in enumerate(nodes):
            plt.annotate(i, (pos[0], pos[1]))

    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")

    if xlims:
        plt.xlim(left=xlims[0], right=xlims[1])
    if ylims:
        plt.ylim(top=ylims[1], bottom=ylims[0])

    plt.show()
