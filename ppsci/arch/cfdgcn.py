import os

import paddle
# import su2paddle
from typing import Dict
from typing import Tuple

from paddle import nn
from ppsci.data.dataset import airfoil_dataset

try:
    import pgl
except ModuleNotFoundError:
    pass

def is_cw(points, triangles, ret_val=False):
    tri_pts = points[triangles]
    a = tri_pts[:, 0] - tri_pts[:, 1]
    b = tri_pts[:, 1] - tri_pts[:, 2]
    cross = b[:, 0] * a[:, 1] - b[:, 1] * a[:, 0]

    if not ret_val:
        return cross > 0
    else:
        return cross

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
        normals = left_orthogonal(marker_surfaces) / marker_surfaces.norm(dim=1).unsqueeze(1)

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
    new_edges = paddle.concat(new_edges, axis=1) if new_edges else paddle.to_tensor([], dtype=paddle.int64)
    return new_elems, new_edges

class MeshGCN(nn.Layer):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=6, fine_marker_dict=None):
        super().__init__()
        self.fine_marker_dict = paddle.unique(paddle.to_tensor(fine_marker_dict['airfoil']))
        self.sdf = None
        in_channels += 1  # account for sdf

        channels = [in_channels]
        channels += [hidden_channels] * (num_layers - 1)
        channels += [out_channels]

        self.convs = nn.LayerList()
        for i in range(num_layers):
            self.convs.append(GCNConv(channels[i], channels[i + 1]))

        # self.convs.append(GATConv(channels[0], channels[1], num_heads=4))
        # for i in range(1, num_layers - 1):
        #     self.convs.append(GATConv(channels[i] * 4, channels[i + 1], num_heads=4))
        # self.convs.append(GATConv(channels[num_layers - 1]*4, channels[num_layers], num_heads=1))

    def forward(self, graphs):
        pred_fields = []
        for graph in graphs:
            x = graph.node_feat["feature"]

            if self.sdf is None:
                with paddle.no_grad():
                    self.sdf = signed_dist_graph(x[:, :2], self.fine_marker_dict).unsqueeze(1)
            x = paddle.concat([x, self.sdf], axis=-1)

            for i, conv in enumerate(self.convs[:-1]):
                x = conv(graph, x)
                x = F.relu(x)

            pred_field = self.convs[-1](graph, x)
            pred_fields.append(pred_field)
        return pred_fields
        
class CFDGCN(nn.Layer):
    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        coarse_mesh,
        fine_marker_dict,
        process_sim=lambda x, y: x,
        freeze_mesh=False,
        num_convs=6,
        num_end_convs=3,
        hidden_channels=512,
        out_channels=3,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        meshes_temp_dir = "temp_meshes"
        os.makedirs(meshes_temp_dir, exist_ok=True)
        self.mesh_file = meshes_temp_dir + "/" + str(os.getpid()) + "_mesh.su2"

        if not coarse_mesh:
            raise ValueError("Need to provide a coarse mesh for CFD-GCN.")
        nodes, edges, self.elems, self.marker_dict = airfoil_dataset._get_mesh_graph(coarse_mesh)
        if not freeze_mesh:
            self.nodes = paddle.to_tensor(nodes, stop_gradient=False)
        else:
            self.nodes = paddle.to_tensor(nodes, stop_gradient=True)

        self.elems, new_edges = quad2tri(sum(self.elems, []))
        self.elems = [self.elems]
        self.edges = paddle.to_tensor(edges)
        # print(self.edges.dtype, new_edges.dtype)
        self.edges = paddle.concat([self.edges, new_edges], axis=1)
        self.marker_inds = paddle.to_tensor(sum(self.marker_dict.values(), [])).unique()
        assert (
            is_cw(self.nodes, paddle.to_tensor(self.elems[0])).nonzero().shape[0] == 0
        ), "Mesh has flipped elems"

        self.process_sim = process_sim
        # self.su2 = su2paddle.SU2Module(config_file, mesh_file=self.mesh_file)
        # print(f'Mesh filename: {self.mesh_file.format(batch_index="*")}', flush=True)
        self.fine_marker_dict = paddle.to_tensor(fine_marker_dict["airfoil"]).unique()
        self.sdf = None

        self.num_convs = num_end_convs
        self.convs = []
        if self.num_convs > 0:
            self.convs = nn.LayerList()
            in_channels = out_channels + hidden_channels
            for i in range(self.num_convs - 1):
                self.convs.append(pgl.nn.GCNConv(in_channels, hidden_channels))
                in_channels = hidden_channels
            self.convs.append(pgl.nn.GCNConv(in_channels, out_channels))

        self.num_pre_convs = num_convs - num_end_convs
        self.pre_convs = []
        if self.num_pre_convs > 0:
            in_channels = 5 + 1  # one extra channel for sdf
            self.pre_convs = nn.LayerList()
            for i in range(self.num_pre_convs - 1):
                self.pre_convs.append(pgl.nn.GCNConv(in_channels, hidden_channels))
                in_channels = hidden_channels
            self.pre_convs.append(pgl.nn.GCNConv(in_channels, hidden_channels))

        self.sim_info = {}  # store output of coarse simulation for logging / debugging

    def forward(self, x: Dict[str, "pgl.Graph"]) -> Dict[str, paddle.Tensor]:
        graphs = x[self.input_keys[0]]
        batch_size = len(graphs)
        nodes_list = []
        aoa_list = []
        mach_or_reynolds_list = []
        fine_x_list = []
        for graph in graphs:
            x = graph.node_feat["feature"]

            if self.sdf is None:
                with paddle.no_grad():
                    self.sdf = signed_dist_graph(
                        x[:, :2], self.fine_marker_dict
                    ).unsqueeze(1)
            fine_x = paddle.concat([x, self.sdf], axis=1)

            for i, conv in enumerate(self.pre_convs):
                fine_x = nn.functional.relu(conv(graph, fine_x))
            fine_x_list.append(fine_x)

            nodes = self.get_nodes()  # [353,2]
            self.write_mesh_file(
                nodes, self.elems, self.marker_dict, filename=self.mesh_file
            )

            nodes_list.append(nodes)
            aoa_list.append(graph.aoa)
            mach_or_reynolds_list.append(graph.mach_or_reynolds)

        # paddle stack for [batch,nodes],[batch,nodes],[batch,1],[batch,1] for su2
        # su2 can apply each item of one batch with mpi
        nodes_input = paddle.stack(nodes_list, axis=0)
        aoa_input = paddle.stack(aoa_list, axis=0)
        mach_or_reynolds_input = paddle.stack(mach_or_reynolds_list, axis=0)

        # batch_y = self.su2(
        #     nodes_input[..., 0],
        #     nodes_input[..., 1],
        #     aoa_input[..., None],
        #     mach_or_reynolds_input[..., None],
        # )
        batch_y = self.process_sim(
            batch_y, False
        )  # [8,353] * 3, a list with three items

        pred_fields = []
        for idx in range(batch_size):
            graph = graphs[idx]
            coarse_y = paddle.stack([y[idx].flatten() for y in batch_y], axis=1).astype(
                "float32"
            )  # features [353,3]
            nodes = self.get_nodes()  # [353,2]
            x = graph.node_feat[
                "feature"
            ]  # [6684,5] the two-first columns are the node locations
            fine_y = self.upsample(
                features=coarse_y, coarse_nodes=nodes[:, :2], fine_nodes=x[:, :2]
            )
            fine_y = paddle.concat([fine_y, fine_x_list[idx]], axis=1)

            for i, conv in enumerate(self.convs[:-1]):
                fine_y = nn.functional.relu(conv(graph, fine_y))
            fine_y = self.convs[-1](graph, fine_y)
            pred_fields.append(fine_y)

        # self.sim_info['nodes'] = nodes[:, :2]
        # self.sim_info['elems'] = [self.elems] * batch_size
        # self.sim_info['batch'] = graph
        # self.sim_info['output'] = coarse_y

        return {self.output_keys[0]: pred_field}

    def upsample(self, features, coarse_nodes, fine_nodes):
        """

        :param features: [353，3]
        :param coarse_nodes: [353, 2]
        :param fine_nodes: [6684, 2]
        :return:
        """
        coarse_nodes_input = paddle.repeat_interleave(
            coarse_nodes.unsqueeze(0), fine_nodes.shape[0], 0
        )  # [6684,352,2]
        fine_nodes_input = paddle.repeat_interleave(
            fine_nodes.unsqueeze(1), coarse_nodes.shape[0], 1
        )  # [6684,352,2]

        dist_w = 1.0 / (
            paddle.norm(x=coarse_nodes_input - fine_nodes_input, p=2, axis=-1) + 1e-9
        )  # [6684,352]
        knn_value, knn_index = paddle.topk(
            dist_w, k=3, largest=True
        )  # [6684,3],[6684,3]

        weight = knn_value.unsqueeze(-2)
        features_input = features[knn_index]

        output = paddle.bmm(weight, features_input).squeeze(-2) / paddle.sum(
            knn_value, axis=-1, keepdim=True
        )

        # y = knn_interpolate(y.cpu(), coarse_nodes[:, :2].cpu(), fine_nodes.cpu(),
        #                     coarse_batch.cpu(), fine.batch.cpu(), k=3)
        return output

    def get_nodes(self):
        # return torch.cat([self.marker_nodes, self.not_marker_nodes])
        return self.nodes

    @staticmethod
    def write_mesh_file(x, elems, marker_dict, filename="mesh.su2"):
        write_graph_mesh(filename, x[:, :2], elems, marker_dict)

    @staticmethod
    def contiguous_elems_list(elems, inds):
        # Hack to easily have compatibility with MeshEdgePool
        return elems
