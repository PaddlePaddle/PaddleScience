# cython: language_level=3
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: profile=False
# distutils: language = c


import numpy as np
cimport numpy as np

import chgnet.graph.graph

from libc.stdlib cimport free


cdef extern from 'fast_converter_libraries/create_graph.c':
    ctypedef struct Node:
        np.int64_t index
        LongToDirectedEdgeList* neighbors
        np.int64_t num_neighbors

    ctypedef struct NodeIndexPair:
        np.int64_t center
        np.int64_t neighbor

    ctypedef struct UndirectedEdge:
        NodeIndexPair nodes
        np.int64_t index
        np.int64_t* directed_edge_indices
        np.int64_t num_directed_edges
        np.float64_t distance

    ctypedef struct DirectedEdge:
        NodeIndexPair nodes
        np.int64_t index
        const np.int64_t* image
        np.int64_t undirected_edge_index
        np.float64_t distance

    ctypedef struct LongToDirectedEdgeList:
        np.int64_t key
        DirectedEdge** directed_edges_list
        int num_directed_edges_in_group

    ctypedef struct ReturnElems2:
        np.int64_t num_nodes
        np.int64_t num_directed_edges
        np.int64_t num_undirected_edges
        Node* nodes
        UndirectedEdge** undirected_edges_list
        DirectedEdge** directed_edges_list

    ReturnElems2* create_graph(
        np.int64_t* center_index,
        np.int64_t n_e,
        np.int64_t* neighbor_index,
        np.int64_t* image,
        np.float64_t* distance,
        np.int64_t num_atoms)

    void free_LongToDirectedEdgeList_in_nodes(Node* nodes, np.int64_t num_nodes)


    LongToDirectedEdgeList** get_neighbors(Node* node)

def make_graph(
        const np.int64_t[::1] center_index,
        const np.int64_t n_e,
        const np.int64_t[::1] neighbor_index,
        const np.int64_t[:, ::1] image,
        const np.float64_t[::1] distance,
        const np.int64_t num_atoms
    ):
    cdef ReturnElems2* returned
    returned = <ReturnElems2*> create_graph(<np.int64_t*> &center_index[0], n_e, <np.int64_t*> &neighbor_index[0], <np.int64_t*> &image[0][0], <np.float64_t*> &distance[0], num_atoms)

    chg_DirectedEdge = chgnet.graph.graph.DirectedEdge
    chg_Node = chgnet.graph.graph.Node
    chg_UndirectedEdge = chgnet.graph.graph.UndirectedEdge


    image_np = np.asarray(image)

    cdef LongToDirectedEdgeList** node_neighbors
    cdef Node this_node
    cdef LongToDirectedEdgeList this_entry
    py_nodes = []
    cdef DirectedEdge* this_DE


    # Handling nodes + directed edges
    for idx in range(returned[0].num_nodes):
        this_node = returned[0].nodes[idx]
        this_py_node = chg_Node(index=idx)

        node_neighbors = get_neighbors(&this_node)

        # Iterate through all neighbors and populate our py_node.neighbors dict
        for j in range(this_node.num_neighbors):
            this_entry = node_neighbors[j][0]
            directed_edges = []

            for k in range(this_entry.num_directed_edges_in_group):
                this_DE = this_entry.directed_edges_list[k]
                directed_edges.append(this_DE[0].index)

            this_py_node.neighbors[this_entry.key] = directed_edges

        py_nodes.append(this_py_node)

        free(node_neighbors)

    # Handling directed edges
    py_directed_edges_list = []

    for idx in range(returned[0].num_directed_edges):
        this_DE = returned[0].directed_edges_list[idx]
        py_DE = chg_DirectedEdge(nodes = [this_DE[0].nodes.center, this_DE[0].nodes.neighbor], index=this_DE[0].index, info = {"distance": this_DE[0].distance, "image": image_np[this_DE[0].index], "undirected_edge_index": this_DE[0].undirected_edge_index})

        py_directed_edges_list.append(py_DE)


    # Handling undirected edges
    py_undirected_edges_list = []
    cdef UndirectedEdge* UDE

    for idx in range(returned[0].num_undirected_edges):
        UDE = returned[0].undirected_edges_list[idx]
        py_undirected_edge = chg_UndirectedEdge([UDE[0].nodes.center, UDE[0].nodes.neighbor], index = UDE[0].index, info =  {"distance": UDE[0].distance, "directed_edge_index": []})

        for j in range(UDE[0].num_directed_edges):
            py_undirected_edge.info["directed_edge_index"].append(UDE[0].directed_edge_indices[j])

        py_undirected_edges_list.append(py_undirected_edge)


    # Create Undirected_Edges hashmap
    py_undirected_edges = {}
    for undirected_edge in py_undirected_edges_list:
        this_set = frozenset(undirected_edge.nodes)
        if this_set not in py_undirected_edges:
            py_undirected_edges[this_set] = [undirected_edge]
        else:
            py_undirected_edges[this_set].append(undirected_edge)

    # # Update the nodes list to have pointers to DirectedEdges instead of indices
    for node_index in range(returned[0].num_nodes):
        this_neighbors = py_nodes[node_index].neighbors
        for this_neighbor_index in this_neighbors:
            replacement = [py_directed_edges_list[edge_index] for edge_index in this_neighbors[this_neighbor_index]]
            this_neighbors[this_neighbor_index] = replacement


    # Free everything unneeded
    for idx in range(returned[0].num_directed_edges):
        free(returned[0].directed_edges_list[idx])

    for idx in range(returned[0].num_undirected_edges):
        free(returned[0].undirected_edges_list[idx].directed_edge_indices)
        free(returned[0].undirected_edges_list[idx])


    # Free node LongToDirectedEdgeList
    free_LongToDirectedEdgeList_in_nodes(returned[0].nodes, returned[0].num_nodes)

    free(returned[0].directed_edges_list)
    free(returned[0].undirected_edges_list)
    free(returned[0].nodes)

    free(returned)

    return py_nodes, py_directed_edges_list, py_undirected_edges_list, py_undirected_edges
