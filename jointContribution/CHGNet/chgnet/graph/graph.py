from __future__ import annotations

import sys
from abc import ABC
from abc import abstractmethod

from chgnet.utils import write_json


class Node:
    """A node in a graph."""

    def __init__(self, index: int, info: (dict | None) = None) -> None:
        """Initialize a Node.

        Args:
            index (int): the index of this node
            info (dict, optional): any additional information about this node.
        """
        self.index = index
        self.info = info
        self.neighbors: dict[int, list[DirectedEdge | UndirectedEdge]] = {}

    def add_neighbor(self, index, edge) -> None:
        """Draw an directed edge between self and the node specified by index.

        Args:
            index (int): the index of neighboring node
            edge (DirectedEdge): an DirectedEdge object pointing from self to the node.
        """
        if index not in self.neighbors:
            self.neighbors[index] = [edge]
        else:
            self.neighbors[index].append(edge)


class Edge(ABC):
    """Abstract base class for edges in a graph."""

    def __init__(
        self, nodes: list, index: (int | None) = None, info: (dict | None) = None
    ) -> None:
        """Initialize an Edge."""
        self.nodes = nodes
        self.index = index
        self.info = info

    def __repr__(self) -> str:
        """String representation of this edge."""
        nodes, index, info = self.nodes, self.index, self.info
        return f"{type(self).__name__}(nodes={nodes!r}, index={index!r}, info={info!r})"

    def __hash__(self) -> int:
        """Hash this edge."""
        img = (self.info or {}).get("image")
        img_str = "" if img is None else img.tobytes()
        return hash((self.nodes[0], self.nodes[1], img_str))

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Check if two edges are equal."""
        raise NotImplementedError


class UndirectedEdge(Edge):
    """An undirected/bi-directed edge in a graph."""

    __hash__ = Edge.__hash__

    def __eq__(self, other: object) -> bool:
        """Check if two undirected edges are equal."""
        return set(self.nodes) == set(other.nodes) and self.info == other.info


class DirectedEdge(Edge):
    """A directed edge in a graph."""

    __hash__ = Edge.__hash__

    def make_undirected(self, index: int, info: (dict | None) = None) -> UndirectedEdge:
        """Make a directed edge undirected."""
        info = info or {}
        info["distance"] = self.info["distance"]
        return UndirectedEdge(self.nodes, index, info)

    def __eq__(self, other: object) -> bool:
        """Check if the two directed edges are equal.

        Args:
            other (DirectedEdge): another DirectedEdge to compare to

        Returns:
            bool: True if other is the same directed edge, or if other is the directed
                edge with reverse direction of self, else False.
        """
        if not isinstance(other, DirectedEdge):
            return False
        self_img = (self.info or {}).get("image")
        other_img = (other.info or {}).get("image")
        none_img = self_img is other_img is None
        if self.nodes == other.nodes and (none_img or all(self_img == other_img)):
            print(
                "!!!!!! the two directed edges are equal but this operation is not supposed to happen",
                file=sys.stderr,
            )
            return True
        return self.nodes == other.nodes[::-1] and (
            none_img or all(self_img == -1 * other_img)
        )


class Graph:
    """A graph for storing the neighbor information of atoms."""

    def __init__(self, nodes: list[Node]) -> None:
        """Initialize a Graph from a list of nodes."""
        self.nodes = nodes
        self.directed_edges: dict[frozenset[int], list[DirectedEdge]] = {}
        self.directed_edges_list: list[DirectedEdge] = []
        self.undirected_edges: dict[frozenset[int], list[UndirectedEdge]] = {}
        self.undirected_edges_list: list[UndirectedEdge] = []

    def add_edge(
        self, center_index, neighbor_index, image, distance, dist_tol: float = 1e-06
    ) -> None:
        """Add an directed edge to the graph.

        Args:
            center_index (int): center node index
            neighbor_index (int): neighbor node index
            image (np.array): the periodic cell image the neighbor is from
            distance (float): distance between center and neighbor.
            dist_tol (float): tolerance for distance comparison between edges.
                Default = 1e-6
        """
        directed_edge_index = len(self.directed_edges_list)
        this_directed_edge = DirectedEdge(
            [center_index, neighbor_index],
            index=directed_edge_index,
            info={"image": image, "distance": distance},
        )
        tmp = frozenset([center_index, neighbor_index])
        if tmp not in self.undirected_edges:
            this_directed_edge.info["undirected_edge_index"] = len(
                self.undirected_edges_list
            )
            this_undirected_edge = this_directed_edge.make_undirected(
                index=len(self.undirected_edges_list),
                info={"directed_edge_index": [directed_edge_index]},
            )
            self.undirected_edges[tmp] = [this_undirected_edge]
            self.undirected_edges_list.append(this_undirected_edge)
            self.nodes[center_index].add_neighbor(neighbor_index, this_directed_edge)
            self.directed_edges_list.append(this_directed_edge)
        else:
            for undirected_edge in self.undirected_edges[tmp]:
                if (
                    abs(undirected_edge.info["distance"] - distance) < dist_tol
                    and len(undirected_edge.info["directed_edge_index"]) == 1
                ):
                    added_dir_edge = self.directed_edges_list[
                        undirected_edge.info["directed_edge_index"][0]
                    ]
                    if added_dir_edge == this_directed_edge:
                        this_directed_edge.info[
                            "undirected_edge_index"
                        ] = added_dir_edge.info["undirected_edge_index"]
                        self.nodes[center_index].add_neighbor(
                            neighbor_index, this_directed_edge
                        )
                        self.directed_edges_list.append(this_directed_edge)
                        undirected_edge.info["directed_edge_index"].append(
                            directed_edge_index
                        )
                        return
            this_directed_edge.info["undirected_edge_index"] = len(
                self.undirected_edges_list
            )
            this_undirected_edge = this_directed_edge.make_undirected(
                index=len(self.undirected_edges_list),
                info={"directed_edge_index": [directed_edge_index]},
            )
            self.undirected_edges[tmp].append(this_undirected_edge)
            self.undirected_edges_list.append(this_undirected_edge)
            self.nodes[center_index].add_neighbor(neighbor_index, this_directed_edge)
            self.directed_edges_list.append(this_directed_edge)

    def adjacency_list(self) -> tuple[list[list[int]], list[int]]:
        """Get the adjacency list
        Return:
            graph: the adjacency list
                [[0, 1],
                 [0, 2],
                 ...
                 [5, 2]
                 ...  ]]
                the fist column specifies center/source node,
                the second column specifies neighbor/destination node
            directed2undirected:
                [0, 1, ...]
                a list of length = num_directed_edge that specifies
                the undirected edge index corresponding to the directed edges
                represented in each row in the graph adjacency list.
        """
        graph = [edge.nodes for edge in self.directed_edges_list]
        directed2undirected = [
            edge.info["undirected_edge_index"] for edge in self.directed_edges_list
        ]
        return graph, directed2undirected

    def line_graph_adjacency_list(self, cutoff) -> tuple[list[list[int]], list[int]]:
        """Get the line graph adjacency list.

        Args:
            cutoff (float): a float to indicate the maximum edge length to be included
                in constructing the line graph, this is used to decrease computation
                complexity

        Return:
            line_graph:
                [[0, 1, 1, 2, 2],
                [0, 1, 1, 4, 23],
                [1, 4, 23, 5, 66],
                ... ...  ]
                the fist column specifies node(atom) index at this angle,
                the second column specifies 1st undirected edge(left bond) index,
                the third column specifies 1st directed edge(left bond) index,
                the fourth column specifies 2nd undirected edge(right bond) index,
                the fifth column specifies 2nd directed edge(right bond) index,.
            undirected2directed:
                [32, 45, ...]
                a list of length = num_undirected_edge that
                maps the undirected edge index to one of its directed edges indices
        """
        if len(self.directed_edges_list) != 2 * len(self.undirected_edges_list):
            raise ValueError(
                f"Error: number of directed edges={len(self.directed_edges_list)} != 2 * number of undirected edges={len(self.undirected_edges_list)}!This indicates directed edges are not complete"
            )
        line_graph = []
        undirected2directed = []
        for u_edge in self.undirected_edges_list:
            undirected2directed.append(u_edge.info["directed_edge_index"][0])
            if u_edge.info["distance"] > cutoff:
                continue
            if len(u_edge.info["directed_edge_index"]) != 2:
                raise ValueError(
                    f"Did not find 2 Directed_edges !!!undirected edge {u_edge} has:edge.info['directed_edge_index'] = {u_edge.info['directed_edge_index']}len directed_edges_list = {len(self.directed_edges_list)}len undirected_edges_list = {len(self.undirected_edges_list)}"
                )
            for center, dir_edge in zip(
                u_edge.nodes, u_edge.info["directed_edge_index"], strict=True
            ):
                for directed_edges in self.nodes[center].neighbors.values():
                    for directed_edge in directed_edges:
                        if directed_edge.index == dir_edge:
                            continue
                        if directed_edge.info["distance"] < cutoff:
                            line_graph.append(
                                [
                                    center,
                                    u_edge.index,
                                    dir_edge,
                                    directed_edge.info["undirected_edge_index"],
                                    directed_edge.index,
                                ]
                            )
        return line_graph, undirected2directed

    def undirected2directed(self) -> list[int]:
        """The index map from undirected_edge index to one of its directed_edge
        index.
        """
        return [
            undirected_edge.info["directed_edge_index"][0]
            for undirected_edge in self.undirected_edges_list
        ]

    def as_dict(self) -> dict:
        """Return dictionary serialization of a Graph."""
        return {
            "nodes": self.nodes,
            "directed_edges": self.directed_edges,
            "directed_edges_list": self.directed_edges_list,
            "undirected_edges": self.undirected_edges,
            "undirected_edges_list": self.undirected_edges_list,
        }

    def to(self, filename="graph.json") -> None:
        """Save graph dictionary to file."""
        write_json(self.as_dict(), filename)

    def __repr__(self) -> str:
        """Return string representation of the Graph."""
        num_nodes = len(self.nodes)
        num_directed_edges = len(self.directed_edges_list)
        num_undirected_edges = len(self.undirected_edges_list)
        return f"Graph(num_nodes={num_nodes!r}, num_directed_edges={num_directed_edges!r}, num_undirected_edges={num_undirected_edges!r})"
