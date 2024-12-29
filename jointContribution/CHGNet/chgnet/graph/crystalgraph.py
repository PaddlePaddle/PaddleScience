from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import Any

import paddle

if TYPE_CHECKING:
    from typing_extensions import Self
DTYPE = "float32"


class CrystalGraph:
    """A data class for crystal graph."""

    def __init__(
        self,
        atomic_number: paddle.Tensor,
        atom_frac_coord: paddle.Tensor,
        atom_graph: paddle.Tensor,
        atom_graph_cutoff: float,
        neighbor_image: paddle.Tensor,
        directed2undirected: paddle.Tensor,
        undirected2directed: paddle.Tensor,
        bond_graph: paddle.Tensor,
        bond_graph_cutoff: float,
        lattice: paddle.Tensor,
        graph_id: (str | None) = None,
        mp_id: (str | None) = None,
        composition: (str | None) = None,
    ) -> None:
        """Initialize the crystal graph.

        Attention! This data class is not intended to be created manually. CrystalGraph
        should be returned by a CrystalGraphConverter

        Args:
            atomic_number (Tensor): the atomic numbers of atoms in the structure
                [n_atom]
            atom_frac_coord (Tensor): the fractional coordinates of the atoms
                [n_atom, 3]
            atom_graph (Tensor): a directed graph adjacency list,
                (center atom indices, neighbor atom indices, undirected bond index)
                for bonds in bond_fea
                [num_directed_bonds, 2]
            atom_graph_cutoff (float): the cutoff radius to draw edges in atom_graph
            neighbor_image (Tensor): the periodic image specifying the location of
                neighboring atom
                see: https://github.com/materialsproject/pymatgen/blob/ca2175c762e37ea7
                c9f3950ef249bc540e683da1/pymatgen/core/structure.py#L1485-L1541
                [num_directed_bonds, 3]
            directed2undirected (Tensor): the mapping from directed edge index to
                undirected edge index for the atom graph
                [num_directed_bonds]
            undirected2directed (Tensor): the mapping from undirected edge index to
                one of its directed edge index, this is essentially the inverse
                mapping of the directed2undirected this tensor is needed for
                computation efficiency.
                Note that num_directed_bonds = 2 * num_undirected_bonds
                [num_undirected_bonds]
            bond_graph (Tensor): a directed graph adjacency list,
                (atom indices, 1st undirected bond idx, 1st directed bond idx,
                2nd undirected bond idx, 2nd directed bond idx) for angles in angle_fea
                [n_angle, 5]
            bond_graph_cutoff (float): the cutoff bond length to include bond
                as nodes in bond_graph
            lattice (Tensor): lattices of the input structure
                [3, 3]
            graph_id (str | None): an id to keep track of this crystal graph
                Default = None
            mp_id (str | None): Materials Project id of this structure
                Default = None
            composition: Chemical composition of the compound, used just for
                better tracking of the graph
                Default = None.

        Raises:
            ValueError: if len(directed2undirected) != 2 * len(undirected2directed)
        """
        super().__init__()
        self.atomic_number = atomic_number
        self.atom_frac_coord = atom_frac_coord
        self.atom_graph = atom_graph
        self.atom_graph_cutoff = atom_graph_cutoff
        self.neighbor_image = neighbor_image
        self.directed2undirected = directed2undirected
        self.undirected2directed = undirected2directed
        self.bond_graph = bond_graph
        self.bond_graph_cutoff = bond_graph_cutoff
        self.lattice = lattice
        self.graph_id = graph_id
        self.mp_id = mp_id
        self.composition = composition
        if len(directed2undirected) != 2 * len(undirected2directed):
            raise ValueError(
                f"{graph_id} number of directed indices ({len(directed2undirected)}) != 2 * number of undirected indices ({2 * len(undirected2directed)})!"
            )

    def to(self, device: str = "cpu") -> CrystalGraph:
        """Move the graph to a device. Default = 'cpu'."""
        return CrystalGraph(
            atomic_number=self.atomic_number.to(device),
            atom_frac_coord=self.atom_frac_coord.to(device),
            atom_graph=self.atom_graph.to(device),
            atom_graph_cutoff=self.atom_graph_cutoff,
            neighbor_image=self.neighbor_image.to(device),
            directed2undirected=self.directed2undirected.to(device),
            undirected2directed=self.undirected2directed.to(device),
            bond_graph=self.bond_graph.to(device),
            bond_graph_cutoff=self.bond_graph_cutoff,
            lattice=self.lattice.to(device),
            graph_id=self.graph_id,
            mp_id=self.mp_id,
            composition=self.composition,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the graph to a dictionary."""
        return {
            "atomic_number": self.atomic_number,
            "atom_frac_coord": self.atom_frac_coord,
            "atom_graph": self.atom_graph,
            "atom_graph_cutoff": self.atom_graph_cutoff,
            "neighbor_image": self.neighbor_image,
            "directed2undirected": self.directed2undirected,
            "undirected2directed": self.undirected2directed,
            "bond_graph": self.bond_graph,
            "bond_graph_cutoff": self.bond_graph_cutoff,
            "lattice": self.lattice,
            "graph_id": self.graph_id,
            "mp_id": self.mp_id,
            "composition": self.composition,
        }

    def save(self, fname: (str | None) = None, save_dir: str = ".") -> str:
        """Save the graph to a file.

        Args:
            fname (str, optional): File name. Defaults to None.
            save_dir (str, optional): Directory to save the file. Defaults to ".".

        Returns:
            str: The path to the saved file.
        """
        if fname is not None:
            save_name = os.path.join(save_dir, fname)
        elif self.graph_id is not None:
            save_name = os.path.join(save_dir, f"{self.graph_id}.pt")
        else:
            save_name = os.path.join(save_dir, f"{self.composition}.pt")
        paddle.save(obj=self.to_dict(), path=save_name)
        return save_name

    @classmethod
    def from_file(cls, file_name: str) -> Self:
        """Load a crystal graph from a file.

        Args:
            file_name (str): The path to the file.

        Returns:
            CrystalGraph: The loaded graph.
        """
        return paddle.load(path=str(file_name))

    @classmethod
    def from_dict(cls, dic: dict[str, Any]) -> Self:
        """Load a CrystalGraph from a dictionary."""
        return cls(**dic)

    def __repr__(self) -> str:
        """String representation of the graph."""
        composition = self.composition
        atom_graph_cutoff = self.atom_graph_cutoff
        bond_graph_cutoff = self.bond_graph_cutoff
        atom_graph_len = self.atom_graph
        n_atoms = len(self.atomic_number)
        atom_graph_len = len(self.atom_graph)
        bond_graph_len = len(self.bond_graph)
        return f"CrystalGraph(composition={composition!r}, atom_graph_cutoff={atom_graph_cutoff!r}, bond_graph_cutoff={bond_graph_cutoff!r}, n_atoms={n_atoms!r}, atom_graph_len={atom_graph_len!r}, bond_graph_len={bond_graph_len!r})"

    @property
    def num_isolated_atoms(self) -> int:
        """Number of isolated atoms given the atom graph cutoff
        Isolated atoms are disconnected nodes in the atom graph
        that will not get updated in CHGNet.
        These atoms will always have calculated force equal to zero.

        With the default CHGNet atom graph cutoff radius, only ~ 0.1% of MPtrj dataset
        structures has isolated atoms.
        """
        return len(self.atomic_number) - paddle.unique(x=self.atom_graph[:, 0]).size
