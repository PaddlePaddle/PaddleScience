"""Contains the Node class, which represents a step in the combinatorial molecule construction process."""
import math
from functools import cached_property
from typing import Any
from typing import Callable


class Node:
    """A Node represents a step in the combinatorial molecule construction process.

    :param explore_weight: The hyperparameter that encourages exploration.
    :param scoring_fn: A function that takes as input a SMILES representing a molecule and returns a score.
    :param node_id: The ID of the Node, which should correspond to the order in which the Mode was visited.
    :param molecules: A tuple of SMILES. The first element is the currently constructed molecule
                        while the remaining elements are the building blocks that are about to be added.
    :param unique_building_block_ids: A set of building block IDS used in this Node.
    :param construction_log: A tuple of dictionaries containing information about each reaction
                                used to construct the molecules in this Node.
    :param rollout_num: The number of the rollout on which this Node was created.
    """

    def __init__(
        self,
        explore_weight: float,
        scoring_fn: Callable[[str], float],
        node_id: (int) = None,
        molecules: (tuple[str]) = None,
        unique_building_block_ids: (set[int]) = None,
        construction_log: (tuple[dict[str, Any]]) = None,
        rollout_num: (int) = None,
    ) -> None:

        self.explore_weight = explore_weight
        self.scoring_fn = scoring_fn
        self.node_id = node_id
        self.molecules = molecules if molecules is not None else tuple()
        self.unique_building_block_ids = (
            unique_building_block_ids
            if unique_building_block_ids is not None
            else set()
        )
        self.construction_log = (
            construction_log if construction_log is not None else tuple()
        )
        self.W = 0.0
        self.N = 0
        self.rollout_num = rollout_num
        self.num_children = 0

    @classmethod
    def compute_score(
        cls, molecules: tuple[str], scoring_fn: Callable[[str], float]
    ) -> float:
        """Computes the score of the molecules.

        :param molecules: A tuple of SMILES. The first element is the currently constructed molecule
                          while the remaining elements are the building blocks that are about to be added.
        :param scoring_fn: A function that takes as input a SMILES representing a molecule and returns a score.
        :return: The score of the molecules.
        """
        return (
            sum(scoring_fn(molecule) for molecule in molecules) / len(molecules)
            if len(molecules) > 0
            else 0.0
        )

    @cached_property
    def P(self) -> float:
        """The property score of this Node. (Note: The value is cached, so it assumes the Node is immutable.)"""
        return self.compute_score(molecules=self.molecules, scoring_fn=self.scoring_fn)

    def Q(self) -> float:
        """Value that encourages exploitation of Nodes with high reward."""
        return self.W / self.N if self.N > 0 else 0.0

    def U(self, n: int) -> float:
        """Value that encourages exploration of Nodes with few visits."""
        return self.explore_weight * self.P * math.sqrt(1 + n) / (1 + self.N)

    @property
    def num_molecules(self) -> int:
        """Gets the number of building blocks in the Node."""
        return len(self.molecules)

    @property
    def num_reactions(self) -> int:
        """Gets the number of reactions used so far to generate the molecule in the Node."""
        return len(self.construction_log)

    def __hash__(self) -> int:
        """Hashes the Node based on the building blocks."""
        return hash(self.molecules)

    def __eq__(self, other: Any) -> bool:
        """Checks if the Node is equal to another Node based on the building blocks."""
        if not isinstance(other, Node):
            return False
        return self.molecules == other.molecules
