"""QueryMol class for storing a SMARTS query and associated helper functions."""
from functools import cache
from typing import Iterable

try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
from synthemol.utils import convert_to_mol
from synthemol.utils import strip_atom_mapping


class QueryMol:
    """Contains a molecule query in the form of a SMARTS string along with helper functions.

    :param smarts: A SMARTS string representing the molecular query.
    """

    def __init__(self, smarts: str) -> None:
        self.smarts_with_atom_mapping = smarts
        self.smarts = strip_atom_mapping(smarts)
        self.query_mol = Chem.MolFromSmarts(self.smarts)
        self._all_building_block_set = self._all_building_block_list = None
        (self._allowed_building_block_set) = self._allowed_building_block_list = None

    @property
    def all_building_blocks(self) -> (list[str]):
        """Gets a set of all building block SMILES for this QueryMol."""
        return self._all_building_block_list

    @all_building_blocks.setter
    def all_building_blocks(self, all_building_blocks: Iterable[str]) -> None:
        """Sets the all building block SMILES for this QueryMol. Can only be set once.

        Note: Filters the all SMILES to only include those that match the SMARTS of this QueryMol.

        :param all_building_blocks: An iterable of building block SMILES that are allowed for this QueryMol.
        """
        if self._all_building_block_set is not None:
            raise ValueError("All building block SMILES has already been set.")
        self._all_building_block_set = set(all_building_blocks)
        self._all_building_block_list = sorted(self._all_building_block_set)

    @property
    def allowed_building_blocks(self) -> (list[str]):
        """Gets a sorted list of allowed building block SMILES for this QueryMol."""
        return self._allowed_building_block_list

    @allowed_building_blocks.setter
    def allowed_building_blocks(self, allowed_building_blocks: Iterable[str]) -> None:
        """Sets the allowed building block SMILES for this QueryMol. Can only be set once.

        :param allowed_building_blocks: An iterable of building block SMILES that are allowed for this QueryMol.
        """
        if self._allowed_building_block_set is not None:
            raise ValueError("Allowed building block SMILES has already been set.")
        self._allowed_building_block_set = set(allowed_building_blocks)
        self._allowed_building_block_list = sorted(self._allowed_building_block_set)

    def has_substruct_match(self, smiles: str) -> bool:
        """Determines whether the provided molecule includes this QueryMol as a substructure.

        Note: Only checks SMARTS match, not self.allowed_set.

        :param smiles: A SMILES string
        :return: True if the molecule includes this QueryMol as a substructure, False otherwise.
        """
        mol = convert_to_mol(smiles, add_hs=True)
        return mol.HasSubstructMatch(self.query_mol)

    @cache
    def has_match(self, smiles: str) -> bool:
        """Determines whether the provided molecule matches this QueryMol.

        Always checks for SMARTS match and additionally checks for match in self.allowed_set
        if self.allowed_set is not None.

        Note: Caching assumes that self._allow_set does not change.

        :param smiles: A SMILES string.
        :return: True if the molecule matches this QueryMol, False otherwise.
        """
        if (
            self._allowed_building_block_set is not None
            and smiles not in self._allowed_building_block_set
        ):
            if (
                self._all_building_block_set is not None
                and smiles not in self._all_building_block_set
            ):
                pass
            else:
                return False
        return self.has_substruct_match(smiles)

    def __str__(self) -> str:
        """Gets a string representation of the QueryMol."""
        return f"QueryMol({self.smarts})"
