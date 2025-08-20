"""Utility functions for synthemol."""
import re
from typing import Any

import numpy as np

try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass

from .constants import MOLECULE_TYPE


def strip_atom_mapping(smarts: str) -> str:
    """Strips the atom mapping from a SMARTS (i.e., any ":" followed by digits).

    :param smarts: A SMARTS string with atom mapping indices.
    :return: The same SMARTS string but without the atom mapping indices.
    """
    return re.sub("\\[([^:]+)(:\\d+)]", "[\\1]", smarts)


def convert_to_mol(mol: MOLECULE_TYPE, add_hs: bool = False) -> Chem.Mol:
    """Converts a SMILES to an RDKit Mol object (if not already converted) and optionally adds Hs.

    :param mol: A SMILES string or an RDKit Mol object.
    :param add_hs: Whether to add Hs.
    :return: An RDKit Mol object with Hs added optionally.
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if add_hs:
        mol = Chem.AddHs(mol)
    return mol


def random_choice(
    rng: np.random.Generator, array: list[Any], size: (int) = None, replace: bool = True
) -> Any:
    """An efficient random choice function built on top of NumPy.

    :param rng: A NumPy random number generator.
    :param array: An array to choose from.
    :param size: The number of elements to choose. If None, a single element is chosen.
    :param replace: Whether to allow replacement.
    :return: A list of elements chosen from the array.
    """
    if size is None:
        return array[rng.integers(len(array))]
    return [array[i] for i in rng.choice(len(array), size=size, replace=replace)]
