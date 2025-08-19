"""Utility functions for synthemol reactions."""
import pickle
from pathlib import Path

from synthemol.reactions.reaction import Reaction


def set_all_building_blocks(
    reactions: tuple[Reaction], building_blocks: set[str]
) -> None:
    """Sets the allowed building block SMILES for all Reactions in a list of Reactions.

    Note: Modifies Reactions in place.

    :param reactions: A tuple of Reactions whose allowed building block SMILES will be set.
    :param building_blocks: A set of allowed building block SMILES.
    """
    for reaction in reactions:
        for reactant in reaction.reactants:
            reactant.all_building_blocks = building_blocks


def load_and_set_allowed_reaction_building_blocks(
    reactions: tuple[Reaction], reaction_to_reactant_to_building_blocks_path: Path
) -> None:
    """Loads a mapping of allowed building blocks for each reaction and sets the allowed SMILES for each reaction.

    :param reactions: A tuple of Reactions whose allowed SMILES will be set.
    :param reaction_to_reactant_to_building_blocks_path: Path to a PKL file mapping from reaction ID
                                                         to reactant index to a set of allowed building block SMILES.
    """
    with open(reaction_to_reactant_to_building_blocks_path, "rb") as f:
        reaction_to_reactant_to_building_blocks: dict[
            int, dict[int, set[str]]
        ] = pickle.load(f)
    for reaction in reactions:
        for reactant_index, reactant in enumerate(reaction.reactants):
            reactant.allowed_building_blocks = reaction_to_reactant_to_building_blocks[
                reaction.id
            ][reactant_index]
