"""Reaction class and helper functions."""
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ModuleNotFoundError:
    pass
from synthemol.reactions.query_mol import QueryMol
from synthemol.utils import MOLECULE_TYPE
from synthemol.utils import convert_to_mol


class Reaction:
    """A chemical reaction including SMARTS for the reactants, product, and reaction along with helper functions.

    :param reactants: A list of QueryMols containing the reactants of the reaction.
    :param product: A QueryMol containing the product of the reaction.
    :param reaction_id: The ID of the reaction.
    """

    def __init__(
        self, reactants: list[QueryMol], product: QueryMol, reaction_id: (int) = None
    ) -> None:
        self.reactants = reactants
        self.product = product
        self.id = reaction_id
        self.reaction_smarts = f"{'.'.join(f'({reactant.smarts_with_atom_mapping})' for reactant in self.reactants)}>>({self.product.smarts_with_atom_mapping})"
        self.reaction = AllChem.ReactionFromSmarts(self.reaction_smarts)

    def get_reactant_matches(self, smiles: str) -> list[int]:
        """Gets the indices of the reactants that match the provided SMILES.

        :param smiles: The SMILES to match.
        :return: A list of indices of the reactants that match the provided SMILES.
        """
        return [
            reactant_index
            for reactant_index, reactant in enumerate(self.reactants)
            if reactant.has_match(smiles)
        ]

    @property
    def num_reactants(self) -> int:
        """Gets the number of reactants in the reaction."""
        return len(self.reactants)

    def run_reactants(
        self, reactants: list[MOLECULE_TYPE]
    ) -> tuple[tuple[Chem.Mol, ...], ...]:
        """Runs the reaction on the provided reactants.

        :param reactants: A list of reactants.
        :return: A tuple of tuples of RDKit Mol objects representing the products of the reaction.
        """
        return self.reaction.RunReactants(
            [convert_to_mol(reactant, add_hs=True) for reactant in reactants]
        )

    def __repr__(self) -> str:
        """Gets the string representation of the Reaction."""
        return (
            f"{self.__class__.__name__}(id={self.id}, reaction={self.reaction_smarts})"
        )
