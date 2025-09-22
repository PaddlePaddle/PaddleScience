"""synthemol.reactions package."""
from .custom import CUSTOM_REACTIONS
from .query_mol import QueryMol
from .reaction import Reaction
from .real import REAL_REACTIONS
from .utils import load_and_set_allowed_reaction_building_blocks
from .utils import set_all_building_blocks

if CUSTOM_REACTIONS is None:
    REACTIONS: tuple[Reaction] = REAL_REACTIONS
else:
    REACTIONS: tuple[Reaction] = CUSTOM_REACTIONS

__all__ = [
    "CUSTOM_REACTIONS",
    "QueryMol",
    "Reaction",
    "REAL_REACTIONS",
    "load_and_set_allowed_reaction_building_blocks",
    "set_all_building_blocks",
    "REACTIONS",
]
