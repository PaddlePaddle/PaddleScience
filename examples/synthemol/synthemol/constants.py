"""Contains constants shared throughout synthemol."""
import os
from typing import Literal

try:
    from rdkit.Chem import Mol
except ModuleNotFoundError:
    pass
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

CHEMBL_SMILES_COL = "Smiles"
REAL_SPACE_SIZE = 31507987117
REAL_REACTION_COL = "reaction"
REAL_BUILDING_BLOCK_COLS = ["reagent1", "reagent2", "reagent3", "reagent4"]
REAL_BUILDING_BLOCK_ID_COL = "reagent_id"
REAL_TYPE_COL = "Type"
REAL_SMILES_COL = "smiles"
SMILES_COL = "smiles"
ACTIVITY_COL = "activity"
SCORE_COL = "score"
MODEL_TYPE = Literal["random_forest", "mlp", "chemprop"]
FINGERPRINT_TYPES = Literal["morgan", "rdkit"]
MOLECULE_TYPE = Mol
SKLEARN_MODEL_TYPES = [
    RandomForestClassifier,
    RandomForestRegressor,
    MLPClassifier,
    MLPRegressor,
]
SKLEARN_MODEL_NAME_TYPES = Literal["random_forest", "mlp"]
MODEL_TYPES = Literal["random_forest", "mlp", "chemprop"]
DATASET_TYPES = Literal["classification", "regression"]
OPTIMIZATION_TYPES = Literal["maximize", "minimize"]
# with resources.path('synthemol', 'resources') as resources_dir:
#    DATA_DIR = resources_dir
DATA_DIR = os.path.join("./synthemol", "resources")
BUILDING_BLOCKS_PATH = DATA_DIR + "/real/building_blocks.csv"
REACTION_TO_BUILDING_BLOCKS_PATH = DATA_DIR + "/real/reaction_to_building_blocks.pkl"
