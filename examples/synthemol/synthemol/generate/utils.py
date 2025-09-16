"""Utility functions for generating molecules."""
from functools import cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from synthemol.constants import FINGERPRINT_TYPES
from synthemol.constants import MODEL_TYPES
from synthemol.generate.node import Node
from synthemol.models.chemprop_models import chemprop_predict_on_molecule_ensemble
from synthemol.models.sklearn_models import sklearn_predict_on_molecule_ensemble


def create_model_scoring_fn(
    model_path: Path,
    model_type: MODEL_TYPES,
    fingerprint_type: (FINGERPRINT_TYPES) = None,
    smiles_to_score: (dict[str, float]) = None,
) -> Callable[[str], float]:
    """Creates a function that scores a molecule using a model or ensemble of models.

    :param model_path: A path to a model or directory of models.
    :param model_type: The type of model.
    :param fingerprint_type: The type of fingerprint to use.
    :param smiles_to_score: An optional dictionary mapping SMILES to precomputed scores.
    :return: A function that scores a molecule using a model or ensemble of models.
    """
    if model_type != "chemprop" and fingerprint_type is None:
        raise ValueError("Must define fingerprint_type if using a scikit-learn model.")
    """
    if model_path.is_dir():
        model_paths = list(model_path.glob('**/*.pt' if model_type ==
            'chemprop' else '**/*.pkl'))
        if len(model_paths) == 0:
            raise ValueError(
                f'Could not find any models in directory {model_path}.')
    else:
        model_paths = [model_path]
    """  # hth: skip for pre-computed
    if model_type == "chemprop":
        """
        paddle.seed(seed=0)
        models = [chemprop_load(model_path=model_path) for model_path in
            model_paths]
        scalers = [chemprop_load_scaler(model_path=model_path) for
            model_path in model_paths]
        """  # hth
        models = []
        scalers = []  # hth

        def model_scorer(smiles: str, fingerprint: (np.ndarray) = None) -> float:
            return chemprop_predict_on_molecule_ensemble(
                models=models, smiles=smiles, fingerprint=fingerprint, scalers=scalers
            )

    else:
        """
        models = [sklearn_load(model_path=model_path) for model_path in
            model_paths]
        """
        # skip for model_type=chemprop only
        models = []

        def model_scorer(smiles: str, fingerprint: np.ndarray) -> float:
            return sklearn_predict_on_molecule_ensemble(
                models=models, fingerprint=fingerprint
            )

    @cache
    def model_scoring_fn(smiles: str) -> float:
        if smiles_to_score is not None and smiles in smiles_to_score:
            return smiles_to_score[smiles]
        if fingerprint_type is not None:
            # fingerprint = compute_fingerprint(smiles, fingerprint_type=
            #    fingerprint_type)
            pass  # hth
        else:
            fingerprint = None
        return model_scorer(smiles=smiles, fingerprint=fingerprint)

    return model_scoring_fn


def save_generated_molecules(
    nodes: list[Node], building_block_id_to_smiles: dict[int, str], save_path: Path
) -> None:
    """Save generated molecules to a CSV file.

    :param nodes: A list of Nodes containing molecules. Only nodes with a single molecule are saved.
    :param building_block_id_to_smiles: A dictionary mapping building block IDs to SMILES.
    :param save_path: A path to a CSV file where the molecules will be saved.
    """
    nodes = [node for node in nodes if node.num_molecules == 1]
    construction_dicts = []
    max_reaction_num = 0
    reaction_num_to_max_reactant_num = {}
    for node in nodes:
        construction_dict = {"num_reactions": len(node.construction_log)}
        max_reaction_num = max(max_reaction_num, len(node.construction_log))
        for reaction_index, reaction_log in enumerate(node.construction_log):
            reaction_num = reaction_index + 1
            construction_dict[f"reaction_{reaction_num}_id"] = reaction_log[
                "reaction_id"
            ]
            reaction_num_to_max_reactant_num[reaction_num] = max(
                reaction_num_to_max_reactant_num.get(reaction_num, 0),
                len(reaction_log["building_block_ids"]),
            )
            for reactant_index, building_block_id in enumerate(
                reaction_log["building_block_ids"]
            ):
                reactant_num = reactant_index + 1
                construction_dict[
                    f"building_block_{reaction_num}_{reactant_num}_id"
                ] = building_block_id
                construction_dict[
                    f"building_block_{reaction_num}_{reactant_num}_smiles"
                ] = building_block_id_to_smiles.get(building_block_id, "")
        construction_dicts.append(construction_dict)
    columns = [
        "smiles",
        "node_id",
        "num_expansions",
        "rollout_num",
        "score",
        "Q_value",
        "num_reactions",
    ]
    for reaction_num in range(1, max_reaction_num + 1):
        columns.append(f"reaction_{reaction_num}_id")
        for reactant_num in range(
            1, reaction_num_to_max_reactant_num[reaction_num] + 1
        ):
            columns.append(f"building_block_{reaction_num}_{reactant_num}_id")
            columns.append(f"building_block_{reaction_num}_{reactant_num}_smiles")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(
        data=[
            {
                "smiles": node.molecules[0],
                "node_id": node.node_id,
                "num_expansions": node.N,
                "rollout_num": node.rollout_num,
                "score": node.P,
                "Q_value": node.Q(),
                **construction_dict,
            }
            for node, construction_dict in zip(nodes, construction_dicts)
        ],
        columns=columns,
    )
    data.to_csv(save_path, index=False)
