"""Contains training and predictions functions for Chemprop models."""
from pathlib import Path
from typing import List

import numpy as np
import paddle
from nn_utils import activate_dropout

from ppsci.arch.chemprop_molecule import MoleculeModel
from ppsci.data.dataset.synthemol_dataset import MoleculeDataLoader
from ppsci.data.dataset.synthemol_dataset import MoleculeDatapoint
from ppsci.data.dataset.synthemol_dataset import MoleculeDataset
from ppsci.data.dataset.synthemol_dataset import StandardScaler


def my_chemprop_load(
    model,
    model_path: Path,
    device: (paddle.CPUPlace, paddle.CUDAPlace, str) = paddle.CPUPlace(),
) -> MoleculeModel:
    pass
    state_dict = paddle.load(str(model_path))
    model.set_state_dict(state_dict)
    return model


def predict(
    model: MoleculeModel,
    data_loader: MoleculeDataLoader,
    disable_progress_bar: bool = False,
    scaler: StandardScaler = None,
    return_unc_parameters: bool = False,
    dropout_prob: float = 0.0,
) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A :class:`~chemprop.features.scaler.StandardScaler` object fit on the training targets.
    :param return_unc_parameters: A bool indicating whether additional uncertainty parameters would be returned alongside the mean predictions.
    :param dropout_prob: For use during uncertainty prediction only. The propout probability used in generating a dropout ensemble.
    :return: A list of lists of predictions. The outer list is molecules while the inner list is tasks. If returning uncertainty parameters as well,
        it is a tuple of lists of lists, of a length depending on how many uncertainty parameters are appropriate for the loss function.
    """
    model.eval()
    if dropout_prob > 0.0:

        def activate_dropout_(model):
            return activate_dropout(model, dropout_prob)

        model.apply(activate_dropout_)

    preds = []

    var, lambdas, alphas, betas = [], [], [], []

    # for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
    for i, batch in enumerate(data_loader):
        # batch: MoleculeDataset
        # mol_batch = batch.batch_graph()
        # features_batch = batch.features()
        # atom_descriptors_batch = batch.atom_descriptors()
        # atom_features_batch = batch.atom_features()
        # bond_features_batch = batch.bond_features()
        # print("ceshi data:", batch, mol_batch, features_batch,
        #        atom_descriptors_batch, atom_features_batch,
        #        bond_features_batch)

        (
            mol_batch,
            features_batch,
            _,
            _,
            atom_descriptors_batch,
            atom_features_batch,
            bond_features_batch,
            _,
            _,
            _,
        ) = batch

        paddle_batch = {}
        paddle_batch["mol_batch"] = mol_batch
        paddle_batch["features_batch"] = features_batch
        paddle_batch["atom_descriptors_batch"] = atom_descriptors_batch
        paddle_batch["atom_features_batch"] = atom_features_batch
        paddle_batch["bond_features_batch"] = bond_features_batch

        with paddle.no_grad():
            # batch_preds = model(mol_batch, features_batch,
            #    atom_descriptors_batch, atom_features_batch,
            #    bond_features_batch)
            batch_preds = model(paddle_batch)["pred"]  # hth
        batch_preds = batch_preds.data.cpu().numpy()
        if model.loss_function == "mve":
            batch_preds, batch_var = np.split(batch_preds, 2, axis=1)
        elif model.loss_function == "dirichlet":
            if model.classification:
                batch_alphas = np.reshape(
                    batch_preds,
                    [tuple(batch_preds.shape)[0], tuple(batch_preds.shape)[1] // 2, 2],
                )
                batch_preds = batch_alphas[:, :, 1] / np.sum(batch_alphas, axis=2)
            elif model.multiclass:
                batch_alphas = batch_preds
                batch_preds = batch_preds / np.sum(batch_alphas, axis=2, keepdims=True)
        elif model.loss_function == "evidential":
            batch_preds, batch_lambdas, batch_alphas, batch_betas = np.split(
                batch_preds, 4, axis=1
            )
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)
            if model.loss_function == "mve":
                batch_var = batch_var * scaler.stds**2
            elif model.loss_function == "evidential":
                batch_betas = batch_betas * scaler.stds**2
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)
        if model.loss_function == "mve":
            var.extend(batch_var.tolist())
        elif model.loss_function == "dirichlet" and model.classification:
            alphas.extend(batch_alphas.tolist())
        elif model.loss_function == "evidential":
            lambdas.extend(batch_lambdas.tolist())
            alphas.extend(batch_alphas.tolist())
            betas.extend(batch_betas.tolist())
    if return_unc_parameters:
        if model.loss_function == "mve":
            return preds, var
        elif model.loss_function == "dirichlet":
            return preds, alphas
        elif model.loss_function == "evidential":
            return preds, lambdas, alphas, betas
    return preds


def chemprop_build_data_loader(
    smiles: list[str],
    fingerprints: np.ndarray = None,
    properties: list[int] = None,
    shuffle: bool = False,
    num_workers: int = 0,
) -> MoleculeDataLoader:
    """Builds a chemprop MoleculeDataLoader.

    :param smiles: A list of SMILES strings.
    :param fingerprints: A 2D array of molecular fingerprints (num_molecules, num_features).
    :param properties: A list of molecular properties (num_molecules,).
    :param shuffle: Whether to shuffle the data loader.
    :param num_workers: The number of workers for the data loader.
                        Zero workers needed for deterministic behavior and faster training/testing when CPU only.
    :return: A Chemprop data loader.
    """
    if fingerprints is None:
        fingerprints = [None] * len(smiles)

    if properties is None:
        properties = [None] * len(smiles)
    else:
        properties = [[float(prop)] for prop in properties]

    return MoleculeDataLoader(
        dataset=MoleculeDataset(
            [
                MoleculeDatapoint(
                    smiles=[smiles],
                    targets=prop,
                    features=fingerprint,
                )
                for smiles, fingerprint, prop in zip(smiles, fingerprints, properties)
            ]
        ),
        num_workers=num_workers,
        shuffle=shuffle,
    )


def chemprop_predict(
    model: MoleculeModel,
    smiles: list[str],
    fingerprints: np.ndarray,
    num_workers: int = 0,
) -> np.ndarray:
    """Predicts molecular properties using a Chemprop model.

    :param model: A Chemprop model.
    :param smiles: A list of SMILES strings.
    :param fingerprints: A 2D array of molecular fingerprints (num_molecules, num_features).
    :param num_workers: The number of workers for the data loader.
    :return: A 1D array of predicted properties (num_molecules,).
    """
    # Set up data loader
    data_loader = chemprop_build_data_loader(
        smiles=smiles, fingerprints=fingerprints, num_workers=num_workers
    )

    _chemprop_predict = predict

    # pred = _chemprop_predict(model=model, data_loader=data_loader)

    # Make predictions
    print(len(data_loader))
    preds = np.array(_chemprop_predict(model=model, data_loader=data_loader))
    print(preds.shape)

    return preds
