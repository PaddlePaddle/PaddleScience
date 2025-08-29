# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from os import path as osp

import cairosvg
import hydra
import moflow_transform
import numpy as np
import paddle
import pandas as pd
from moflow_utils import Hyperparameters
from moflow_utils import _to_numpy_array
from moflow_utils import adj_to_smiles
from moflow_utils import check_novelty
from moflow_utils import check_validity
from moflow_utils import construct_mol
from moflow_utils import correct_mol
from moflow_utils import penalized_logp
from moflow_utils import valid_mol
from moflow_utils import valid_mol_can_with_seg
from omegaconf import DictConfig
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from tabulate import tabulate

import ppsci
from ppsci.utils import logger


def generate_mols(model, batch_size=20, temp=0.7, z_mu=None, true_adj=None):
    """generate mols

    Args:
        model (object): Generated eval Moflownet model
        batch_size (int, optional): Batch size during evaling per GPU. Defaults to 20.
        temp (float, optional): temperature of the gaussian distribution. Defaults to 0.7.
        z_mu (int, optional): latent vector of a molecule. Defaults to None.
        true_adj (paddle.Tensor, optional): True Adjacency. Defaults to None.

    Returns:
        Tuple(paddle.Tensor, paddle.Tensor): Adjacency and nodes
    """
    z_dim = model.b_size + model.a_size
    mu = np.zeros(z_dim)
    sigma_diag = np.ones(z_dim)
    if model.hyper_params.learn_dist:
        if len(model.ln_var) == 1:
            sigma_diag = np.sqrt(np.exp(model.ln_var.item())) * sigma_diag
        elif len(model.ln_var) == 2:
            sigma_diag[: model.b_size] = (
                np.sqrt(np.exp(model.ln_var[0].item())) * sigma_diag[: model.b_size]
            )
            sigma_diag[model.b_size + 1 :] = (
                np.sqrt(np.exp(model.ln_var[1].item())) * sigma_diag[model.b_size + 1 :]
            )
    sigma = temp * sigma_diag
    with paddle.no_grad():
        if z_mu is not None:
            mu = z_mu
            sigma = 0.01 * np.eye(z_dim)
        z = np.random.normal(mu, sigma, (batch_size, z_dim))
        z = paddle.to_tensor(data=z).astype(paddle.get_default_dtype())
        adj, x = model.reverse(z, true_adj=true_adj)
    return adj, x


def generate_mols_interpolation_grid(
    model, z0=None, true_adj=None, seed=0, mols_per_row=13, delta=1.0
):
    np.random.seed(seed)
    latent_size = model.b_size + model.a_size
    if z0 is None:
        mu = np.zeros([latent_size], dtype=np.float32)
        sigma = 0.02 * np.eye(latent_size, dtype=np.float32)
        z0 = np.random.multivariate_normal(mu, sigma).astype(np.float32)
    x = np.random.randn(latent_size)
    x /= np.linalg.norm(x)
    y = np.random.randn(latent_size)
    y -= y.dot(x) * x
    y /= np.linalg.norm(y)
    num_mols_to_edge = mols_per_row // 2
    z_list = []
    for dx in range(-num_mols_to_edge, num_mols_to_edge + 1):
        for dy in range(-num_mols_to_edge, num_mols_to_edge + 1):
            z = z0 + x * delta * dx + y * delta * dy
            z_list.append(z)
    z_array = paddle.to_tensor(data=z_list).astype(dtype="float32")
    adj, xf = model.reverse(z_array, true_adj=true_adj)
    return adj, xf


def visualize_interpolation_between_2_points(
    filepath,
    model,
    mol_smiles=None,
    mols_per_row=15,
    n_interpolation=100,
    seed=0,
    atomic_num_list=[6, 7, 8, 9, 0],
    true_data=None,
    device=None,
    data_name="qm9",
):
    if mol_smiles is not None:
        raise NotImplementedError
    else:
        with paddle.no_grad():
            np.random.seed(seed)
            mol_index = np.random.randint(0, len(true_data["edges"]), 2)
            adj0 = np.expand_dims(true_data["edges"][mol_index[0]], axis=0)
            x0 = np.expand_dims(true_data["nodes"][mol_index[0]], axis=0)
            adj0 = paddle.to_tensor(data=adj0)
            x0 = paddle.to_tensor(data=x0)
            smile0 = adj_to_smiles(adj0, x0, atomic_num_list)[0]
            mol0 = Chem.MolFromSmiles(smile0)
            fp0 = AllChem.GetMorganFingerprint(mol0, 2)
            adj1 = np.expand_dims(true_data["edges"][mol_index[1]], axis=0)
            x1 = np.expand_dims(true_data["nodes"][mol_index[1]], axis=0)
            adj1 = paddle.to_tensor(data=adj1)
            x1 = paddle.to_tensor(data=x1)
            smile1 = adj_to_smiles(adj1, x1, atomic_num_list)[0]
            # mol1 = Chem.MolFromSmiles(smile1)
            # fp1 = AllChem.GetMorganFingerprint(mol1, 2)
            logger.info("seed smile0: {}, seed smile1: {}".format(smile0, smile1))
            x_tumple0 = {"nodes": x0, "edges": adj0}
            # x_tumple1 = {"nodes": x1, "edges": adj1}
            output_dict = model(x_tumple0)
            z0 = output_dict["output"]
            z0[0] = z0[0].reshape([tuple(z0[0].shape)[0], -1])
            z0[1] = z0[1].reshape([tuple(z0[1].shape)[0], -1])
            z0 = paddle.concat(x=(z0[0], z0[1]), axis=1).squeeze(axis=0)
            z0 = _to_numpy_array(z0)

            output_dict = model(x_tumple0)
            z1 = output_dict["output"]
            z1[0] = z1[0].reshape([tuple(z1[0].shape)[0], -1])
            z1[1] = z1[1].reshape([tuple(z1[1].shape)[0], -1])
            z1 = paddle.concat(x=(z1[0], z1[1]), axis=1).squeeze(axis=0)
            z1 = _to_numpy_array(z1)
    d = z1 - z0
    z_list = [
        (z0 + i * 1.0 / (n_interpolation + 1) * d) for i in range(n_interpolation + 2)
    ]
    z_array = paddle.to_tensor(data=z_list).astype(dtype="float32")

    adjm, xm = model.reverse(z_array)
    adjm = _to_numpy_array(adjm)
    xm = _to_numpy_array(xm)
    interpolation_mols = [
        valid_mol(construct_mol(x_elem, adj_elem, atomic_num_list))
        for x_elem, adj_elem in zip(xm, adjm)
    ]
    valid_mols = [mol for mol in interpolation_mols if mol is not None]
    valid_mols_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
    valid_mols_smiles_unique = list(set(valid_mols_smiles))
    valid_mols_unique = [Chem.MolFromSmiles(s) for s in valid_mols_smiles_unique]
    valid_mols_smiles_unique_label = []
    for s, m in zip(valid_mols_smiles_unique, valid_mols_unique):
        fp = AllChem.GetMorganFingerprint(m, 2)
        sim = DataStructs.TanimotoSimilarity(fp, fp0)
        s = "{:.2f}\n".format(sim) + s
        if s == smile0:
            s = "***[" + s + "]***"
        valid_mols_smiles_unique_label.append(s)
    logger.info(
        "interpolation_mols valid {} / {}".format(
            len(valid_mols), len(interpolation_mols)
        )
    )
    if data_name == "qm9":
        psize = 200, 200
    else:
        psize = 200, 200
    img = Draw.MolsToGridImage(
        valid_mols_unique,
        legends=valid_mols_smiles_unique_label,
        molsPerRow=mols_per_row,
        subImgSize=psize,
    )
    img.save(filepath + "_.png")
    svg = Draw.MolsToGridImage(
        valid_mols_unique,
        legends=valid_mols_smiles_unique_label,
        molsPerRow=mols_per_row,
        subImgSize=psize,
        useSVG=True,
    )
    cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), write_to=filepath + ".pdf")
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=filepath + ".png")
    logger.message("Dump {}.png/pdf done".format(filepath))


def visualize_interpolation(
    filepath,
    model,
    mol_smiles=None,
    mols_per_row=13,
    delta=0.1,
    seed=0,
    atomic_num_list=[6, 7, 8, 9, 0],
    true_data=None,
    data_name="qm9",
    keep_duplicate=False,
    correct=True,
):
    if mol_smiles is not None:
        raise NotImplementedError
    else:
        with paddle.no_grad():
            np.random.seed(seed)
            mol_index = np.random.randint(0, len(true_data))
            adj = np.expand_dims(true_data["edges"][mol_index], axis=0)
            x = np.expand_dims(true_data["nodes"][mol_index], axis=0)
            # adj = paddle.to_tensor(data=adj)
            # x = paddle.to_tensor(data=x)
            smile0 = adj_to_smiles(adj, x, atomic_num_list)[0]
            mol0 = Chem.MolFromSmiles(smile0)
            fp0 = AllChem.GetMorganFingerprint(mol0, 2)
            logger.info("seed smile: {}".format(smile0))
            x_tumple = {"nodes": paddle.to_tensor(x), "edges": paddle.to_tensor(adj)}
            output_dict = model(x_tumple)
            z0 = output_dict["output"]
            z0[0] = z0[0].reshape([tuple(z0[0].shape)[0], -1])
            z0[1] = z0[1].reshape([tuple(z0[1].shape)[0], -1])
            z0 = paddle.concat(x=(z0[0], z0[1]), axis=1).squeeze(axis=0)
            z0 = _to_numpy_array(z0)
    adjm, xm = generate_mols_interpolation_grid(
        model, z0=z0, mols_per_row=mols_per_row, delta=delta, seed=seed
    )
    adjm = _to_numpy_array(adjm)
    xm = _to_numpy_array(xm)
    if correct:
        interpolation_mols = []
        for x_elem, adj_elem in zip(xm, adjm):
            mol = construct_mol(x_elem, adj_elem, atomic_num_list)
            cmol = correct_mol(mol)
            vcmol = valid_mol_can_with_seg(cmol)
            interpolation_mols.append(vcmol)
    else:
        interpolation_mols = [
            valid_mol(construct_mol(x_elem, adj_elem, atomic_num_list))
            for x_elem, adj_elem in zip(xm, adjm)
        ]
    valid_mols = [mol for mol in interpolation_mols if mol is not None]
    valid_mols_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
    if keep_duplicate:
        valid_mols_smiles_unique = valid_mols_smiles
    else:
        valid_mols_smiles_unique = list(set(valid_mols_smiles))
    valid_mols_unique = [Chem.MolFromSmiles(s) for s in valid_mols_smiles_unique]
    valid_mols_smiles_unique_label = []
    logger.info(
        "interpolation_mols:{}, valid_mols:{}, valid_mols_smiles_unique:{}".format(
            len(interpolation_mols), len(valid_mols), len(valid_mols_smiles_unique)
        )
    )
    for s, m in zip(valid_mols_smiles_unique, valid_mols_unique):
        fp = AllChem.GetMorganFingerprint(m, 2)
        sim = DataStructs.TanimotoSimilarity(fp, fp0)
        s = " {:.2f}".format(sim)
        valid_mols_smiles_unique_label.append(s)
    if keep_duplicate:
        molsPerRow = mols_per_row
    else:
        molsPerRow = 9
    k = len(valid_mols_smiles_unique)
    logger.info(
        "interpolation_mols valid {} / {}".format(
            len(valid_mols), len(interpolation_mols)
        )
    )
    if data_name == "qm9":
        psize = 150, 150
    else:
        psize = 150, 150
    img = Draw.MolsToGridImage(
        valid_mols_unique[:k],
        molsPerRow=molsPerRow,
        legends=valid_mols_smiles_unique_label[:k],
        subImgSize=psize,
    )
    img.save(filepath + "_.png")
    svg = Draw.MolsToGridImage(
        valid_mols_unique[:k],
        molsPerRow=molsPerRow,
        legends=valid_mols_smiles_unique_label[:k],
        subImgSize=psize,
        useSVG=True,
    )
    cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), write_to=filepath + ".pdf")
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=filepath + ".png")
    logger.info("Dump {}.png/pdf done".format(filepath))


def evaluate(cfg: DictConfig):
    # set training hyper-parameters
    b_hidden_ch = cfg.get(cfg.data_name).b_hidden_ch
    a_hidden_gnn = cfg.get(cfg.data_name).a_hidden_gnn
    a_hidden_lin = cfg.get(cfg.data_name).a_hidden_lin
    mask_row_size_list = list(cfg.get(cfg.data_name).mask_row_size_list)
    mask_row_stride_list = list(cfg.get(cfg.data_name).mask_row_stride_list)
    a_n_type = len(cfg.get(cfg.data_name).atomic_num_list)
    atomic_num_list = list(cfg.get(cfg.data_name).atomic_num_list)

    model_params = Hyperparameters(
        b_n_type=cfg.get(cfg.data_name).b_n_type,
        b_n_flow=cfg.get(cfg.data_name).b_n_flow,
        b_n_block=cfg.get(cfg.data_name).b_n_block,
        b_n_squeeze=cfg.get(cfg.data_name).b_n_squeeze,
        b_hidden_ch=b_hidden_ch,
        b_affine=True,
        b_conv_lu=cfg.get(cfg.data_name).b_conv_lu,
        a_n_node=cfg.get(cfg.data_name).a_n_node,
        a_n_type=a_n_type,
        a_hidden_gnn=a_hidden_gnn,
        a_hidden_lin=a_hidden_lin,
        a_n_flow=cfg.get(cfg.data_name).a_n_flow,
        a_n_block=cfg.get(cfg.data_name).a_n_block,
        mask_row_size_list=mask_row_size_list,
        mask_row_stride_list=mask_row_stride_list,
        a_affine=True,
        learn_dist=cfg.get(cfg.data_name).learn_dist,
        seed=cfg.seed,
        noise_scale=cfg.get(cfg.data_name).noise_scale,
    )

    logger.info("Model params:\n" + tabulate(model_params.print()))

    batch_size = cfg.EVAL.batch_size

    # set model for testing
    model_cfg = dict(cfg.MODEL)
    model_cfg.update({"hyper_params": model_params})
    model = ppsci.arch.MoFlowNet(**model_cfg)
    ppsci.utils.save_load.load_pretrain(model, path=cfg.EVAL.pretrained_model_path)
    model.eval()

    # set transforms
    if cfg.data_name == "qm9":
        transform_fn = moflow_transform.transform_fn
    elif cfg.data_name == "zinc250k":
        transform_fn = moflow_transform.transform_fn_zinc250k
        cfg.Random.update({"delta": 0.1})

    # set select eval model
    cfg.EVAL.update(cfg.get(cfg.EVAL_mode))
    # set select eval data
    valid_idx_path = osp.join(cfg.FILE_PATH, cfg.get(cfg.data_name).valid_idx)
    valid_idx = moflow_transform.get_val_ids(valid_idx_path, cfg.data_name)

    # set dataloader config
    dataloader_cfg = {
        "dataset": {
            "name": "MOlFLOWDataset",
            "file_path": cfg.FILE_PATH,
            "data_name": cfg.data_name,
            "mode": cfg.mode,
            "valid_idx": valid_idx,
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.get(cfg.data_name).label_keys,
            "smiles_col": cfg.get(cfg.data_name).smiles_col,
            "transform_fn": transform_fn,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "batch_size": cfg.EVAL.batch_size,
        "num_workers": cfg.EVAL.num_workers,
    }

    test = ppsci.data.dataset.build_dataset(dataloader_cfg["dataset"])
    dataloader_cfg["dataset"].update({"mode": "train"})
    train = ppsci.data.dataset.build_dataset(dataloader_cfg["dataset"])
    logger.info(
        "{} in total, {}  training data, {}  testing data, {} batchsize, train/batchsize {}".format(
            len(train) + len(test),
            len(train),
            len(test),
            batch_size,
            len(train) / batch_size,
        )
    )

    if cfg.EVAL.reconstruct:
        train_dataloader = ppsci.data.build_dataloader(train, dataloader_cfg)
        reconstruction_rate_list = []
        max_iter = len(train_dataloader)
        input_keys = cfg.MODEL.input_keys
        output_keys = cfg.MODEL.output_keys
        for i, batch in enumerate(train_dataloader, start=0):
            output_dict = model(batch[0])
            x = batch[0][input_keys[0]]
            adj = batch[0][input_keys[1]]
            z = output_dict[output_keys[0]]
            z0 = z[0].reshape([tuple(z[0].shape)[0], -1])
            z1 = z[1].reshape([tuple(z[1].shape)[0], -1])
            adj_rev, x_rev = model.reverse(paddle.concat(x=[z0, z1], axis=1))
            reverse_smiles = adj_to_smiles(adj_rev.cpu(), x_rev.cpu(), atomic_num_list)
            train_smiles = adj_to_smiles(adj.cpu(), x.cpu(), atomic_num_list)
            lb = np.array([int(a != b) for a, b in zip(train_smiles, reverse_smiles)])
            idx = np.where(lb)[0]
            if len(idx) > 0:
                for k in idx:
                    logger.info(
                        "{}, train: {}, reverse: {}".format(
                            i * batch_size + k, train_smiles[k], reverse_smiles[k]
                        )
                    )
            reconstruction_rate = 1.0 - lb.mean()
            reconstruction_rate_list.append(reconstruction_rate)
            logger.message(
                "iter/total: {}/{}, reconstruction_rate:{}".format(
                    i, max_iter, reconstruction_rate
                )
            )
        reconstruction_rate_total = np.array(reconstruction_rate_list).mean()
        logger.message(
            "reconstruction_rate for all the train data:{} in {}".format(
                reconstruction_rate_total, len(train)
            )
        )
        exit(0)

    if cfg.EVAL.int2point:
        inputs = train.input
        labels = train.label
        items = []
        for idx in range(len(train)):
            input_item = [value[idx] for key, value in inputs.items()]
            label_item = [value[idx] for key, value in labels.items()]
            item = input_item + label_item
            item = transform_fn(item)
            items.append(item)
        items = np.array(items, dtype=object).T
        inputs = {key: np.stack(items[i], axis=0) for i, key in enumerate(inputs)}

        mol_smiles = None
        gen_dir = osp.join(cfg.output_dir, cfg.EVAL_mode)
        logger.message("Dump figure in {}".format(gen_dir))
        if not osp.exists(gen_dir):
            os.makedirs(gen_dir)
        for seed in range(cfg.EVAL.inter_times):
            filepath = osp.join(
                gen_dir, "2points_interpolation-2point_molecules_seed{}".format(seed)
            )
            visualize_interpolation_between_2_points(
                filepath,
                model,
                mol_smiles=mol_smiles,
                mols_per_row=15,
                n_interpolation=50,
                atomic_num_list=atomic_num_list,
                seed=seed,
                true_data=inputs,
                data_name=cfg.data_name,
            )
        exit(0)

    if cfg.EVAL.intgrid:
        inputs = train.input
        labels = train.label
        items = []
        for idx in range(len(train)):
            input_item = [value[idx] for key, value in inputs.items()]
            label_item = [value[idx] for key, value in labels.items()]
            item = input_item + label_item
            item = transform_fn(item)
            items.append(item)
        items = np.array(items, dtype=object).T
        inputs = {key: np.stack(items[i], axis=0) for i, key in enumerate(inputs)}

        mol_smiles = None
        gen_dir = os.path.join(cfg.output_dir, cfg.EVAL_mode)
        logger.message("Dump figure in {}".format(gen_dir))
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)
        for seed in range(cfg.EVAL.inter_times):
            filepath = os.path.join(
                gen_dir, "generated_interpolation-grid_molecules_seed{}".format(seed)
            )
            visualize_interpolation(
                filepath,
                model,
                mol_smiles=mol_smiles,
                mols_per_row=9,
                delta=cfg.EVAL.delta,
                atomic_num_list=atomic_num_list,
                seed=seed,
                true_data=inputs,
                data_name=cfg.data_name,
                keep_duplicate=True,
            )
            filepath = os.path.join(
                gen_dir,
                "generated_interpolation-grid_molecules_seed{}_unique".format(seed),
            )
            visualize_interpolation(
                filepath,
                model,
                mol_smiles=mol_smiles,
                mols_per_row=9,
                delta=cfg.EVAL.delta,
                atomic_num_list=atomic_num_list,
                seed=seed,
                true_data=inputs,
                data_name=cfg.data_name,
                keep_duplicate=False,
            )
        exit(0)

    inputs = train.input
    labels = train.label
    items = []
    for idx in range(len(train)):
        input_item = [value[idx] for key, value in inputs.items()]
        label_item = [value[idx] for key, value in labels.items()]
        item = input_item + label_item
        item = transform_fn(item)
        items.append(item)
    items = np.array(items, dtype=object).T
    inputs = {key: np.stack(items[i], axis=0) for i, key in enumerate(inputs)}

    train_x = [a for a in inputs["nodes"]]
    train_adj = [a for a in inputs["edges"]]
    train_smiles = adj_to_smiles(train_adj, train_x, atomic_num_list)

    valid_ratio = []
    unique_ratio = []
    novel_ratio = []
    abs_unique_ratio = []
    abs_novel_ratio = []
    for i in range(cfg.EVAL.n_experiments):
        adj, x = generate_mols(
            model, batch_size=batch_size, true_adj=None, temp=cfg.EVAL.temperature
        )
        val_res = check_validity(
            adj, x, atomic_num_list, correct_validity=cfg.EVAL.correct_validity
        )
        novel_r, abs_novel_r = check_novelty(
            val_res["valid_smiles"], train_smiles, tuple(x.shape)[0]
        )
        novel_ratio.append(novel_r)
        abs_novel_ratio.append(abs_novel_r)
        unique_ratio.append(val_res["unique_ratio"])
        abs_unique_ratio.append(val_res["abs_unique_ratio"])
        valid_ratio.append(val_res["valid_ratio"])
        # n_valid = len(val_res["valid_mols"])
        if cfg.save_score:
            assert len(val_res["valid_smiles"]) == len(val_res["valid_mols"])
            smiles_qed_plogp = [
                (sm, Descriptors.qed(mol), penalized_logp(mol))
                for sm, mol in zip(val_res["valid_smiles"], val_res["valid_mols"])
            ]
            smiles_qed_plogp.sort(key=lambda tup: tup[2], reverse=True)
            gen_dir = os.path.join(cfg.output_dir, cfg.EVAL_mode)
            os.makedirs(gen_dir, exist_ok=True)
            filepath = os.path.join(
                gen_dir, "smiles_qed_plogp_{}_RankedByPlogp.csv".format(i)
            )
            df = pd.DataFrame(
                smiles_qed_plogp, columns=["Smiles", "QED", "Penalized_logp"]
            )
            df.to_csv(filepath, index=None, header=True)
            smiles_qed_plogp.sort(key=lambda tup: tup[1], reverse=True)
            filepath2 = os.path.join(
                gen_dir, "smiles_qed_plogp_{}_RankedByQED.csv".format(i)
            )
            df2 = pd.DataFrame(
                smiles_qed_plogp, columns=["Smiles", "QED", "Penalized_logp"]
            )
            df2.to_csv(filepath2, index=None, header=True)
        if cfg.EVAL.save_fig:
            gen_dir = os.path.join(cfg.output_dir, cfg.EVAL_mode)
            os.makedirs(gen_dir, exist_ok=True)
            filepath = os.path.join(gen_dir, "generated_mols_{}.png".format(i))
            img = Draw.MolsToGridImage(
                val_res["valid_mols"],
                legends=val_res["valid_smiles"],
                molsPerRow=20,
                subImgSize=(300, 300),
            )
            img.save(filepath)
    logger.info(
        "validity: mean={:.2f}%, sd={:.2f}%, vals={}".format(
            np.mean(valid_ratio), np.std(valid_ratio), valid_ratio
        )
    )
    logger.info(
        "novelty: mean={:.2f}%, sd={:.2f}%, vals={}".format(
            np.mean(novel_ratio), np.std(novel_ratio), novel_ratio
        )
    )
    logger.info(
        "uniqueness: mean={:.2f}%, sd={:.2f}%, vals={}".format(
            np.mean(unique_ratio), np.std(unique_ratio), unique_ratio
        )
    )
    logger.info(
        "abs_novelty: mean={:.2f}%, sd={:.2f}%, vals={}".format(
            np.mean(abs_novel_ratio), np.std(abs_novel_ratio), abs_novel_ratio
        )
    )
    logger.info(
        "abs_uniqueness: mean={:.2f}%, sd={:.2f}%, vals={}".format(
            np.mean(abs_unique_ratio), np.std(abs_unique_ratio), abs_unique_ratio
        )
    )


@hydra.main(version_base=None, config_path="./conf", config_name="moflow_test.yaml")
def main(cfg: DictConfig):
    evaluate(cfg)


if __name__ == "__main__":
    main()
