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

from os import path as osp

import hydra
import moflow_transform
import numpy as np
import paddle
import pandas as pd
from moflow_utils import Hyperparameters
from moflow_utils import adj_to_smiles
from moflow_utils import check_validity
from moflow_utils import penalized_logp
from omegaconf import DictConfig
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from tabulate import tabulate

import ppsci
from ppsci.data.dataset.moflow_dataset import MolGraph
from ppsci.utils import logger


def load_property_csv(filepath, normalize=True):
    """Use qed and plogp in zinc250k_property.csv which are recalculated by rdkit
    the recalculated qed results are in tiny inconsistent with qed in zinc250k.csv
    e.g
    zinc250k_property.csv:
    qed,plogp,smile
    0.7319008436872337,3.1399057164163766,CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1
    0.9411116113894995,0.17238635659148804,C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1
    import rdkit
    m = rdkit.Chem.MolFromSmiles('CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1')
    rdkit.Chem.QED.qed(m): 0.7319008436872337
    from mflow.utils.environment import penalized_logp
    penalized_logp(m):  3.1399057164163766
    However, in oringinal:
    zinc250k.csv
    ,smiles,logP,qed,SAS
    0,CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1,5.0506,0.702012232801,2.0840945720726807
    1,C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1,3.1137,0.928975488089,3.4320038192747795

    0.7319008436872337 v.s. 0.702012232801
    and no plogp in zinc250k.csv dataset!
    """
    df = pd.read_csv(filepath)
    if normalize:
        # m = df["plogp"].mean()
        # std = df["plogp"].std()
        # mn = df["plogp"].min()
        mx = df["plogp"].max()
        lower = -10
        df["plogp"] = df["plogp"].clip(lower=lower, upper=5)
        df["plogp"] = (df["plogp"] - lower) / (mx - lower)
    tuples = [tuple(x) for x in df.values]
    logger.info("Load {} done, length: {}".format(filepath, len(tuples)))
    return tuples


def smiles_to_adj(mol_smiles, data_name="qm9"):
    """Use simles to adj, atoms

    Args:
        mol_smiles: eg. CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1
    """
    if data_name == "qm9":
        out_size = 9
        transform_fn = moflow_transform.transform_fn
    elif data_name == "zinc250k":
        out_size = 38
        transform_fn = moflow_transform.transform_fn_zinc250k

    preprocessor = MolGraph(out_size=out_size, kekulize=True)
    canonical_smiles, mol = preprocessor.prepare_smiles_and_mol(
        Chem.MolFromSmiles(mol_smiles)
    )
    atoms, adj = preprocessor.get_input_features(mol)
    atoms, adj, _ = transform_fn((atoms, adj, None))
    adj = np.expand_dims(adj, axis=0)
    atoms = np.expand_dims(atoms, axis=0)
    adj = paddle.to_tensor(data=adj)
    atoms = paddle.to_tensor(data=atoms)
    return adj, atoms


def optimize_mol(
    model,
    property_model,
    smiles,
    sim_cutoff,
    lr=2.0,
    num_iter=20,
    data_name="qm9",
    atomic_num_list=[6, 7, 8, 9, 0],
    property_name="qed",
    debug=True,
    random=False,
):
    """General for Optimize model.

    Args:
        model: MoFlowNet pre-trained model
        property_model: Optimize qed or plogp model
        smiles: eg. CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1
        sim_cutoff: add similarity property
        lr: learning rate
        num_iter: learning total step
        data_name: dataset name
        atomic_num_list: atom list in smiles
        property_name: Optimize qed or plogp model name
        debug: To run optimization with more information
        random: Random Generation from sampling or not
    """
    if property_name == "qed":
        propf = Descriptors.qed
    elif property_name == "plogp":
        propf = penalized_logp
    else:
        raise ValueError("Wrong property_name{}".format(property_name))
    model.eval()
    property_model.eval()
    with paddle.no_grad():
        bond, atoms = smiles_to_adj(smiles, data_name)
        x = {"nodes": atoms, "edges": bond}
        mol_vec, _ = property_model.encode(x)
        if debug:
            adj_rev, x_rev = property_model.reverse(mol_vec)
            reverse_smiles = adj_to_smiles(adj_rev.cpu(), x_rev.cpu(), atomic_num_list)
            logger.info(smiles, reverse_smiles)
            output_dict = model(x)
            z = output_dict["output"]
            # sum_log_det_jacs = output_dict["sum_log_det"]
            z0 = z[0].reshape([tuple(z[0].shape)[0], -1])
            z1 = z[1].reshape([tuple(z[1].shape)[0], -1])
            adj_rev, x_rev = model.reverse(paddle.concat(x=[z0, z1], axis=1))
            reverse_smiles2 = adj_to_smiles(adj_rev.cpu(), x_rev.cpu(), atomic_num_list)
            train_smiles2 = adj_to_smiles(bond.cpu(), atoms.cpu(), atomic_num_list)
            logger.info(train_smiles2, reverse_smiles2)
    mol = Chem.MolFromSmiles(smiles)
    fp1 = AllChem.GetMorganFingerprint(mol, 2)
    start = smiles, propf(mol), None
    out_0 = mol_vec.clone().detach()
    out_0.stop_gradient = False
    cur_vec = out_0
    out_1 = mol_vec.clone().detach()
    out_1.stop_gradient = False
    start_vec = out_1
    visited = []
    for step in range(num_iter):
        prop_val = property_model.propNN(cur_vec).squeeze()
        grad = paddle.grad(outputs=prop_val, inputs=cur_vec)[0]
        if random:
            rad = paddle.randn(shape=cur_vec.data.shape, dtype=cur_vec.data.dtype)
            cur_vec = start_vec.data + lr * rad / paddle.sqrt(x=rad * rad)
        else:
            cur_vec = cur_vec.data + lr * grad.data / paddle.sqrt(
                x=grad.data * grad.data
            )
        out_2 = cur_vec.clone().detach()
        out_2.stop_gradient = False
        cur_vec = out_2
        visited.append(cur_vec)
    hidden_z = paddle.concat(x=visited, axis=0)
    adj, x = property_model.reverse(hidden_z)
    val_res = check_validity(adj, x, atomic_num_list, debug=debug)
    valid_mols = val_res["valid_mols"]
    valid_smiles = val_res["valid_smiles"]
    results = []
    sm_set = set()
    sm_set.add(smiles)
    for m, s in zip(valid_mols, valid_smiles):
        if s in sm_set:
            continue
        sm_set.add(s)
        p = propf(m)
        fp2 = AllChem.GetMorganFingerprint(m, 2)
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        if sim >= sim_cutoff:
            results.append((s, p, sim, smiles))
    results.sort(key=lambda tup: tup[1], reverse=True)
    return results, start


def fit_model(
    model,
    data,
    data_prop,
    N,
    property_name="qed",
    max_epochs=10,
    learning_rate=0.001,
    weight_decay=1e-05,
):
    """Train for Optimize model.

    Args:
        model: MoFlowNet pre-trained model
        data: dataloader
        data_prop: true smiles list
        N: dataset number
        property_name: Optimize qed or plogp model name
        smiles: eg. CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1
        max_epochs: train epochs
        learning_rate: train learning rate
        weight_decay: train weight_decay
    """
    model.train()
    metrics = paddle.nn.MSELoss()
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    assert len(data_prop) == N
    iter_per_epoch = len(data)
    log_step = 20
    if property_name == "qed":
        col = 0
    elif property_name == "plogp":
        col = 1
    else:
        raise ValueError("Wrong property_name{}".format(property_name))
    for epoch in range(max_epochs):
        for i, batch in enumerate(data):
            x = batch[0]["nodes"]
            bs = tuple(x.shape)[0]
            ps = i * bs
            pe = min((i + 1) * bs, N)
            true_y = [[tt[col]] for tt in data_prop[ps:pe]]
            true_y = (
                paddle.to_tensor(data=true_y)
                .astype(dtype="float32")
                .cuda(blocking=True)
            )
            optimizer.clear_grad()
            output_dict = model(batch[0])
            y = output_dict["output"][1]
            loss = metrics(y, true_y)
            loss.backward()
            optimizer.step()

            if (i + 1) % log_step == 0:
                logger.info(
                    "Epoch [{}/{}], Iter [{}/{}], loss: {:.5f},".format(
                        epoch + 1, max_epochs, i + 1, iter_per_epoch, loss.item()
                    )
                )
    return model


def find_top_score_smiles(
    model,
    property_model,
    data_name,
    property_name,
    train_prop,
    topk,
    atomic_num_list,
    debug,
    file_path,
):
    """
    Args:
        model: MoFlowNet pre-trained model
        property_model: Optimize qed or plogp model
        data_name: dataset name
        property_name: Optimize qed or plogp model name
        train_prop: true smiles list
        topk: Top k smiles as seeds
        atomic_num_list: atom list in smiles
        debug: To run optimization with more information
        file_path: result save path
    """
    if property_name == "qed":
        col = 0
    elif property_name == "plogp":
        col = 1
    logger.info("Finding top {} score".format(property_name))
    train_prop_sorted = sorted(train_prop, key=lambda tup: tup[col], reverse=True)
    result_list = []
    for i, r in enumerate(train_prop_sorted):
        if i >= topk:
            break
        if i % 50 == 0:
            logger.info("Optimization {}/{}".format(i, topk))
        qed, plogp, smile = r
        results, ori = optimize_mol(
            model,
            property_model,
            smile,
            sim_cutoff=0,
            lr=0.005,
            num_iter=100,
            data_name=data_name,
            atomic_num_list=atomic_num_list,
            property_name=property_name,
            random=False,
            debug=debug,
        )
        result_list.extend(results)
    result_list.sort(key=lambda tup: tup[1], reverse=True)
    train_smile = set()
    for i, r in enumerate(train_prop_sorted):
        qed, plogp, smile = r
        train_smile.add(smile)
        mol = Chem.MolFromSmiles(smile)
        smile2 = Chem.MolToSmiles(mol, isomericSmiles=True)
        train_smile.add(smile2)
    result_list_novel = []
    for i, r in enumerate(result_list):
        smile, score, sim, smile_original = r
        if smile not in train_smile:
            result_list_novel.append(r)
    save_file_path = osp.join(file_path, property_name + "_discovered_sorted.csv")
    f = open(save_file_path, "w")
    for r in result_list_novel:
        smile, score, sim, smile_original = r
        f.write("{},{},{},{}\n".format(score, smile, sim, smile_original))
        f.flush()
    f.close()
    logger.message("Dump done!")


def constrain_optimization_smiles(
    model,
    property_model,
    data_name,
    property_name,
    train_prop,
    topk,
    atomic_num_list,
    debug,
    file_path,
    sim_cutoff=0.0,
):
    """
    Args:
        model: MoFlowNet pre-trained model
        property_model: Optimize qed or plogp model
        data_name: dataset name
        property_name: Optimize qed or plogp model name
        train_prop: true smiles list
        topk: Top k smiles as seeds
        atomic_num_list: atom list in smiles
        debug: To run optimization with more information
        file_path: result save path
        sim_cutoff: add similarity property
    """
    if property_name == "qed":
        col = 0
    elif property_name == "plogp":
        col = 1
    logger.message("Constrained optimization of {} score".format(property_name))
    train_prop_sorted = sorted(train_prop, key=lambda tup: tup[col])
    result_list = []
    nfail = 0
    for i, r in enumerate(train_prop_sorted):
        if i >= topk:
            break
        if i % 50 == 0:
            logger.info("Optimization {}/{},".format(i, topk))
        qed, plogp, smile = r
        results, ori = optimize_mol(
            model,
            property_model,
            smile,
            sim_cutoff=sim_cutoff,
            lr=0.005,
            num_iter=100,
            data_name=data_name,
            atomic_num_list=atomic_num_list,
            property_name=property_name,
            random=False,
            debug=debug,
        )
        if len(results) > 0:
            smile2, property2, sim, _ = results[0]
            plogp_delta = property2 - plogp
            if plogp_delta >= 0:
                result_list.append(
                    (smile2, property2, sim, smile, qed, plogp, plogp_delta)
                )
            else:
                nfail += 1
                logger.info("Failure:{}:{}".format(i, smile))
        else:
            nfail += 1
            logger.info("Failure:{}:{}".format(i, smile))
    df = pd.DataFrame(
        result_list,
        columns=[
            "smile_new",
            "prop_new",
            "sim",
            "smile_old",
            "qed_old",
            "plogp_old",
            "plogp_delta",
        ],
    )
    logger.info(df.describe())
    save_file_path = osp.join(file_path, property_name + "_constrain_optimization.csv")
    df.to_csv(save_file_path, index=False)
    logger.message("Dump done!")
    logger.info("nfail:{} in total:{}".format(nfail, topk))
    logger.info("success rate: {}".format((topk - nfail) * 1.0 / topk))


def optimize(cfg: DictConfig):
    # set hyper-parameters
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

    hidden = cfg.OPTIMIZE.hidden
    logger.info("Hidden dim for output regression:{}".format(hidden))

    # set transforms
    if cfg.data_name == "qm9":
        transform_fn = moflow_transform.transform_fn
    elif cfg.data_name == "zinc250k":
        transform_fn = moflow_transform.transform_fn_zinc250k

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
        "batch_size": cfg.OPTIMIZE.batch_size,
        "num_workers": 0,
    }

    # set model
    model_cfg = dict(cfg.MODEL)
    model_cfg.update({"hyper_params": model_params})
    model = ppsci.arch.MoFlowNet(**model_cfg)
    ppsci.utils.save_load.load_pretrain(model, path=cfg.TRAIN.pretrained_model_path)

    model_prop_cfg = dict(cfg.MODEL_Prop)
    model_prop_cfg.update(
        {
            "model": model,
            "hidden_size": hidden,
        }
    )
    property_model = ppsci.arch.MoFlowProp(**model_prop_cfg)
    train = ppsci.data.dataset.build_dataset(dataloader_cfg["dataset"])
    train_dataloader = ppsci.data.build_dataloader(train, dataloader_cfg)
    train_idx = train.train_idx
    property_model_path = osp.join(
        cfg.output_dir, "{}_model.pdparams".format(cfg.OPTIMIZE.property_name)
    )

    if not osp.exists(property_model_path):
        logger.message("Training regression model over molecular embedding:")
        property_csv_path = osp.join(
            cfg.FILE_PATH, "{}_property.csv".format(cfg.data_name)
        )
        prop_list = load_property_csv(property_csv_path, normalize=True)
        train_prop = [prop_list[i] for i in train_idx]
        # test_prop = [prop_list[i] for i in valid_idx]

        N = len(train)
        property_model = fit_model(
            property_model,
            train_dataloader,
            train_prop,
            N,
            property_name=cfg.OPTIMIZE.property_name,
            max_epochs=cfg.OPTIMIZE.max_epochs,
            learning_rate=cfg.OPTIMIZE.learning_rate,
            weight_decay=cfg.OPTIMIZE.weight_decay,
        )
        logger.message(
            "saving {} regression model to: {}".format(
                cfg.OPTIMIZE.property_name, property_model_path
            )
        )
        paddle.save(obj=property_model.state_dict(), path=property_model_path)

    else:
        logger.message("Loading trained regression model for optimization")
        property_csv_path = osp.join(
            cfg.FILE_PATH, "{}_property.csv".format(cfg.data_name)
        )
        prop_list = load_property_csv(property_csv_path, normalize=True)
        train_prop = [prop_list[i] for i in train_idx]
        # test_prop = [prop_list[i] for i in valid_idx]

        logger.message(
            "loading {} regression model from: {}".format(
                cfg.OPTIMIZE.property_name, property_model_path
            )
        )

        state_dict = paddle.load(path=property_model_path)
        property_model.set_state_dict(state_dict)
        property_model.eval()
        model.eval()
        if cfg.OPTIMIZE.topscore:
            logger.message("Finding top score:")
            find_top_score_smiles(
                model,
                property_model,
                cfg.data_name,
                cfg.OPTIMIZE.property_name,
                train_prop,
                cfg.OPTIMIZE.topk,
                atomic_num_list,
                cfg.OPTIMIZE.debug,
                cfg.output_dir,
            )
        if cfg.OPTIMIZE.consopt:
            logger.message("Constrained optimization:")
            constrain_optimization_smiles(
                model,
                property_model,
                cfg.data_name,
                cfg.OPTIMIZE.property_name,
                train_prop,
                cfg.OPTIMIZE.topk,
                atomic_num_list,
                cfg.OPTIMIZE.debug,
                cfg.output_dir,
                sim_cutoff=cfg.OPTIMIZE.sim_cutoff,
            )


@hydra.main(version_base=None, config_path="./conf", config_name="moflow_optimize.yaml")
def main(cfg: DictConfig):
    optimize(cfg)


if __name__ == "__main__":
    main()
