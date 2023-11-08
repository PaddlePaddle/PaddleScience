import sys

import matplotlib.pyplot as plt
import numpy as np
import paddle
from scipy.io import loadmat

sys.path.insert(0, "pycamotk")
from pyCaMOtk.create_dbc_strct import create_dbc_strct
from pyCaMOtk.create_fem_resjac import create_fem_resjac
from pyCaMOtk.create_femsp_cg import create_femsp_cg
from pyCaMOtk.create_mesh_hcube import mesh_hcube
from pyCaMOtk.geom_mltdim import Hypercube
from pyCaMOtk.geom_mltdim import Simplex
from pyCaMOtk.LinearElasticityHandCode import *
from pyCaMOtk.mesh import Mesh
from pyCaMOtk.mesh import get_gdof_from_bndtag
from pyCaMOtk.solve_fem import solve_fem
from pyCaMOtk.visualize_fem import visualize_fem

sys.path.insert(0, "source")
import setup_prob_eqn_handcode
import TensorFEMCore
from GCNNModel import LinearElasticityNet2D
from GCNNModel import e2vcg2connectivity
from TensorFEMCore import Double
from TensorFEMCore import ReshapeFix
from TensorFEMCore import solve_fem_GCNN

sys.path.insert(0, "utils")
from utils import Data

paddle.seed(0)


class LinearElasticity:
    def __init__(self) -> None:
        # GCNN model
        self.model = LinearElasticityNet2D()

    def train(
        self,
        Ufem,
        ndof,
        xcg,
        connectivity,
        LossF,
        tol,
        maxit,
        dbc,
        ndim,
        nnode,
        etype,
        e2vcg,
        e2bnd,
    ):
        ii = 0
        Graph = []
        Ue = Double(Ufem.flatten().reshape(ndof, 1))
        fcn_id = Double(np.asarray([ii]))
        Ue_aug = paddle.concat((fcn_id, Ue), axis=0)
        xcg_gcnn = np.zeros((2, 2 * xcg.shape[1]))
        for i in range(xcg.shape[1]):
            xcg_gcnn[:, 2 * i] = xcg[:, i]
            xcg_gcnn[:, 2 * i + 1] = xcg[:, i]
        Uin = Double(xcg_gcnn.T)
        graph = Data(x=Uin, y=Ue_aug, edge_index=connectivity)
        Graph.append(graph)
        DataList = [[Graph[0]]]
        TrainDataloader = DataList
        [self.model, info] = solve_fem_GCNN(
            TrainDataloader, LossF, self.model, tol, maxit
        )
        np.save("modelCircleDet.npy", info)
        solution = self.model(Graph[0].to("cuda"))
        solution = ReshapeFix(paddle.clone(solution), [len(solution.flatten()), 1], "C")
        solution[dbc.dbc_idx] = Double(dbc.dbc_val.reshape([len(dbc.dbc_val), 1]))
        solution = solution.detach().cpu().numpy()
        xcg_defGCNN = xcg + np.reshape(solution, [ndim, nnode], order="F")
        msh_defGCNN = Mesh(etype, xcg_defGCNN, e2vcg, e2bnd, ndim)
        uabsGCNN = np.sqrt(
            solution[[i for i in range(ndof) if i % 2 == 0]] ** 2
            + solution[[i for i in range(ndof) if i % 2 != 0]] ** 2
        )
        return msh_defGCNN, uabsGCNN

    def plot_hard_way(self, msh_defGCNN, uabsGCNN, e2vcg, msh_def, uabs):
        fig = plt.figure()
        ax1 = plt.subplot(1, 2, 1)
        visualize_fem(
            ax1, msh_defGCNN, uabsGCNN[e2vcg], {"plot_elem": False, "nref": 1}, []
        )
        ax1.set_title("GCNN solution")
        ax2 = plt.subplot(1, 2, 2)
        visualize_fem(ax2, msh_def, uabs[e2vcg], {"plot_elem": False, "nref": 1}, [])
        ax2.set_title("FEM solution")
        fig.tight_layout(pad=3.0)
        plt.savefig("GCNN.pdf", bbox_inches="tight")

    def plot_square(self, msh_defGCNN, uabsGCNN, e2vcg, msh_def, uabs):
        plt.figure()
        ax1 = plt.subplot(1, 1, 1)
        _, cbar1 = visualize_fem(
            ax1, msh_defGCNN, uabsGCNN[e2vcg], {"plot_elem": False, "nref": 4}, []
        )
        ax1.axis("off")
        cbar1.remove()
        plt.margins(0, 0)
        plt.savefig(
            "gcnn_2dlinearelasticity_square.png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=800,
        )

        plt.figure()
        ax2 = plt.subplot(1, 1, 1)
        _, cbar2 = visualize_fem(
            ax2, msh_def, uabs[e2vcg], {"plot_elem": False, "nref": 4}, []
        )
        ax2.axis("off")
        cbar2.remove()
        plt.margins(0, 0)
        plt.savefig(
            "fem_2dlinearelasticity_square.png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=800,
        )

    def hard_way(self):
        # FEM
        etype = "simplex"
        ndim = 2
        dat = loadmat("./msh/cylshk0a-simp-nref0p1.mat")
        xcg = dat["xcg"] / 10
        e2vcg = dat["e2vcg"] - 1
        e2bnd = dat["e2bnd"] - 1
        msh = Mesh(etype, xcg, e2vcg, e2bnd, ndim)
        xcg = msh.xcg
        e2vcg = msh.e2vcg
        e2bnd = msh.e2bnd
        porder = msh.porder
        [ndim, nnode] = xcg.shape
        nvar = ndim
        ndof = nnode * nvar

        lam = lambda x, el: 1
        mu = lambda x, el: 1
        f = lambda x, el: np.zeros([ndim, 1])
        bnd2nbc = [0.0, 1.0, 2.0, 3.0, 4.0]
        tb = lambda x, n, bnd, el, fc: np.asarray([[2], [0]]) * (
            bnd == 2 or bnd == 2.0 or (bnd - 2) ** 2 < 1e-8
        ) + np.asarray([[0], [0]])
        prob = setup_linelast_base_handcode(ndim, lam, mu, f, tb, bnd2nbc)
        # Create finite element space
        femsp = create_femsp_cg(prob, msh, porder, e2vcg, porder, e2vcg)
        ldof2gdof = femsp.ldof2gdof_var.ldof2gdof
        geo = Simplex(ndim, porder)
        f2v = geo.f2n
        dbc_idx = get_gdof_from_bndtag(
            [i for i in range(ndim)], [0], nvar, ldof2gdof, e2bnd, f2v
        )
        dbc_idx.sort()
        dbc_idx = np.asarray(dbc_idx)
        dbc_val = 0 * dbc_idx
        dbc = create_dbc_strct(ndof, dbc_idx, dbc_val)
        femsp.dbc = dbc
        tol = 1.0e-8
        maxit = 100000
        [Ufem, info] = solve_fem(
            "cg",
            msh.transfdatacontiguous,
            femsp.elem,
            femsp.elem_data,
            femsp.ldof2gdof_eqn.ldof2gdof,
            femsp.ldof2gdof_var.ldof2gdof,
            msh.e2e,
            femsp.spmat,
            dbc,
            None,
            tol,
            maxit,
        )

        xcg_def = xcg + np.reshape(Ufem, [ndim, nnode], order="F")
        msh_def = Mesh(etype, xcg_def, e2vcg, e2bnd, ndim)
        uabs = np.sqrt(
            Ufem[[i for i in range(ndof) if i % 2 == 0]] ** 2
            + Ufem[[i for i in range(ndof) if i % 2 != 0]] ** 2
        )
        fig = plt.figure()
        ax1 = plt.subplot(1, 1, 1)
        visualize_fem(ax1, msh_def, uabs[e2vcg], {"plot_elem": False, "nref": 1}, [])
        ax1.set_title("FEM solution")
        fig.tight_layout(pad=3.0)

        idx_xcg = [
            i
            for i in range(xcg.shape[1])
            if 2 * i not in dbc_idx and 2 * i + 1 not in dbc_idx
        ]

        obsidx = np.asarray([5, 11, 26, 32, 38])  # max is 9

        idx_whole = []
        for i in obsidx:
            idx_whole.append(2 * i)
            idx_whole.append(2 * i + 1)
        obsxcg = msh_def.xcg[:, obsidx]
        ax1.plot(obsxcg[0, :], obsxcg[1, :], "o")

        dbc_idx_new = np.hstack((dbc_idx, idx_whole))
        dbc_val_new = Ufem[dbc_idx_new]
        dbc = create_dbc_strct(msh.xcg.shape[1] * nvar, dbc_idx_new, dbc_val_new)

        Src_new = self.model.source
        K_new = paddle.to_tensor([[0], [0]], dtype="float32").reshape((2,))
        parsfuncI = lambda x: paddle.concat((Src_new[0:1], Src_new[1:2], K_new), axis=0)
        # GCNN
        connectivity = e2vcg2connectivity(e2vcg, "ele")
        prob = setup_prob_eqn_handcode.setup_linelast_base_handcode(
            ndim, lam, mu, f, tb, bnd2nbc
        )
        femsp_gcnn = create_femsp_cg(prob, msh, porder, e2vcg, porder, e2vcg, dbc)
        LossF = []
        fcn = lambda u_: TensorFEMCore.create_fem_resjac(
            "cg",
            u_,
            msh.transfdatacontiguous,
            femsp_gcnn.elem,
            femsp_gcnn.elem_data,
            femsp_gcnn.ldof2gdof_eqn.ldof2gdof,
            femsp_gcnn.ldof2gdof_var.ldof2gdof,
            msh.e2e,
            femsp_gcnn.spmat,
            dbc,
            [i for i in range(ndof) if i not in dbc_idx],
            parsfuncI,
            None,
        )
        LossF.append(fcn)
        msh_defGCNN, uabsGCNN = self.train(
            Ufem,
            ndof,
            xcg,
            connectivity,
            LossF,
            tol,
            maxit,
            dbc,
            ndim,
            nnode,
            etype,
            e2vcg,
            e2bnd,
        )
        self.plot_hard_way(msh_defGCNN, uabsGCNN, e2vcg, msh_def, uabs)

    def main_square(self):
        # FEM
        nvar = 2
        etype = "hcube"
        lims = np.asarray([[0, 1], [0, 1]])
        nel = [2, 2]
        porder = 2
        nf = 4
        msh = mesh_hcube(etype, lims, nel, porder).getmsh()
        xcg = msh.xcg
        e2vcg = msh.e2vcg
        e2bnd = msh.e2bnd
        porder = msh.porder
        [ndim, nnode] = xcg.shape
        nvar = ndim
        ndof = nnode * nvar

        lam = lambda x, el: 1
        mu = lambda x, el: 1
        f = lambda x, el: np.zeros([ndim, 1])
        bnd2nbc = np.asarray([0, 1, 2, 3])
        tb = lambda x, n, bnd, el, fc: np.asarray([[0.5], [0]]) * (
            (bnd - 2) ** 2 < 1e-8
        ) + np.asarray([[0], [0]])
        prob = setup_linelast_base_handcode(ndim, lam, mu, f, tb, bnd2nbc)
        # Create finite element space
        femsp = create_femsp_cg(prob, msh, porder, e2vcg, porder, e2vcg)
        ldof2gdof = femsp.ldof2gdof_var.ldof2gdof
        geo = Hypercube(ndim, porder)
        f2v = geo.f2n
        dbc_idx = get_gdof_from_bndtag(
            [i for i in range(ndim)], [0], nvar, ldof2gdof, e2bnd, f2v
        )
        dbc_idx.sort()
        dbc_idx = np.asarray(dbc_idx)
        dbc_val = 0 * dbc_idx
        dbc = create_dbc_strct(ndof, dbc_idx, dbc_val)
        femsp.dbc = dbc
        tol = 1.0e-8
        maxit = 4500

        [Ufem, info] = solve_fem(
            "cg",
            msh.transfdatacontiguous,
            femsp.elem,
            femsp.elem_data,
            femsp.ldof2gdof_eqn.ldof2gdof,
            femsp.ldof2gdof_var.ldof2gdof,
            msh.e2e,
            femsp.spmat,
            dbc,
            None,
            tol,
            maxit,
        )

        xcg_def = xcg + np.reshape(Ufem, [ndim, nnode], order="F")
        msh_def = Mesh(etype, xcg_def, e2vcg, e2bnd, ndim)
        uabs = np.sqrt(
            Ufem[[i for i in range(ndof) if i % 2 == 0]] ** 2
            + Ufem[[i for i in range(ndof) if i % 2 != 0]] ** 2
        )
        # GCNN
        connectivity = e2vcg2connectivity(e2vcg, "ele")
        prob = setup_prob_eqn_handcode.setup_linelast_base_handcode(
            ndim, lam, mu, f, tb, bnd2nbc
        )
        femsp_gcnn = create_femsp_cg(prob, msh, porder, e2vcg, porder, e2vcg, dbc)
        LossF = []
        fcn = lambda u_: TensorFEMCore.create_fem_resjac(
            "cg",
            u_,
            msh.transfdatacontiguous,
            femsp_gcnn.elem,
            femsp_gcnn.elem_data,
            femsp_gcnn.ldof2gdof_eqn.ldof2gdof,
            femsp_gcnn.ldof2gdof_var.ldof2gdof,
            msh.e2e,
            femsp_gcnn.spmat,
            dbc,
        )
        fcn_fem = lambda u_: create_fem_resjac(
            "cg",
            u_,
            msh.transfdatacontiguous,
            femsp.elem,
            femsp.elem_data,
            femsp.ldof2gdof_eqn.ldof2gdof,
            femsp.ldof2gdof_var.ldof2gdof,
            msh.e2e,
            femsp.spmat,
            dbc,
        )
        LossF.append(fcn)
        msh_defGCNN, uabsGCNN = self.train(
            Ufem,
            ndof,
            xcg,
            connectivity,
            LossF,
            tol,
            maxit,
            dbc,
            ndim,
            nnode,
            etype,
            e2vcg,
            e2bnd,
        )
        self.plot_square(msh_defGCNN, uabsGCNN, e2vcg, msh_def, uabs)


if __name__ == "__main__":
    le_obj = LinearElasticity()
    le_obj.hard_way()
    le_obj.main_square()
