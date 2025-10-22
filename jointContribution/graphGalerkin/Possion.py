import sys

import matplotlib.pyplot as plt
import numpy as np
import paddle

paddle.seed(1334)

sys.path.insert(0, "pycamotk")
from pyCaMOtk.create_dbc_strct import create_dbc_strct
from pyCaMOtk.create_femsp_cg import create_femsp_cg
from pyCaMOtk.create_mesh_hsphere import mesh_hsphere
from pyCaMOtk.visualize_fem import visualize_fem

sys.path.insert(0, "source")
import setup_prob_eqn_handcode
from FEM_ForwardModel import analyticalPossion
from GCNNModel import PossionNet
from GCNNModel import e2vcg2connectivity
from TensorFEMCore import Double
from TensorFEMCore import create_fem_resjac
from TensorFEMCore import solve_fem_GCNN

sys.path.insert(0, "utils")
from utils import Data


class Possion:
    def __init__(self) -> None:
        # GCNN model
        self.model = PossionNet()

    def params_possion(self):
        """
        e2vcg is a 2D array (NNODE PER ELEM, NELEM): The connectivity of the
        mesh. The (:, e) entries are the global node numbers of the nodes
        that comprise element e. The local node numbers of each element are
        defined by the columns of this matrix, e.g., e2vcg(i, e) is the
        global node number of the ith local node of element e.
        The flux constant Flux=[du/dx, du/dy]^T=K dot [dphi/dx,dphi/dy]
                where phi is the solution polynomial function
        """

        # Set up GCNN-FEM Possion problem
        self.nin = 1  # Number of input variable
        self.nvar = 1  # Number of primanry variable
        etype = "hcube"  # Mesh type
        c = [0, 0]  # Domain center
        r = 1  # Radius
        self.porder = 2  # Polynomial order for solution and geometry basis
        nel = [2, 2]  # Number of element in x and y axis
        self.msh = mesh_hsphere(
            etype, c, r, nel, self.porder
        ).getmsh()  # Create mesh object
        self.xcg = self.msh.xcg  # Extract node coordinates
        self.ndof = self.xcg.shape[1]
        self.e2vcg = self.msh.e2vcg  # Extract element connectivity
        self.connectivity = e2vcg2connectivity(self.msh.e2vcg, "ele")

        self.bnd2nbc = np.asarray([0])  # Define the boundary tag!
        self.K = lambda x, el: np.asarray([[1], [0], [0], [1]])
        self.Qb = (
            lambda x, n, bnd, el, fc: 0
        )  # The primary variable value on the boundary
        dbc_idx = [
            i
            for i in range(self.xcg.shape[1])
            if np.sum(self.xcg[:, i] ** 2) > 1 - 1e-12
        ]  # The boundary node id
        self.dbc_idx = np.asarray(dbc_idx)
        self.dbc_val = dbc_idx * 0  # The boundary node primary variable value

    def train(self):
        paddle.device.set_device("gpu:0")
        dbc = create_dbc_strct(
            self.xcg.shape[1] * self.nvar, self.dbc_idx, self.dbc_val
        )  # Create the class of boundary condition

        Src_new = self.model.source
        K_new = paddle.to_tensor([[1], [0], [0], [1]], dtype="float32").reshape((4,))
        parsfuncI = lambda x: paddle.concat((K_new, Src_new), axis=0)
        S = [2]  # Parametrize the source value in the pde -F_ij,j=S_i
        LossF = []
        # Define the Training Data
        Graph = []
        ii = 0

        for i in S:
            f = lambda x, el: i
            prob = setup_prob_eqn_handcode.setup_linelptc_sclr_base_handcode(
                2, self.K, f, self.Qb, self.bnd2nbc
            )

            femsp = create_femsp_cg(
                prob, self.msh, self.porder, self.e2vcg, self.porder, self.e2vcg, dbc
            )
            fcn = lambda u_: create_fem_resjac(
                "cg",
                u_,
                self.msh.transfdatacontiguous,
                femsp.elem,
                femsp.elem_data,
                femsp.ldof2gdof_eqn.ldof2gdof,
                femsp.ldof2gdof_var.ldof2gdof,
                self.msh.e2e,
                femsp.spmat,
                dbc,
                [i for i in range(self.ndof) if i not in self.dbc_idx],
                parsfuncI,
                None,
                self.model,
            )
            LossF.append(fcn)

            Ue = Double(analyticalPossion(self.xcg, i).flatten().reshape(self.ndof, 1))
            fcn_id = Double(np.asarray([ii]))
            Ue_aug = paddle.concat((fcn_id, Ue), axis=0)
            Uin = Double(self.xcg.T)
            graph = Data(x=Uin, y=Ue_aug, edge_index=self.connectivity)
            Graph.append(graph)
            ii = ii + 1
        DataList = [[Graph[i]] for i in range(len(S))]
        TrainDataloader = DataList

        # Training Data
        [model, info] = solve_fem_GCNN(
            TrainDataloader, LossF, self.model, self.tol, self.maxit
        )
        print("K=", self.K)
        print("Min Error=", info["Er"].min())
        print("Mean Error Last 10 iterations=", np.mean(info["Er"][-10:]))
        print("Var  Error Last 10 iterations=", np.var(info["Er"][-10:]))

        np.savetxt("demo0\ErFinal.txt", info["Er"])
        np.savetxt("demo0\Loss.txt", info["Loss"])

        solution = model(Graph[0])
        solution[dbc.dbc_idx] = Double(dbc.dbc_val.reshape([len(dbc.dbc_val), 1]))
        solution = solution.detach().cpu().numpy()
        Ue = Ue.detach().cpu().numpy()
        return solution, Ue

    def plot_disk(self, solution, Ue):
        ax1 = plt.subplot(1, 1, 1)
        _, cbar1 = visualize_fem(
            ax1, self.msh, solution[self.e2vcg], {"plot_elem": True, "nref": 6}, []
        )
        ax1.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            top=False,
            labelbottom=False,
            labelleft=False,
        )
        ax1.axis("off")
        cbar1.remove()
        plt.margins(0, 0)
        plt.savefig(
            "gcnn_possion_circle.png", bbox_inches="tight", pad_inches=-0.11, dpi=800
        )
        plt.close()

        ax2 = plt.subplot(1, 1, 1)
        _, cbar2 = visualize_fem(
            ax2, self.msh, Ue[self.e2vcg], {"plot_elem": True, "nref": 6}, []
        )
        ax2.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            top=False,
            labelbottom=False,
            labelleft=False,
        )
        ax2.axis("off")
        cbar2.remove()
        plt.margins(0, 0)
        plt.savefig(
            "exact_possion_circle.png", bbox_inches="tight", pad_inches=-0.11, dpi=800
        )

    def plot_circle(self, solution, Ue):
        fig = plt.figure()
        ax1 = plt.subplot(1, 2, 1)
        visualize_fem(
            ax1, self.msh, solution[self.e2vcg], {"plot_elem": True, "nref": 6}, []
        )
        ax1.set_title("GCNN solution")
        ax2 = plt.subplot(1, 2, 2)
        visualize_fem(ax2, self.msh, Ue[self.e2vcg], {"plot_elem": True, "nref": 6}, [])
        ax2.set_title("Exact solution")
        fig.tight_layout(pad=3.0)
        plt.savefig("demo0\Demo.pdf", bbox_inches="tight")

    def disk_possion_hard(self):
        # Hyper prameters
        self.tol = 1.0e-16
        self.maxit = 3000
        self.params_possion()
        Ufem = analyticalPossion(self.xcg, 2).flatten().reshape(self.ndof, 1)

        obsidx = np.asarray([8])
        self.dbc_idx = np.hstack((np.asarray(self.dbc_idx), obsidx))
        self.dbc_val = Ufem[self.dbc_idx]

        solution, Ue = self.train()
        self.plot_disk(solution, Ue)

    def circle(self):
        # Hyper prameters
        self.tol = 1.0e-16
        self.maxit = 500
        self.params_possion()
        self.dbc_idx = np.asarray(self.dbc_idx)
        self.dbc_val = self.dbc_idx * 0  # The boundary node primary variable value

        solution, Ue = self.train()
        self.plot_circle(solution, Ue)


if __name__ == "__main__":
    possion_obj = Possion()
    possion_obj.disk_possion_hard()
    possion_obj.circle()
