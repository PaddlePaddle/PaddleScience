import numpy as np
import scipy.io as io

np.random.seed(1234)


class DataSet:
    def __init__(self, num, bs, modes):
        self.num = num
        self.bs = bs
        self.modes = modes
        (
            self.F_train,
            self.U_train,
            self.V_train,
            self.rRe,
            self.F_test,
            self.U_test,
            self.V_test,
            self.x_train,
            self.x_eq_train,
            self.x_test,
            self.u_mean,
            self.u_std,
            self.v_mean,
            self.v_std,
            self.u_basis,
            self.v_basis,
            self.lam_u,
            self.lam_v,
        ) = self.load_data()

    def PODbasis(self):
        s = 65
        num_res = s * s
        u_basis_out = np.reshape(self.u_basis.T, (-1, num_res, 1))
        v_basis_out = np.reshape(self.v_basis.T, (-1, num_res, 1))
        u_basis_out, v_basis_out = self.decoder(u_basis_out, v_basis_out)
        u_basis_out = u_basis_out - self.u_mean
        v_basis_out = v_basis_out - self.v_mean
        u_basis_out = np.reshape(u_basis_out, (-1, s, s))
        v_basis_out = np.reshape(v_basis_out, (-1, s, s))
        save_dict = {
            "u_basis": u_basis_out,
            "v_basis": v_basis_out,
            "lam_u": self.lam_u,
            "lam_v": self.lam_v,
        }
        io.savemat("./Output/basis.mat", save_dict)
        return self.u_basis, self.v_basis

    def samples(self):
        """
        num_train = 40000
        num_test = 10000
        data = io.loadmat('./Data/Data')
        F = data['F']
        U = data['U']
        """

        num_train = 1
        x = np.linspace(-1, 1, self.num)
        y = np.linspace(-1, 1, self.num)
        xx, yy = np.meshgrid(x, y)
        xx = np.reshape(xx, (-1, 1))
        yy = np.reshape(yy, (-1, 1))
        x_train = np.hstack((xx, yy))
        F, U = self.func(x_train)

        Num = self.num * self.num

        F = np.reshape(F, (-1, self.num, self.num, 1))
        U = np.reshape(U, (-1, Num, 1))
        F_train = F[:num_train, :, :]
        U_train = U[:num_train, :, :]
        F_test = F[num_train:, :, :]
        U_test = U[num_train:, :, :]
        return F_train, U_train, F_test, U_test

    def decoder(self, u, v):
        u = u * self.u_std + self.u_mean
        v = v * self.v_std + self.v_mean
        return u, v

    def load_data(self):
        data_train = io.loadmat("./Data/Cavity_Flow")
        data_test = io.loadmat("./Data/Cavity_Flow_Test")

        num_Re = 100
        Re = np.linspace(100, 2080, num_Re).reshape((-1, 1, 1))
        Re = Re.astype(np.float32)
        rRe = 1.0 / Re

        a_train = data_train["u_bc"].astype(np.float32)
        u_train = data_train["u_data"].astype(np.float32)
        v_train = data_train["v_data"].astype(np.float32)

        a_test = data_test["u_bc"].astype(np.float32)
        u_test = data_test["u_data"].astype(np.float32)
        v_test = data_test["v_data"].astype(np.float32)

        xx = data_train["x_2d"].astype(np.float32)
        yy = data_train["y_2d"].astype(np.float32)

        res_step = 4
        u_train = u_train[:, ::res_step, ::res_step]
        v_train = v_train[:, ::res_step, ::res_step]
        xx_train = xx[::res_step, ::res_step]
        yy_train = yy[::res_step, ::res_step]

        xx_eq = xx[1:64, 1:64]
        yy_eq = yy[1:64, 1:64]
        xx_eq = np.reshape(xx_eq, (-1, 1))
        yy_eq = np.reshape(yy_eq, (-1, 1))
        x_eq_train = np.hstack((xx_eq, yy_eq))

        xx_train = np.reshape(xx_train, (-1, 1))
        yy_train = np.reshape(yy_train, (-1, 1))
        x_train = np.hstack((xx_train, yy_train))

        xx_test = np.reshape(xx, (-1, 1))
        yy_test = np.reshape(yy, (-1, 1))
        x_test = np.hstack((xx_test, yy_test))

        """
        perm = np.random.permutation(a.shape[0])
        a = a[perm, :]
        u = u[perm, :, :]
        v = v[perm, :, :]
        """

        num_train = 100
        s_f = 65
        s = u_train.shape[1]
        num_res = s * s

        u_train_mean = np.mean(np.reshape(u_train, (-1, s, s)), 0)
        u_train_std = np.std(np.reshape(u_train, (-1, s, s)), 0)
        v_train_mean = np.mean(np.reshape(v_train, (-1, s, s)), 0)
        v_train_std = np.std(np.reshape(v_train, (-1, s, s)), 0)

        u_train_mean = np.reshape(u_train_mean, (-1, num_res, 1))
        u_train_std = np.reshape(u_train_std, (-1, num_res, 1))
        v_train_mean = np.reshape(v_train_mean, (-1, num_res, 1))
        v_train_std = np.reshape(v_train_std, (-1, num_res, 1))

        F_train = np.reshape(a_train, (-1, s_f))
        U_train = np.reshape(u_train, (-1, num_res, 1))
        V_train = np.reshape(v_train, (-1, num_res, 1))

        # F_train = (F_train - f_train_mean)/(f_train_std + 1.0e-9)
        # U_train = (U_train - u_train_mean)/(u_train_std + 1.0e-9)
        # V_train = (V_train - v_train_mean)/(v_train_std + 1.0e-9)

        # F_test = (F_test - f_train_mean)/(f_train_std + 1.0e-9)

        U = np.reshape(U_train, (-1, num_res))
        V = np.reshape(V_train, (-1, num_res))
        C_u = 1.0 / (num_train - 1) * np.matmul(U.T, U)
        C_v = 1.0 / (num_train - 1) * np.matmul(V.T, V)
        lam_u, phi_u = np.linalg.eigh(C_u)
        lam_v, phi_v = np.linalg.eigh(C_v)

        lam_u = np.flip(lam_u)
        phi_u = np.fliplr(phi_u)
        lam_v = np.flip(lam_v)
        phi_v = np.fliplr(phi_v)

        u_cumsum = np.cumsum(lam_u)
        v_cumsum = np.cumsum(lam_v)
        u_per = u_cumsum[self.modes - 1] / u_cumsum[-1]
        v_per = v_cumsum[self.modes - 1] / v_cumsum[-1]

        u_basis = phi_u[:, : self.modes]
        v_basis = phi_v[:, : self.modes]

        print("Kept Energy: u: %.3f, v: %.3f" % (u_per, v_per))

        num_res = s_f * s_f
        F_test = np.reshape(a_test, (-1, s_f))
        U_test = np.reshape(u_test, (-1, num_res, 1))
        V_test = np.reshape(v_test, (-1, num_res, 1))
        """
        plt.plot(lam_u[:self.modes], 'k-')
        plt.plot(lam_v[:self.modes], 'r--')
        plt.show()
        """

        """
        F_train = np.reshape(f_train, (-1, s))
        U_train = np.reshape(u_train, (-1, num_res, 1))
        V_train = np.reshape(v_train, (-1, num_res, 1))

        F_test = np.reshape(f_test, (-1, s))
        U_test = np.reshape(u_test, (-1, num_res, 1))
        V_test = np.reshape(v_test, (-1, num_res, 1))
        """

        """
        U_ref = np.reshape(U_test, (U_test.shape[0], U_test.shape[1]))
        np.savetxt('./Output/u_ref', U_ref, fmt='%e')
        """

        return (
            F_train,
            U_train,
            V_train,
            rRe,
            F_test,
            U_test,
            V_test,
            x_train,
            x_eq_train,
            x_test,
            u_train_mean,
            u_train_std,
            v_train_mean,
            v_train_std,
            u_basis,
            v_basis,
            lam_u,
            lam_v,
        )

    def minibatch(self):
        batch_id = np.random.choice(self.F_train.shape[0], self.bs, replace=False)
        f_train = [self.F_train[i : i + 1] for i in batch_id]
        f_train = np.concatenate(f_train, axis=0)
        u_train = [self.U_train[i : i + 1] for i in batch_id]
        u_train = np.concatenate(u_train, axis=0)
        v_train = [self.V_train[i : i + 1] for i in batch_id]
        v_train = np.concatenate(v_train, axis=0)
        rRe_train = [self.rRe[i : i + 1] for i in batch_id]
        rRe_train = np.concatenate(rRe_train, axis=0)

        """
        x = np.linspace(0., 1, self.num)
        y = np.linspace(0., 1, self.num)
        xx, yy = np.meshgrid(x, y)
        xx = np.reshape(xx, (-1, 1))
        yy = np.reshape(yy, (-1, 1))
        x_train = np.hstack((xx, yy))
        """

        Xmin = np.array([0.0, 0.0]).reshape((-1, 2))
        Xmax = np.array([1.0, 1.0]).reshape((-1, 2))
        # x_train = np.linspace(-1, 1, self.N).reshape((-1, 1))

        return (
            self.x_train,
            self.x_eq_train,
            f_train,
            rRe_train,
            u_train,
            v_train,
            Xmin,
            Xmax,
        )

    def testbatch(self):
        """
        batch_id = np.random.choice(self.F_test.shape[0], num_test, replace=False)
        f_test = [self.F_test[i:i+1] for i in batch_id]
        f_test = np.concatenate(f_test, axis=0)
        u_test = [self.U_test[i:i+1] for i in batch_id]
        u_test = np.concatenate(u_test, axis=0)
        v_test = [self.V_test[i:i+1] for i in batch_id]
        v_test = np.concatenate(v_test, axis=0)
        batch_id = np.reshape(batch_id, (-1, 1))
        """

        x_test = self.x_test
        f_test, u_test, v_test = self.F_test, self.U_test, self.V_test

        return x_test, f_test, u_test, v_test
