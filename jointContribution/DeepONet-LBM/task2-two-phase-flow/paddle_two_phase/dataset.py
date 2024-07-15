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
            self.F_test,
            self.U_test,
            self.x_train,
            self.u_basis,
            self.lam_u,
        ) = self.load_data()
        """
        self.F_train, self.U_train, self.V_train, \
        self.F_test, self.U_test, self.V_test, \
        self.x_train, \
        self.u_mean, self.u_std, \
        self.v_mean, self.v_std, \
        self.u_basis, self.v_basis, \
        self.lam_u, self.lam_v = self.load_data()
        """

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
        data_train = io.loadmat("./Data/Two_Phase_Flow_Training")
        data_test = io.loadmat("./Data/Two_Phase_Flow_Test")

        step = 4
        num_yb = 100
        num_ye = 900
        num_xb = 0
        num_xe = 256

        """
        a_train = data_train['u_bc'].astype(np.float32)
        u_train = data_train['u_data'].astype(np.float32)
        v_train = data_train['v_data'].astype(np.float32)

        a_test = data_test['u_bc'].astype(np.float32)
        u_test = data_test['u_data'].astype(np.float32)
        v_test = data_test['v_data'].astype(np.float32)
        """
        num = 30
        in_dim = 1
        x = np.linspace(0, 1, num).reshape((1, -1))
        x = x.astype(np.float32)
        a_train = np.arange(20.0, 1000.0, 20.0, dtype=np.float32).reshape((-1, 1))
        # a_train = np.matmul(0.01*a_train, loc)
        u_train = data_train["phi_data"]
        u_train = u_train[:, num_yb:num_ye:step, num_xb:num_xe:step]
        u_train = u_train.astype(np.float32)
        print(u_train.shape)

        # a_test = np.array([110.], dtype=np.float32).reshape((-1, 1))
        a_test = np.arange(55.0, 1000.0, 100.0, dtype=np.float32).reshape((-1, 1))
        print(a_test)
        # a_test = np.matmul(0.01*a_test, loc)
        u_test = data_test["phi_data"]
        # a_test = np.array([100.], dtype=np.float32).reshape((-1, 1))
        # u_test = u_train[4, :, :]
        u_test = u_test[:, num_yb:num_ye:step, num_xb:num_xe:step]
        u_test = u_test.astype(np.float32)
        print(u_test.shape)

        xx = data_train["x_2d"]
        yy = data_train["y_2d"]

        xx = np.reshape(xx, (-1, 1))
        yy = np.reshape(yy, (-1, 1))
        x_train = np.hstack((xx, yy))

        """
        perm = np.random.permutation(a.shape[0])
        a = a[perm, :]
        u = u[perm, :, :]
        v = v[perm, :, :]
        """

        """
        num_train = 100
        num_test = 10
        s = 65
        """
        num_res = u_train.shape[1] * u_train.shape[2]

        F_train = np.reshape(a_train, (-1, in_dim))
        U_train = np.reshape(u_train, (-1, num_res, 1))

        F_test = np.reshape(a_test, (-1, in_dim))
        U_test = np.reshape(u_test, (-1, num_res, 1))

        """
        U = np.reshape(U_train, (-1, num_res))
        C_u = 1./(num_train-1)*np.matmul(U.T, U)
        lam_u, phi_u = np.linalg.eigh(C_u)

        lam_u = np.flip(lam_u)
        phi_u = np.fliplr(phi_u)


        u_cumsum = np.cumsum(lam_u)
        u_per = u_cumsum[self.modes-1]/u_cumsum[-1]
        """

        data_basis = io.loadmat("./Output/basis")
        phi_u = data_basis["phi_basis"]
        u_basis = phi_u[:, : self.modes]
        lam_u = data_basis["lam_u"]
        u_cumsum = np.cumsum(lam_u)
        u_per = u_cumsum[self.modes - 1] / u_cumsum[-1]

        print("Kept Energy: u: %.5f" % (u_per))

        save_dict = {"phi_basis": phi_u, "lam_u": lam_u}
        io.savemat("./Output/basis.mat", save_dict)
        """
        f_train_mean = np.mean(np.reshape(a_train, (-1, s)), 0)
        f_train_std = np.std(np.reshape(a_train, (-1, s)), 0)
        u_train_mean = np.mean(np.reshape(u_train, (-1, s, s)), 0)
        u_train_std = np.std(np.reshape(u_train, (-1, s, s)), 0)
        v_train_mean = np.mean(np.reshape(v_train, (-1, s, s)), 0)
        v_train_std = np.std(np.reshape(v_train, (-1, s, s)), 0)

        u_train_mean = np.reshape(u_train_mean, (-1, num_res, 1))
        u_train_std = np.reshape(u_train_std, (-1, num_res, 1))
        v_train_mean = np.reshape(v_train_mean, (-1, num_res, 1))
        v_train_std = np.reshape(v_train_std, (-1, num_res, 1))

        F_train = np.reshape(a_train, (-1, s))
        U_train = np.reshape(u_train, (-1, num_res, 1))
        V_train = np.reshape(v_train, (-1, num_res, 1))

        F_train = (F_train - f_train_mean)/(f_train_std + 1.0e-9)
        U_train = (U_train - u_train_mean)/(u_train_std + 1.0e-9)
        V_train = (V_train - v_train_mean)/(v_train_std + 1.0e-9)

        F_test = np.reshape(a_test, (-1, s))
        U_test = np.reshape(u_test, (-1, num_res, 1))
        V_test = np.reshape(v_test, (-1, num_res, 1))

        F_test = (F_test - f_train_mean)/(f_train_std + 1.0e-9)

        U = np.reshape(U_train, (-1, num_res))
        V = np.reshape(V_train, (-1, num_res))
        C_u = 1./(num_train-1)*np.matmul(U.T, U)
        C_v = 1./(num_train-1)*np.matmul(V.T, V)
        lam_u, phi_u = np.linalg.eigh(C_u)
        lam_v, phi_v = np.linalg.eigh(C_v)

        lam_u = np.flip(lam_u)
        phi_u = np.fliplr(phi_u)
        lam_v = np.flip(lam_v)
        phi_v = np.fliplr(phi_v)

        u_cumsum = np.cumsum(lam_u)
        v_cumsum = np.cumsum(lam_v)
        u_per = u_cumsum[self.modes-1]/u_cumsum[-1]
        v_per = v_cumsum[self.modes-1]/v_cumsum[-1]

        u_basis = phi_u[:, :self.modes]
        v_basis = phi_v[:, :self.modes]

        print("Kept Energy: u: %.3f, v: %.3f"%(u_per, v_per))
        """

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

        return F_train, U_train, F_test, U_test, x_train, u_basis, lam_u

    def minibatch(self):
        batch_id = np.random.choice(self.F_train.shape[0], self.bs, replace=False)
        f_train = [self.F_train[i : i + 1] for i in batch_id]
        f_train = np.concatenate(f_train, axis=0)
        u_train = [self.U_train[i : i + 1] for i in batch_id]
        u_train = np.concatenate(u_train, axis=0)
        """
        v_train = [self.V_train[i:i+1] for i in batch_id]
        v_train = np.concatenate(v_train, axis=0)
        """

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

        return self.x_train, f_train, u_train, Xmin, Xmax

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

        x_test = self.x_train
        f_test, u_test = self.F_test, self.U_test

        return x_test, f_test, u_test
