import numpy as np
import scipy.io as io

class Dataset:
    def __init__(self, t_range, Neta, Nf_train):
        self.t_range = t_range
        self.Neta = Neta
        self.Nf_train = Nf_train

    def build_data(self):
        data = io.loadmat('./VIV_Training.mat')
        t, eta, f = data['t'], data['eta_y'], data['f_y']

        t_0 = t.min(0)
        tmin = np.reshape(t_0, (-1, 1))
        t_1 = t.max(0)
        tmax = np.reshape(t_1, (-1, 1))

        N = t.shape[0]
        N = 160
        idx = np.random.choice(N, self.Neta, replace=False)
        t_eta = t[idx]
        eta = eta[idx]
      
        #idx = np.random.choice(N, self.Nf_train, replace=False)
        t_f = t[idx]
        f = f[idx]

        return t_eta, eta, t_f, f, tmin, tmax
