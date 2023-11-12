import scipy.io
import numpy as np

try:
    from pyDOE import lhs
    # Only needed for PINN's dataset
except ImportError:
    lhs = None

import paddle
from paddle.io import Dataset
from .utils import get_grid3d, convert_ic, paddle2dgrid

def online_loader(sampler, S, T, time_scale, batchsize=1):
    while True:
        u0 = sampler.sample(batchsize)
        a = convert_ic(u0, batchsize,
                       S, T,
                       time_scale=time_scale)
        yield a

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

class MatReader(object):
    def __init__(self, file_path, to_paddle=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_paddle = to_paddle
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        self.data = scipy.io.loadmat(self.file_path)
        self.old_mat = True

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_paddle:

            x = paddle.to_tensor(x)
            if self.to_cuda:
                x = x

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_paddle(self, to_paddle):
        self.to_paddle = to_paddle

    def set_float(self, to_float):
        self.to_float = to_float

class BurgersLoader(object):
    def __init__(self, datapath, nx=2 ** 10, nt=100, sub=8, sub_t=1, new=False):
        dataloader = MatReader(datapath)
        self.sub = sub
        self.sub_t = sub_t
        self.s = nx // sub
        self.T = nt // sub_t
        self.new = new
        if new:
            self.T += 1
        self.x_data = dataloader.read_field('input')[:, ::sub]
        self.y_data = dataloader.read_field('output')[:, ::sub_t, ::sub]
        self.v = dataloader.read_field('visc').item()

    def make_loader(self, n_sample, batch_size, start=0, train=True):
        Xs = self.x_data[start:start + n_sample]
        ys = self.y_data[start:start + n_sample]
        if self.new:
            gridx = paddle.to_tensor(np.linspace(0, 1, self.s + 1)[:-1], dtype='float32')
            gridt = paddle.to_tensor(np.linspace(0, 1, self.T), dtype='float32')
        else:
            gridx = paddle.to_tensor(np.linspace(0, 1, self.s), dtype='float32')
            gridt = paddle.to_tensor(np.linspace(0, 1, self.T + 1)[1:], dtype='float32')
        gridx = gridx.reshape([1, 1, self.s])
        gridt = gridt.reshape([1, self.T, 1])

        Xs = Xs.reshape([n_sample, 1, self.s]).tile([1, self.T, 1])
        Xs = paddle.stack([Xs, gridx.tile([n_sample, self.T, 1]), gridt.tile([n_sample, 1, self.s])], axis=3)
        dataset = paddle.io.TensorDataset([Xs, ys])
        if train:
            loader = paddle.io.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        else:
            loader = paddle.io.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader

class NSLoader(object):
    def __init__(self, datapath1,
                 nx, nt,
                 datapath2=None, sub=1, sub_t=1,
                 N=100, t_interval=1.0):
        '''
        Load data from npy and reshape to (N, X, Y, T)
        Args:
            datapath1: path to data
            nx:
            nt:
            datapath2: path to second part of data, default None
            sub:
            sub_t:
            N:
            t_interval:
        '''
        self.S = nx // sub
        self.T = int(nt * t_interval) // sub_t + 1
        self.time_scale = t_interval
        data1 = np.load(datapath1)
        data1 = paddle.to_tensor(data1, dtype='float32')[..., ::sub_t, ::sub, ::sub]

        if datapath2 is not None:
            data2 = np.load(datapath2)
            data2 = paddle.to_tensor(data2, dtype='float32')[..., ::sub_t, ::sub, ::sub]
        if t_interval == 0.5:
            data1 = self.extract(data1)
            if datapath2 is not None:
                data2 = self.extract(data2)
        part1 = data1.permute(0, 2, 3, 1)
        if datapath2 is not None:
            part2 = data2.permute(0, 2, 3, 1)
            self.data = paddle.concat((part1, part2), axis=0)
        else:
            self.data = part1

    def make_loader(self, n_sample, batch_size, start=0, train=True):
        if train:
            a_data = self.data[start:start + n_sample, :, :, 0].reshape([n_sample, self.S, self.S])
            u_data = self.data[start:start + n_sample].reshape([n_sample, self.S, self.S, self.T])
        else:
            a_data = self.data[-n_sample:, :, :, 0].reshape([n_sample, self.S, self.S])
            u_data = self.data[-n_sample:].reshape([n_sample, self.S, self.S, self.T])
        a_data = a_data.reshape([n_sample, self.S, self.S, 1, 1]).tile([1, 1, 1, self.T, 1])
        gridx, gridy, gridt = get_grid3d(self.S, self.T, time_scale=self.time_scale)
        a_data = paddle.concat((gridx.tile([n_sample, 1, 1, 1, 1]), gridy.tile([n_sample, 1, 1, 1, 1]),
                            gridt.tile([n_sample, 1, 1, 1, 1]), a_data), axis=-1)
        dataset = paddle.io.TensorDataset(a_data, u_data)
        loader = paddle.io.DataLoader(dataset, batch_size=batch_size, shuffle=train)
        return loader

    def make_dataset(self, n_sample, start=0, train=True):
        if train:
            a_data = self.data[start:start + n_sample, :, :, 0].reshape([n_sample, self.S, self.S])
            u_data = self.data[start:start + n_sample].reshape([n_sample, self.S, self.S, self.T])
        else:
            a_data = self.data[-n_sample:, :, :, 0].reshape([n_sample, self.S, self.S])
            u_data = self.data[-n_sample:].reshape([n_sample, self.S, self.S, self.T])
        a_data = a_data.reshape([n_sample, self.S, self.S, 1, 1]).tile([1, 1, 1, self.T, 1])
        gridx, gridy, gridt = get_grid3d(self.S, self.T)
        a_data = paddle.concat((
            gridx.tile([n_sample, 1, 1, 1, 1]),
            gridy.tile([n_sample, 1, 1, 1, 1]),
            gridt.tile([n_sample, 1, 1, 1, 1]),
            a_data), axis=-1)
        dataset = paddle.io.TensorDataset(a_data, u_data)
        return dataset

    @staticmethod
    def extract(data):
        '''
        Extract data with time range 0-0.5, 0.25-0.75, 0.5-1.0, 0.75-1.25,...
        Args:
            data: tensor with size N x 129 x 128 x 128

        Returns:
            output: (4*N-1) x 65 x 128 x 128
        '''
        T = data.shape[1] // 2
        interval = data.shape[1] // 4
        N = data.shape[0]
        new_data = paddle.zeros(4 * N - 1, T + 1, data.shape[2], data.shape[3])
        for i in range(N):
            for j in range(4):
                if i == N - 1 and j == 3:
                    # reach boundary
                    break
                if j != 3:
                    new_data[i * 4 + j] = data[i, interval * j:interval * j + T + 1]
                else:
                    new_data[i * 4 + j, 0: interval] = data[i, interval * j:interval * j + interval]
                    new_data[i * 4 + j, interval: T + 1] = data[i + 1, 0:interval + 1]
        return new_data

class KFDataset(Dataset):
    def __init__(self, paths, 
                 data_res, pde_res, 
                 raw_res, 
                 n_samples=None, 
                 total_samples=None,
                 idx=0,
                 offset=0,
                 t_duration=1.0):
        super().__init__()
        self.data_res = data_res    # data resolution
        self.pde_res = pde_res      # pde loss resolution
        self.raw_res = raw_res      # raw data resolution
        self.t_duration = t_duration
        self.paths = paths
        self.offset = offset
        self.n_samples = n_samples
        if t_duration == 1.0:
            self.T = self.pde_res[2]
        else:
            self.T = int(self.pde_res[2] * t_duration) + 1    # number of points in time dimension

        self.load()
        if total_samples is not None:
            print(f'Load {total_samples} samples starting from {idx}th sample')
            self.data = self.data[idx:idx + total_samples]
            self.a_data = self.a_data[idx:idx + total_samples]
            
        self.data_s_step = pde_res[0] // data_res[0]
        self.data_t_step = (pde_res[2] - 1) // (data_res[2] - 1)

    def load(self):
        datapath = self.paths[0]
        raw_data = np.load(datapath, mmap_mode='r')
        # subsample ratio
        sub_x = self.raw_res[0] // self.data_res[0]
        sub_t = (self.raw_res[2] - 1) // (self.data_res[2] - 1)
        
        a_sub_x = self.raw_res[0] // self.pde_res[0]
        # load data
        data = raw_data[self.offset: self.offset + self.n_samples, ::sub_t, ::sub_x, ::sub_x]
        # divide data
        if self.t_duration != 0.:
            end_t = self.raw_res[2] - 1
            K = int(1/self.t_duration)
            step = end_t // K
            data = self.partition(data)
            a_data = raw_data[self.offset: self.offset + self.n_samples, 0:end_t:step, ::a_sub_x, ::a_sub_x]
            a_data = a_data.reshape([self.n_samples * K, 1, self.pde_res[0], self.pde_res[1]])    # 2N x 1 x S x S
        else:
            a_data = raw_data[self.offset: self.offset + self.n_samples, 0:1, ::a_sub_x, ::a_sub_x]

        # convert into paddle tensor
        data = paddle.to_tensor(data, dtype='float32')
        a_data = paddle.to_tensor(a_data, dtype='float32').transpose([0, 2, 3, 1])
        self.data = data.transpose([0, 2, 3, 1])

        S = self.pde_res[1]
        
        a_data = a_data[:, :, :, :, None]   # N x S x S x 1 x 1
        gridx, gridy, gridt = get_grid3d(S, self.T)
        self.grid = paddle.concat((gridx[0], gridy[0], gridt[0]), axis=-1)   # S x S x T x 3
        self.a_data = a_data

    def partition(self, data):
        '''
        Args:
            data: tensor with size N x T x S x S

        Returns:
            output: int(1/t_duration) *N x (T//2 + 1) x 128 x 128
        '''
        N, T, S = data.shape[:3]
        K = int(1 / self.t_duration)
        new_data = np.zeros((K * N, T // K + 1, S, S))
        step = T // K
        for i in range(N):
            for j in range(K):
                new_data[i * K + j] = data[i, j * step: (j+1) * step + 1]
        return new_data

    def __getitem__(self, idx):
        a_data = paddle.concat((
            self.grid, 
            self.a_data[idx].tile(1, 1, self.T, 1)
        ), axis=-1)
        return self.data[idx], a_data

    def __len__(self, ):
        return self.data.shape[0]

class BurgerData(Dataset):
    '''
    members: 
        - t, x, Exact: raw data
        - X, T: meshgrid 
        - X_star, u_star: flattened (x, t), u array
        - lb, ub: lower bound and upper bound vector
        - X_u, u: boundary condition data (x, t), u
    '''

    def __init__(self, datapath):
        data = scipy.io.loadmat(datapath)

        # raw 2D data
        self.t = data['t'].flatten()[:, None]  # (100,1)
        self.x = data['x'].flatten()[:, None]  # (256, 1)
        self.Exact = np.real(data['usol']).T  # (100, 256)

        # Flattened sequence
        self.get_flatten_data()
        self.get_boundary_data()

    def __len__(self):
        return self.Exact.shape[0]

    def __getitem__(self, idx):
        return self.X_star[idx], self.u_star[idx]

    def get_flatten_data(self):
        X, T = np.meshgrid(self.x, self.t)
        self.X, self.T = X, T
        self.X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        self.u_star = self.Exact.flatten()[:, None]

        # lower bound of (x, t): 2-dimensional vector
        self.lb = self.X_star.min(0)
        # upper bound of (x, t): 2-dimensional vector
        self.ub = self.X_star.max(0)

    def get_boundary_data(self):
        xx1 = np.hstack((self.X[0:1, :].T, self.T[0:1, :].T))
        uu1 = self.Exact[0:1, :].T
        xx2 = np.hstack((self.X[:, 0:1], self.T[:, 0:1]))
        uu2 = self.Exact[:, 0:1]
        xx3 = np.hstack((self.X[:, -1:], self.T[:, -1:]))
        uu3 = self.Exact[:, -1:]
        self.X_u = np.vstack([xx1, xx2, xx3])
        self.u = np.vstack([uu1, uu2, uu3])

    def sample_xt(self, N=10000):
        '''
        Sample (x, t) pairs within the boundary
        Return:
            - X_f: (N, 2) array
        '''
        X_f = self.lb + (self.ub - self.lb) * lhs(2, N)
        X_f = np.vstack((X_f, self.X_u))
        return X_f

    def sample_xu(self, N=100):
        '''
        Sample N points from boundary data
        Return: 
            - X_u: (N, 2) array 
            - u: (N, 1) array
        '''
        idx = np.random.choice(self.X_u.shape[0], N, replace=False)
        X_u = self.X_u[idx, :]
        u = self.u[idx, :]
        return X_u, u

class DarcyFlow(Dataset):
    def __init__(self,
                 datapath,
                 nx, sub,
                 offset=0,
                 num=1):
        self.S = int(nx // sub) + 1 if sub > 1 else nx
        data = scipy.io.loadmat(datapath)
        a = data['coeff']
        u = data['sol']
        self.a = paddle.to_tensor(a[offset: offset + num, ::sub, ::sub], dtype='float32')
        self.u = paddle.to_tensor(u[offset: offset + num, ::sub, ::sub], dtype='float32')
        self.mesh = paddle2dgrid(self.S, self.S)

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, item):
        fa = self.a[item]
        return paddle.concat([fa.unsqueeze(2), self.mesh], axis=2), self.u[item]

class DarcyIC(Dataset):
    def __init__(self,
                 datapath,
                 nx, sub,
                 offset=0,
                 num=1):
        self.S = int(nx // sub) + 1 if sub > 1 else nx
        data = scipy.io.loadmat(datapath)
        a = data['coeff']
        self.a = paddle.to_tensor(a[offset: offset + num, ::sub, ::sub], dtype='float32')
        self.mesh = paddle2dgrid(self.S, self.S)
        data = scipy.io.loadmat(datapath)
        a = data['coeff']
        u = data['sol']
        self.a = paddle.to_tensor(a[offset: offset + num, ::sub, ::sub], dtype='float32')
        self.u = paddle.to_tensor(u[offset: offset + num, ::sub, ::sub], dtype='float32')
        self.mesh = paddle2dgrid(self.S, self.S)

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, item):
        fa = self.a[item]
        return paddle.concat([fa.unsqueeze(2), self.mesh], axis=2) 

class DarcyCombo(Dataset):
    def __init__(self, 
                 datapath, 
                 nx, 
                 sub, pde_sub, 
                 num=1000, offset=0) -> None:
        super().__init__()
        self.S = int(nx // sub) + 1 if sub > 1 else nx
        self.pde_S = int(nx // pde_sub) + 1 if sub > 1 else nx
        data = scipy.io.loadmat(datapath)
        a = data['coeff']
        u = data['sol']
        self.a = paddle.to_tensor(a[offset: offset + num, ::sub, ::sub], dtype='float32')
        self.u = paddle.to_tensor(u[offset: offset + num, ::sub, ::sub], dtype='float32')
        self.mesh = paddle2dgrid(self.S, self.S)
        self.pde_a = paddle.to_tensor(a[offset: offset + num, ::pde_sub, ::pde_sub], dtype='float32')
        self.pde_mesh = paddle2dgrid(self.pde_S, self.pde_S)

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, item):
        fa = self.a[item]
        pde_a = self.pde_a[item]
        data_ic = paddle.concat([fa.unsqueeze[2], self.mesh], axis=2)
        pde_ic = paddle.concat([pde_a.unsqueeze[2], self.pde_mesh], axis=2)
        return data_ic, self.u[item], pde_ic

'''
dataset class for loading initial conditions for Komogrov flow
'''
class KFaDataset(Dataset):
    def __init__(self, paths, 
                 pde_res, 
                 raw_res, 
                 n_samples=None, 
                 offset=0,
                 t_duration=1.0):
        super().__init__()
        self.pde_res = pde_res      # pde loss resolution
        self.raw_res = raw_res      # raw data resolution
        self.t_duration = t_duration
        self.paths = paths
        self.offset = offset
        self.n_samples = n_samples
        if t_duration == 1.0:
            self.T = self.pde_res[2]
        else:
            self.T = int(self.pde_res[2] * t_duration) + 1    # number of points in time dimension

        self.load()

    def load(self):
        datapath = self.paths[0]
        raw_data = np.load(datapath, mmap_mode='r')
        # subsample ratio
        a_sub_x = self.raw_res[0] // self.pde_res[0]
        # load data
        if self.t_duration != 0.:
            end_t = self.raw_res[2] - 1
            K = int(1/self.t_duration)
            step = end_t // K
            a_data = raw_data[self.offset: self.offset + self.n_samples, 0:end_t:step, ::a_sub_x, ::a_sub_x]
            a_data = a_data.reshape([self.n_samples * K, 1, self.pde_res[0], self.pde_res[1]])    # 2N x 1 x S x S
        else:
            a_data = raw_data[self.offset: self.offset + self.n_samples, 0:1, ::a_sub_x, ::a_sub_x]

        # convert into tensor
        a_data = paddle.to_tensor(a_data, dtype='float32').permute(0, 2, 3, 1)
        S = self.pde_res[1]
        a_data = a_data[:, :, :, :, None]   # N x S x S x 1 x 1
        gridx, gridy, gridt = get_grid3d(S, self.T)
        self.grid = paddle.concat((gridx[0], gridy[0], gridt[0]), axis=-1)   # S x S x T x 3
        self.a_data = a_data

    def __getitem__(self, idx):
        a_data = paddle.concat((
            self.grid, 
            self.a_data[idx].tile(1, 1, self.T, 1)
        ), axis=-1)
        return a_data

    def __len__(self, ):
        return self.a_data.shape[0]