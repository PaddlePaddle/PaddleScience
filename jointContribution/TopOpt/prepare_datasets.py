import h5py
import numpy as np


def generate_train_test(data_path, train_test_ratio, n_sample):
    h5data = h5py.File(data_path, "r")
    X = h5data["iters"]
    Y = h5data["targets"]
    idx = np.arange(n_sample)
    np.random.shuffle(idx)
    train_idx = idx <= train_test_ratio * n_sample
    if train_test_ratio == 1:
        X_train = []
        Y_train = []
        for i in range(n_sample):
            X_train.append(np.array(X[i]))
            Y_train.append(np.array(Y[i]))
        return X_train, Y_train
    else:
        X_train = []
        X_test = []
        Y_train = []
        Y_test = []
        for i in range(n_sample):
            if train_idx[i]:
                X_train.append(np.array(X[i]))
                Y_train.append(np.array(Y[i]))
            else:
                X_test.append(np.array(X[i]))
                Y_test.append(np.array(Y[i]))
        return X_train, Y_train, X_test, Y_test
