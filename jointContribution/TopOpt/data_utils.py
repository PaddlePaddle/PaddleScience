import numpy as np
import paddle


def augmentation(input_dict, label_dict, weight_dict=None):
    """Apply random transformation from D4 symmetry group
    # Arguments
        x_batch, y_batch: input tensors of size `(batch_size, any, height, width)`
    """
    X = paddle.to_tensor(input_dict["input"])
    Y = paddle.to_tensor(label_dict["output"])
    n_obj = len(X)
    indices = np.arange(n_obj)
    np.random.shuffle(indices)

    if len(X.shape) == 3:
        # random horizontal flip
        if np.random.random() > 0.5:
            X = paddle.flip(X, axis=2)
            Y = paddle.flip(Y, axis=2)
        # random vertical flip
        if np.random.random() > 0.5:
            X = paddle.flip(X, axis=1)
            Y = paddle.flip(Y, axis=1)
        # random 90* rotation
        if np.random.random() > 0.5:
            new_perm = list(range(len(X.shape)))
            new_perm[1], new_perm[2] = new_perm[2], new_perm[1]
            X = paddle.transpose(X, perm=new_perm)
            Y = paddle.transpose(Y, perm=new_perm)
        X = X.reshape([1] + X.shape)
        Y = Y.reshape([1] + Y.shape)
    else:
        # random horizontal flip
        batch_size = X.shape[0]
        mask = np.random.random(size=batch_size) > 0.5
        X[mask] = paddle.flip(X[mask], axis=3)
        Y[mask] = paddle.flip(Y[mask], axis=3)
        # random vertical flip
        mask = np.random.random(size=batch_size) > 0.5
        X[mask] = paddle.flip(X[mask], axis=2)
        Y[mask] = paddle.flip(Y[mask], axis=2)
        # random 90* rotation
        mask = np.random.random(size=batch_size) > 0.5
        new_perm = list(range(len(X.shape)))
        new_perm[2], new_perm[3] = new_perm[3], new_perm[2]
        X[mask] = paddle.transpose(X[mask], perm=new_perm)
        Y[mask] = paddle.transpose(Y[mask], perm=new_perm)

    return {"input": X}, {"output": Y}, None
