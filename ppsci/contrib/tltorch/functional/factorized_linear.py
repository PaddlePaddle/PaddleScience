from .factorized_tensordot import tensor_dot_tucker, tensor_dot_cp
import tensorly as tl
tl.set_backend('pytorch')

# Author: Jean Kossaifi

def linear_tucker(tensor, tucker_matrix, transpose=True, channels_first=True):
    if transpose:
        contraction_axis = 1
    else:
        contraction_axis = 0
    n_rows = len(tucker_matrix.tensorized_shape[contraction_axis])
    tensor = tensor.reshape(-1, *tucker_matrix.tensorized_shape[contraction_axis])

    modes_tensor = list(range(tensor.ndim - n_rows, tensor.ndim))
    if transpose:
        modes_tucker = list(range(n_rows, tucker_matrix.order))
    else:
        modes_tucker = list(range(n_rows))

    return tensor_dot_tucker(tensor, tucker_matrix, (modes_tensor, modes_tucker))

def linear_cp(tensor, cp_matrix, transpose=True):
    if transpose:
        out_features, in_features = len(cp_matrix.tensorized_shape[0]), len(cp_matrix.tensorized_shape[1])
        in_shape = cp_matrix.tensorized_shape[1]
        modes_cp = list(range(out_features, cp_matrix.order))
    else:
        in_features, out_features = len(cp_matrix.tensorized_shape[0]), len(cp_matrix.tensorized_shape[1])
        in_shape = cp_matrix.tensorized_shape[0]
        modes_cp = list(range(in_features))
    tensor = tensor.reshape(-1, *in_shape)

    modes_tensor = list(range(1, tensor.ndim))

    return tensor_dot_cp(tensor, cp_matrix, (modes_tensor, modes_cp))


def linear_blocktt(tensor, tt_matrix, transpose=True):
    if transpose:
        contraction_axis = 1
    else:
        contraction_axis = 0
    ndim = len(tt_matrix.tensorized_shape[contraction_axis])
    tensor = tensor.reshape(-1, *tt_matrix.tensorized_shape[contraction_axis])

    bs = 'a'
    start = ord(bs) + 1
    in_idx = bs + ''.join(chr(i) for i in [start+i for i in range(ndim)])
    factors_idx = []
    for i in range(ndim):
        if transpose:
            idx = [start+ndim*2+i, start+ndim+i, start+i, start+ndim*2+i+1]
        else:
            idx = [start+ndim*2+i, start+i, start+ndim+i, start+ndim*2+i+1]
        factors_idx.append(''.join(chr(j) for j in idx))
    out_idx = bs + ''.join(chr(i) for i in [start + ndim + i for i in range(ndim)])
    eq = in_idx + ',' + ','.join(i for i in factors_idx) + '->' + out_idx
    res = tl.einsum(eq, tensor, *tt_matrix.factors)
    return tl.reshape(res, (tl.shape(res)[0], -1))

