# Author: Jean Kossaifi

import tensorly as tl
from tensorly.tenalg.tenalg_utils import _validate_contraction_modes
tl.set_backend('pytorch') 

einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


def tensor_dot_tucker(tensor, tucker, modes, batched_modes=()):
    """Batched tensor contraction between a dense tensor and a Tucker tensor on specified modes
    
    Parameters
    ----------
    tensor : DenseTensor
    tucker : TuckerTensor
    modes : int list or int
        modes on which to contract tensor1 and tensor2
    batched_modes : int or tuple[int]

    Returns
    -------
    contraction : tensor contracted with cp on the specified modes
    """
    modes_tensor, modes_tucker = _validate_contraction_modes(
        tl.shape(tensor), tucker.tensor_shape, modes)
    input_order = tensor.ndim
    weight_order = tucker.order
    
    batched_modes_tensor, batched_modes_tucker = _validate_contraction_modes(
        tl.shape(tensor), tucker.tensor_shape, batched_modes)

    sorted_modes_tucker = sorted(modes_tucker+batched_modes_tucker, reverse=True)
    sorted_modes_tensor = sorted(modes_tensor+batched_modes_tensor, reverse=True)
    
    # Symbol for dimensionality of the core
    rank_sym = [einsum_symbols[i] for i in range(weight_order)]
    
    # Symbols for tucker weight size
    tucker_sym = [einsum_symbols[i+weight_order] for i in range(weight_order)]
    
    # Symbols for input tensor
    tensor_sym = [einsum_symbols[i+2*weight_order] for i in range(tensor.ndim)]
    
    # Output: input + weights symbols after removing contraction symbols
    output_sym = tensor_sym + tucker_sym
    for m in sorted_modes_tucker:
        if m in modes_tucker: #not batched
            output_sym.pop(m+input_order)
    for m in sorted_modes_tensor:
        # It's batched, always remove
        output_sym.pop(m)
        
    # print(tensor_sym, tucker_sym, modes_tensor, batched_modes_tensor)
    for i, e in enumerate(modes_tensor):
        tensor_sym[e] = tucker_sym[modes_tucker[i]]
    for i, e in enumerate(batched_modes_tensor):
        tensor_sym[e] = tucker_sym[batched_modes_tucker[i]]

    # Form the actual equation: tensor, core, factors -> output
    eq = ''.join(tensor_sym)
    eq += ',' + ''.join(rank_sym)
    eq += ',' + ','.join(f'{s}{r}' for s,r in zip(tucker_sym,rank_sym))
    eq += '->' + ''.join(output_sym)
    
    return tl.einsum(eq, tensor, tucker.core, *tucker.factors)



def tensor_dot_cp(tensor, cp, modes):
    """Contracts a to CP tensors in factorized form
    
    Returns
    -------
    tensor = tensor x cp_matrix.to_matrix().T
    """
    try:
        cp_shape = cp.tensor_shape
    except AttributeError:
        cp_shape = cp.shape
    modes_tensor, modes_cp = _validate_contraction_modes(tl.shape(tensor), cp_shape, modes)

    tensor_order = tl.ndim(tensor)
    # CP rank = 'a', start at b
    start = ord('b')
    eq_in = ''.join(f'{chr(start+index)}' for index in range(tensor_order))
    eq_factors = []
    eq_res = ''.join(eq_in[i] if i not in modes_tensor else '' for i in range(tensor_order))
    counter_joint = 0 # contraction modes, shared indices between tensor and CP
    counter_free = 0 # new uncontracted modes from the CP
    for i in range(len(cp.factors)):
        if i in modes_cp:
            eq_factors.append(f'{eq_in[modes_tensor[counter_joint]]}a')
            counter_joint += 1
        else:
            eq_factors.append(f'{chr(start+tensor_order+counter_free)}a')
            eq_res += f'{chr(start+tensor_order+counter_free)}'
            counter_free += 1
    
    eq_factors = ','.join(f for f in eq_factors)
    eq = eq_in + ',a,' + eq_factors + '->' + eq_res
    res = tl.einsum(eq, tensor, cp.weights, *cp.factors)
    
    return res