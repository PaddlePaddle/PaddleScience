import numpy as np
import math
from bisect import insort_left

# Author : Jean Kossaifi

def factorize(value, min_value=2, remaining=-1):
    """Factorize an integer input value into it's smallest divisors
    
    Parameters
    ----------
    value : int
        integer to factorize
    min_value : int, default is 2
        smallest divisors to use
    remaining : int, default is -1
        DO NOT SPECIFY THIS VALUE, IT IS USED FOR TAIL RECURSION

    Returns
    -------
    factorization : int tuple
        ints such that prod(factorization) == value
    """
    if value <= min_value or remaining == 0:
        return (value, )
    lim = math.isqrt(value)
    for i in range(min_value, lim+1):
        if value == i:
            return (i, )
        if not (value % i):
            return (i, *factorize(value//i, min_value=min_value, remaining=remaining-1))
    return (value, )

def merge_ints(values, size):
    """Utility function to merge the smallest values in a given tuple until it's length is the given size
    
    Parameters
    ----------
    values : int list
        list of values to merge
    size : int
        target len of the list
        stop merging when len(values) <= size
    
    Returns
    -------
    merge_values : list of size ``size``
    """
    if len(values) <= 1:
        return values

    values = sorted(list(values))
    while (len(values) > size):
        a, b, *values = values
        insort_left(values, a*b)
    
    return tuple(values)
    
def get_tensorized_shape(in_features, out_features, order=None, min_dim=2, verbose=True):
    """ Factorizes in_features and out_features such that:
    * they both are factorized into the same number of integers
    * they should both be factorized into `order` integers
    * each of the factors should be at least min_dim
    
    This is used to tensorize a matrix of size (in_features, out_features) into a higher order tensor
    
    Parameters
    ----------
    in_features, out_features : int
    order : int
        the number of integers that each input should be factorized into
    min_dim : int
        smallest acceptable integer value for the factors
        
    Returns
    -------
    in_tensorized, out_tensorized : tuple[int]
        tuples of ints used to tensorize each dimension
        
    Notes
    -----
    This is a bruteforce solution but is enough for the dimensions we encounter in DNNs
    """
    in_ten = factorize(in_features, min_value=min_dim)
    out_ten = factorize(out_features, min_value=min_dim, remaining=len(in_ten))
    if order is not None:
        merge_size = min(order, len(in_ten), len(out_ten))
    else:
        merge_size = min(len(in_ten), len(out_ten))

    if len(in_ten) > merge_size:
        in_ten = merge_ints(in_ten, size=merge_size)
    if len(out_ten) > merge_size:
        out_ten = merge_ints(out_ten, size=merge_size)

    if verbose:
        print(f'Tensorizing (in, out)=({in_features, out_features}) -> ({in_ten, out_ten})')
    return in_ten, out_ten
