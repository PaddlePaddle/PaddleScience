from textwrap import indent
from typing import Optional, List, Tuple, Dict, Union, Any

import paddle
import scipy.sparse
from storage import SparseStorage

class SparseTensor(object):
    storage: SparseStorage

    def __init__(self, row: Optional[paddle.Tensor] = None,
                 rowptr: Optional[paddle.Tensor] = None,
                 col: Optional[paddle.Tensor] = None,
                 value: Optional[paddle.Tensor] = None,
                 sparse_sizes: Optional[Tuple[int, int]] = None,
                 is_sorted: bool = False):
        self.storage = SparseStorage(row=row, rowptr=rowptr, col=col,
                                     value=value, sparse_sizes=sparse_sizes,
                                     rowcount=None, colptr=None, colcount=None,
                                     csr2csc=None, csc2csr=None,
                                     is_sorted=is_sorted)
        
    