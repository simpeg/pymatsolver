from pyMKL import pardisoinit, pardiso
from ctypes import POINTER, byref, c_longlong, c_int
import numpy as np
import scipy.sparse as sp
from numpy import ctypeslib

"""
mtype options
1 -> real and structurally symmetric
2 -> real and symmetric positive definite
-2 -> real and symmetric indefinite
3 -> complex and structurally symmetric
4 -> complex and Hermitian positive definite
-4 -> complex and Hermitian indefinite
6 -> complex and symmetric
11 -> real and nonsymmetric
13 -> complex and nonsymmetric


phase options
11 -> Analysis
12 -> Analysis, numerical factorization
13 -> Analysis, numerical factorization, solve, iterative refinement
22 -> Numerical factorization
23 -> Numerical factorization, solve, iterative refinement
33 -> Solve, iterative refinement
331 -> like phase=33, but only forward substitution
332 -> like phase=33, but only diagonal substitution (if available)
333 -> like phase=33, but only backward substitution
0 -> Release internal memory for L and U matrix number mnum
-1 -> Release all internal memory for all matrices
"""

class pardisoSolver(object):
    """docstring for pardisoSolver"""
    def __init__(self, A, mtype=11, verbose=False):
        
        self.mtype = mtype
        if mtype in [1, 3]:
            msg = "mtype = 1 - structurally symmetric matrices not supported"
            raise NotImplementedError(msg)
        elif mtype in [2, -2, 4, -4, 6, 11, 13]:
            pass
        else:
            msg = "Invalid mtype: mtype={}".format(mtype)
            raise ValueError(msg)
            

        self.n = A.shape[0]

        if mtype in [4, -4, 6, 13]:
            # Complex matrix
            self.dtype = np.complex128
        elif mtype in [2, -2, 11]:
            # Real matrix
            self.dtype = np.float64
        self.ctypes_dtype = ctypeslib.ndpointer(self.dtype)

        # If A is symmetric, store only the upper triangular portion 
        if mtype in [2, -2, 4, -4, 6]:
            A = sp.triu(A, format='csr')
        elif mtype in [11, 13]:
            A = A.tocsr()
        
        self.a = A.data
        self.ia = A.indptr
        self.ja = A.indices

        self._MKL_a = self.a.ctypes.data_as(self.ctypes_dtype)
        self._MKL_ia = self.ia.ctypes.data_as(POINTER(c_int))
        self._MKL_ja = self.ja.ctypes.data_as(POINTER(c_int))

        # Hardcode some parameters for now...
        self.maxfct = 1
        self.mnum = 1
        self.perm = 0

        if verbose:
            self.msglvl = 1
        else:
            self.msglvl = 0

        # Initialize handle to data structure
        self.pt = np.zeros(64, np.int64)
        self._MKL_pt = self.pt.ctypes.data_as(POINTER(c_longlong))

        # Initialize parameters
        self.iparm = np.zeros(64, dtype=np.int32)
        self._MKL_iparm = self.iparm.ctypes.data_as(POINTER(c_int))

        # Initialize pardiso
        pardisoinit(self._MKL_pt, byref(c_int(self.mtype)), self._MKL_iparm)

        # Set iparm
        self.iparm[1] = 3 # Use parallel nested dissection for reordering
        self.iparm[23] = 1 # Use parallel factorization
        self.iparm[34] = 1 # Zero base indexing


    def run_pardiso(self, phase, rhs=None):
        
        if rhs is None:
            nrhs = 0
            x = np.zeros(1)
            rhs = np.zeros(1)
        else:
            nrhs = 1
            rhs = rhs.astype(self.dtype)
            x = np.zeros(self.n, dtype=self.dtype)

        MKL_rhs = rhs.ctypes.data_as(self.ctypes_dtype)
        MKL_x = x.ctypes.data_as(self.ctypes_dtype)
        ERR = 0

        pardiso(self._MKL_pt,               # pt
                byref(c_int(self.maxfct)),  # maxfct
                byref(c_int(self.mnum)),    # mnum
                byref(c_int(self.mtype)),   # mtype
                byref(c_int(phase)),        # phase
                byref(c_int(self.n)),       # n
                self._MKL_a,                # a
                self._MKL_ia,               # ia
                self._MKL_ja,               # ja
                byref(c_int(self.perm)),    # perm
                byref(c_int(nrhs)),         # nrhs
                self._MKL_iparm,            # iparm
                byref(c_int(self.msglvl)),  # msglvl
                MKL_rhs,                    # b
                MKL_x,                      # x
                byref(c_int(ERR)))          # error

        return x
