import MumpsInterface
import scipy.sparse as sp, numpy as np

class MumpsSolver(object):

    transpose = False
    symmetric = False
    pointer = None

    @property
    def T(self):
        newMS = self.__class__(self.A, symmetric=self.symmetric, fromPointer=self.pointer)
        newMS.transpose = not self.transpose
        return newMS

    def __init__(self, A, symmetric=False, fromPointer=None):
        self.A = A.tocsc()
        self.symmetric = symmetric

        if fromPointer is None:
            self.factor()
        else:
            self.pointer = fromPointer

    @property
    def isfactored(self):
        return self.pointer is not None

    def factor(self):
        if self.isfactored:
            return
        sym = 1 if self.symmetric else 0
        ierr, p = MumpsInterface.factor_mumps(sym,
                             self.A.data,
                             self.A.indices+1,
                             self.A.indptr+1)
        if not ierr == 0:
            raise Exception('MumpsError, boooo.')
        self.pointer = p

    def solve(self, rhs):
        rhs = rhs.flatten(order='F')
        n = self.A.shape[0]
        nrhs = rhs.size // n
        assert rhs.size % n == 0, 'Incorrect shape of RHS.'
        T = 1 if self.transpose else 0
        sol = MumpsInterface.solve_mumps(self.pointer, nrhs, rhs, T)
        if nrhs > 1:
            return sol.reshape((n,nrhs), order='F')
        return sol

    def clean(self):
        MumpsInterface.destroy_mumps(self.pointer)
        self.pointer = None

    def __mul__(self, val):
        if type(val) is np.ndarray:
            return self.solve(val)
        raise TypeError('Can only multiply by a numpy array.')
