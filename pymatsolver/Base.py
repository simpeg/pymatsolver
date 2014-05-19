import scipy.sparse as sp, numpy as np

def mkvc(x):
    """Creates a vector with the number of dimension specified
    """
    if type(x) == np.matrix:
        x = np.array(x)

    if hasattr(x, 'tovec'):
        x = x.tovec()

    assert type(x) == np.ndarray, "Vector must be a numpy array"

    return x.flatten(order='F')

def sdiag(h):
    """Sparse diagonal matrix"""
    return sp.spdiags(mkvc(h), 0, h.size, h.size, format="csr")


class BaseSolver(object):

    def __init__(self, A):
        self.A = A.tocsr()

    @property
    def _transposeClass(self):
        return self.__class__

    @property
    def T(self):
        if self._transposeClass is None:
            raise Exception('The transpose for the %s class is not possible.' % self.__name__)
        newS = self._transposeClass(self.A.T)
        return newS

    def _solve(self, rhs):

        n = self.A.shape[0]
        assert rhs.size % n == 0, 'Incorrect shape of rhs.'
        nrhs = rhs.size // n

        if len(rhs.shape) == 1 or rhs.shape[1] == 1:
            x = self._solve1(rhs)
        else:
            x = self._solveM(rhs)

        if nrhs == 1:
            return x.flatten()
        elif nrhs > 1:
            return x.reshape((n,nrhs), order='F')


    def clean(self):
        pass

    def __mul__(self, val):
        if type(val) is np.ndarray:
            return self._solve(val)
        raise TypeError('Can only multiply by a numpy array.')


class SolverException(Exception):
    pass


class DiagonalSolver(BaseSolver):

    _transposeClass = None

    def __init__(self, A):
        self.A = A
        self._diagonal = A.diagonal()

    def _solve1(self, rhs):
        return rhs.flatten()/self._diagonal

    def _solveM(self, rhs):
        n = self.A.shape[0]
        nrhs = rhs.size // n
        return rhs/self._diagonal.repeat(nrhs).reshape((n,nrhs))

if __name__ == '__main__':
    A = sp.identity(5)*2.0
    rhs = np.c_[np.arange(1,6),np.arange(2,11,2)]
    print DiagonalSolver(A).solve(rhs)
