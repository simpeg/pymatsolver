import scipy.sparse as sp, numpy as np
from pymatsolver.Base import BaseSolver


class ForwardSolver(BaseSolver):

    _transposeClass = None

    def __init__(self, A):
        self.A = A.tocsr()

    def _solveM(self, rhs):

        vals = self.A.data
        rowptr = self.A.indptr
        colind = self.A.indices
        x = np.empty_like(rhs)
        for i in xrange(self.A.shape[0]):
            ith_row = vals[rowptr[i] : rowptr[i+1]]
            cols = colind[rowptr[i] : rowptr[i+1]]
            x_vals = x[cols]
            x[i] = (rhs[i] - np.dot(ith_row[:-1], x_vals[:-1])) / ith_row[-1]
        return x

    _solve1 = _solveM


class BackwardSolver(BaseSolver):

    _transposeClass = None

    def __init__(self, A):
        self.A = A.tocsr()

    def _solveM(self, rhs):

        vals = self.A.data
        rowptr = self.A.indptr
        colind = self.A.indices
        x = np.empty_like(rhs)
        for i in reversed(xrange(self.A.shape[0])):
            ith_row = vals[rowptr[i] : rowptr[i+1]]
            cols = colind[rowptr[i] : rowptr[i+1]]
            x_vals = x[cols]
            x[i] = (rhs[i] - np.dot(ith_row[1:], x_vals[1:])) / ith_row[0]
        return x

    _solve1 = _solveM


if __name__ == '__main__':
    TOL = 1e-12
    n = 30
    A = sp.rand(n, n, 0.4) + sp.identity(n)
    AL = sp.tril(A)
    ALinv = ForwardSolver(AL)
    e = np.ones((n,5))
    rhs = AL * e
    x = ALinv * rhs
    print np.linalg.norm(e-x,np.inf), TOL

    AU = sp.triu(A)
    AUinv = BackwardSolver(AU)
    e = np.ones((n,5))
    rhs = AU * e
    x = AUinv * rhs
    print np.linalg.norm(e-x,np.inf), TOL
