import scipy.sparse as sp, numpy as np
from pymatsolver.Base import BaseSolver

import TriSolve

class ForwardSolver(BaseSolver):

    _transposeClass = None

    def __init__(self, A):
        self.A = A.tocsr()

    def _solve1(self, rhs):
        A = self.A
        x = TriSolve.forward(A.data, A.indptr, A.indices, rhs, self.A.nnz, rhs.size, 1)
        return x.flatten()

    def _solveM(self, rhs):
        A = self.A
        return TriSolve.forward(A.data, A.indptr, A.indices, rhs, self.A.nnz, *rhs.shape)


class BackwardSolver(BaseSolver):

    _transposeClass = None

    def __init__(self, A):
        self.A = A.tocsr()

    def _solve1(self, rhs):
        A = self.A
        x = TriSolve.backward(A.data, A.indptr, A.indices, rhs, self.A.nnz, rhs.size, 1)
        return x.flatten()

    def _solveM(self, rhs):
        A = self.A
        return TriSolve.backward(A.data, A.indptr, A.indices, rhs, self.A.nnz, *rhs.shape)


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
