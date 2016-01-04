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
