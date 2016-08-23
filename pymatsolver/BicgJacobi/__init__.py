import scipy.sparse as sp
import numpy as np
from pymatsolver.Base import BaseSolver

def JacobiHandle(A):
    nSize = A.shape[0]
    Ainv = sp.spdiags(1./A.diagonal(), 0, nSize, nSize)
    return sp.linalg.interface.aslinearoperator(Ainv)

class BicgJacobiSolver(BaseSolver):
    """

    Bicg Solver with Jacobi preconditioner

    documentation::

        XXX

    """

    isfactored = None
    solver = None
    M = None
    maxIter = 1000
    TOL = 1E-6

    def __init__(self, A, symmetric=True):
        self.A = A
        self.symmetric = symmetric
        self.dtype = A.dtype
        self.solver = sp.linalg.bicgstab
        # Jacobi Preconditioner
        self.M = JacobiHandle(A)
        self.isfactored = True

    def factor(self):
        if self.isfactored is not True:
            self.M = JacobiHandle(self.A)
            self.isfactored = True

    def _solve1(self, rhs):
        self.factor()
        sol, info = self.solver(self.A, rhs, tol=self.TOL, maxiter=self.maxIter, M=self.M)
        return sol

    def _solveM(self, rhs):
        self.factor()
        sol = []
        for icol in range(rhs.shape[1]):
            sol.append(self.solver(self.A, rhs[:, icol].flatten(),
                       tol=self.TOL, maxiter=self.maxIter, M=self.M)[0])
        out = np.hstack(sol)
        out.shape
        return out

    def clean(self):
        self.M = None
        self.A = None


if __name__ == '__main__':
    from SimPEG import EM, np, Utils, sp, Mesh
    mesh = Mesh.TensorMesh([40, 40, 40])
    A = mesh.faceDiv*mesh.faceDiv.T
    rhs = np.random.randn(mesh.nC)
    RHS = np.c_[rhs, rhs]
    Ainv_bicg = BicgJacobiSolver(A)
    sol_bicg = Ainv_bicg*RHS
