import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import bicgstab, cg, aslinearoperator
from .wrappers import WrapIterative

# The tol kwarg was removed from bicgstab in scipy 1.14.0.
# See https://docs.scipy.org/doc/scipy-1.12.0/reference/generated/scipy.sparse.linalg.bicgstab.html
RTOL_ARG_NAME = "rtol" if Version(scipy.__version__) >= Version("1.14.0") else "tol"

SolverCG = WrapIterative(cg, name="SolverCG")
SolverBiCG = WrapIterative(bicgstab, name="SolverBiCG")

class BiCGJacobi(Base):
    """Bicg Solver with Jacobi preconditioner"""

    _factored = False
    solver = None
    maxiter = 1000
    rtol = 1E-6
    atol = 0.0

    def __init__(self, A, symmetric=True):
        self.A = A
        self.symmetric = symmetric
        self.dtype = A.dtype
        self.solver = bicgstab

    def factor(self):
        if self._factored:
            return
        nSize = self.A.shape[0]
        Ainv = sp.spdiags(1./self.A.diagonal(), 0, nSize, nSize)
        self.M = aslinearoperator(Ainv)
        self._factored = True

    @property
    def _tols(self):
        return {RTOL_ARG_NAME: self.rtol, 'atol': self.atol}


    def _solve1(self, rhs):
        self.factor()
        sol, info = self.solver(
            self.A, rhs,
            maxiter=self.maxiter,
            M=self.M,
            **self._tols,
        )
        return sol

    def _solveM(self, rhs):
        self.factor()
        sol = []
        for icol in range(rhs.shape[1]):
            sol.append(self.solver(self.A, rhs[:, icol].flatten(),
                       maxiter=self.maxiter, M=self.M, **self._tols,)[0])
        out = np.hstack(sol)
        out.shape
        return out

    def clean(self):
        self.M = None
        self.A = None
        self._factored = False

