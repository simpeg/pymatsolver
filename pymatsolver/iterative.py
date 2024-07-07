import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import bicgstab, cg
from .solvers import Base
from .wrappers import WrapIterative


SolverCG = WrapIterative(cg, name="SolverCG")
SolverBiCG = WrapIterative(bicgstab, name="SolverBiCG")

import scipy
_rtol_call = False
scipy_major, scipy_minor, scipy_patch = scipy.__version__.split(".")
if int(scipy_major) >= 1 and int(scipy_minor) >= 12:
    _rtol_call = True

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
        self.M = sp.linalg.interface.aslinearoperator(Ainv)
        self._factored = True

    @property
    def _tols(self):
        if _rtol_call:
            return {'rtol': self.rtol, 'atol': self.atol}
        else:
            return {'tol': self.rtol, 'atol': self.atol}


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
                       maxiter=self.maxiter, M=self.M,
            **self._tols,)[0])
        out = np.hstack(sol)
        out.shape
        return out

    def clean(self):
        self.M = None
        self.A = None
        self._factored = False

