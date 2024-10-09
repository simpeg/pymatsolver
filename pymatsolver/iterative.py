import warnings
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import bicgstab, cg, aslinearoperator
from packaging.version import Version
from .wrappers import WrapIterative
from .solvers import Base

# The tol kwarg was removed from bicgstab in scipy 1.14.0.
# See https://docs.scipy.org/doc/scipy-1.12.0/reference/generated/scipy.sparse.linalg.bicgstab.html
RTOL_ARG_NAME = "rtol" if Version(scipy.__version__) >= Version("1.14.0") else "tol"

SolverCG = WrapIterative(cg, name="SolverCG")
SolverBiCG = WrapIterative(bicgstab, name="SolverBiCG")

class BiCGJacobi(Base):
    """Diagonal pre-conditioned BiCG solver.

    Parameters
    ----------
    A : matrix
        The matrix to solve, must have a ``diagonal()`` method.
    symmetric: boolean, optional
        .. deprecated:: 0.3.0
            `symmetric` is deprecated. It is unused, and will be removed in pymatsolver 0.4.0.
    maxiter : int, optional
        The maximum number of BiCG iterations to perform.
    rtol : float, optional
        The relative tolerance for the BiCG solver to terminate.
    atol : float, optional
        The absolute tolerance for the BiCG solver to terminate.
    check_accuracy : bool, optional
        Whether to check the accuracy of the solution.
    check_rtol : float, optional
        The relative tolerance to check against for accuracy.
    check_atol : float, optional
        The absolute tolerance to check against for accuracy.
    accuracy_tol : float, optional
        Relative accuracy tolerance.
        .. deprecated:: 0.3.0
            `accuracy_tol` will be removed in pymatsolver 0.4.0. Use `check_rtol` and `check_atol` instead.
    **kwargs
        Extra keyword arguments passed to the base class.
    """

    def __init__(self, A, symmetric=None, maxiter=1000, rtol=1E-6, atol=0.0, check_accuracy=False, check_rtol=1e-6, check_atol=0, accuracy_tol=None, **kwargs):
        if symmetric is not None:
            warnings.warn(
                "The symmetric keyword argument is unused and is deprecated. It will be removed in pymatsolver 0.4.0.",
                FutureWarning, stacklevel=2
            )
        super().__init__(A, check_accuracy=check_accuracy, check_rtol=check_rtol, check_atol=check_atol, accuracy_tol=accuracy_tol, **kwargs)
        self._factored = False
        self.maxiter = maxiter
        self.rtol = rtol
        self.atol = atol

    @property
    def maxiter(self):
        return self._maxiter

    @maxiter.setter
    def maxiter(self, value):
        self._maxiter = int(value)

    @property
    def rtol(self):
        return self._rtol

    @rtol.setter
    def rtol(self, value):
        value = float(value)
        if value > 0:
            self._rtol = value
        else:
            raise ValueError("rtol must be greater than 0.")

    @property
    def atol(self):
        return self._atol

    @atol.setter
    def atol(self, value):
        value = float(value)
        if value >= 0:
            self._atol = value
        else:
            raise ValueError("atol must be greater than or equal to 0.")

    def get_attributes(self):
        attrs = super().get_attributes()
        attrs["maxiter"] = self.maxiter
        attrs["rtol"] = self.rtol
        attrs["atol"] = self.atol
        return attrs

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


    def _solve_single(self, rhs):
        self.factor()
        sol, info = bicgstab(
            self.A, rhs,
            maxiter=self.maxiter,
            M=self.M,
            **self._tols,
        )
        return sol

    def _solve_multiple(self, rhs):
        self.factor()
        sol = np.empty_like(rhs)
        for icol in range(rhs.shape[1]):
            sol[:, icol] = self._solve_single(rhs[:, icol])
        return sol

