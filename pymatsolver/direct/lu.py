import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from pymatsolver.solvers import Base, _SharedFactorMixin
from pymatsolver.wrappers import _valid_kwargs_for_func


class SolverLU(_SharedFactorMixin, Base):
    """Wrapped SolverLU solver, using `scipy.sparse.linalg.splu`.

    The transposed solver (``.T``) re-uses this solver's LU factorization instead of
    factoring ``A.T``. Transposed and conjugated solvers share this factorization, and are
    updated along with it when it is re-factored.

    Parameters
    ----------
    A
        The matrix to use for the solver.
    is_symmetric : bool, optional
        Whether the matrix is symmetric. By default, it will perform some simple tests to check for symmetry, and
        default to ``False`` if those fail.
    is_positive_definite : bool, optional
        Whether the matrix is positive definite.
    is_hermitian : bool, optional
        Whether the matrix is hermitian.
    check_accuracy : bool, optional
        Whether to check the accuracy of the solution.
    check_rtol : float, optional
        The relative tolerance to check against for accuracy.
    check_atol : float, optional
        The absolute tolerance to check against for accuracy.
    **kwargs
        Extra keyword arguments which will attempted to be passed to `scipy.sparse.linalg.splu`.
    """

    def __init__(self, A, is_symmetric=None, is_positive_definite=False, is_hermitian=None, check_accuracy=False, check_rtol=1e-6, check_atol=0, **kwargs):
        super().__init__(A, is_symmetric=is_symmetric, is_positive_definite=is_positive_definite, is_hermitian=is_hermitian, check_accuracy=check_accuracy, check_rtol=check_rtol, check_atol=check_atol)
        self.kwargs = _valid_kwargs_for_func(splu, **kwargs)
        self.solver = splu(sp.csc_matrix(self.A), **self.kwargs)

    def transpose(self):
        if self.is_symmetric:
            return self
        return super().transpose()

    def _solve_single(self, rhs):
        rhs = rhs.astype(self.dtype)
        return self.solver.solve(rhs, trans='T' if self._transposed else 'N')

    _solve_multiple = _solve_single

    def _refactor(self, A):
        self._A = A
        self.solver = splu(sp.csc_matrix(self._shared.A), **self.kwargs)
