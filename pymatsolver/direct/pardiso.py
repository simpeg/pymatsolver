from pymatsolver.solvers import Base
try:
    from pydiso.mkl_solver import MKLPardisoSolver
    from pydiso.mkl_solver import set_mkl_pardiso_threads, get_mkl_pardiso_max_threads
    _available = True
except ImportError:
    _available = False

class Pardiso(Base):
    """The Pardiso direct solver.

    This solver uses the `pydiso` Intel MKL wrapper to factorize a sparse matrix, and use that
    factorization for solving.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Matrix to solve with.
    n_threads : int, optional
        Number of threads to use for the `Pardiso` routine in Intel's MKL.
    is_symmetric : bool, optional
        Whether the matrix is symmetric. By default, it will perform some simple tests to check for symmetry, and
        default to ``False`` if those fail.
    is_positive_definite : bool, optional
        Whether the matrix is positive definite.
    is_hermitian : bool, optional
        Whether the matrix is hermitian. By default, it will perform some simple tests to check, and default to
        ``False`` if those fail.
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
        Extra keyword arguments. If there are any left here a warning will be raised.
    """

    _transposed = False

    def __init__(self, A, n_threads=None, is_symmetric=None, is_positive_definite=False, is_hermitian=None, check_accuracy=False, check_rtol=1e-6, check_atol=0, accuracy_tol=None, **kwargs):
        if not _available:
            raise ImportError("Pardiso solver requires the pydiso package to be installed.")
        super().__init__(A, is_symmetric=is_symmetric, is_positive_definite=is_positive_definite, is_hermitian=is_hermitian, check_accuracy=check_accuracy, check_rtol=check_rtol, check_atol=check_atol, accuracy_tol=accuracy_tol, **kwargs)
        self.solver = MKLPardisoSolver(
            self.A,
            matrix_type=self._matrixType(),
            factor=False
        )
        if n_threads is not None:
            self.n_threads = n_threads

    def _matrixType(self):
        """
            Set basic matrix type:

            Real::

                 1:  structurally symmetric
                 2:  symmetric positive definite
                -2:  symmetric indefinite
                11:  nonsymmetric

            Complex::

                 6:  symmetric
                 4:  hermitian positive definite
                -4:  hermitian indefinite
                 3:  structurally symmetric
                13:  nonsymmetric

        """
        if self.is_real:
            if self.is_symmetric:
                if self.is_positive_definite:
                    return 2
                else:
                    return -2
            else:
                return 11
        else:
            if self.is_symmetric:
                return 6
            elif self.is_hermitian:
                if self.is_positive_definite:
                    return 4
                else:
                    return -4
            else:
                return 13

    def factor(self, A=None):
        """(Re)factor the A matrix.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
            The matrix to be factorized. If a previous factorization has been performed, this will
            reuse the previous factorization's analysis.
        """
        if A is not None and self.A is not A:
            self._A = A
            self.solver.refactor(self.A)

    def _solve_multiple(self, rhs):
        sol = self.solver.solve(rhs, transpose=self._transposed)
        return sol

    def transpose(self):
        trans_obj = Pardiso.__new__(Pardiso)
        trans_obj._A = self.A
        for attr, value in self.get_attributes().items():
            setattr(trans_obj, attr, value)
        trans_obj.solver = self.solver
        trans_obj._transposed = not self._transposed
        return trans_obj

    @property
    def n_threads(self):
        """Number of threads to use for the Pardiso solver routine.

        This property is global to all Pardiso solver objects for a single python process.

        Returns
        -------
        int
        """
        return get_mkl_pardiso_max_threads()

    @n_threads.setter
    def n_threads(self, n_threads):
        set_mkl_pardiso_threads(n_threads)

    _solve_single = _solve_multiple
