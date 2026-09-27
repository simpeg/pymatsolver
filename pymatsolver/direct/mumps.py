from pymatsolver.solvers import _SharedFactorBase
try:
    from mumps import Context
    _available = True
except ImportError:
    Context = None
    _available = False

class Mumps(_SharedFactorBase):
    """The MUMPS direct solver.

    This solver uses the python-mumps wrappers to factorize a sparse matrix, and use that factorization for solving.

    Parameters
    ----------
    A
        Matrix to solve with.
    ordering : str, default 'metis'
        Which ordering algorithm to use. See the `python-mumps` documentation for more details.
    is_symmetric : bool, optional
        Whether the matrix is symmetric. By default, it will perform some simple tests to check for symmetry, and
        default to ``False`` if those fail.
    is_positive_definite : bool, optional
        Whether the matrix is positive definite.
    check_accuracy : bool, optional
        Whether to check the accuracy of the solution.
    check_rtol : float, optional
        The relative tolerance to check against for accuracy.
    check_atol : float, optional
        The absolute tolerance to check against for accuracy.
    **kwargs
        Extra keyword arguments. If there are any left here a warning will be raised.
    """

    def __init__(self, A, ordering=None, is_symmetric=None, is_positive_definite=False, check_accuracy=False, check_rtol=1e-6, check_atol=0, **kwargs):
        if not _available:
            raise ImportError(
                "The Mumps solver requires the python-mumps package to be installed."
            )
        is_hermitian = kwargs.pop('is_hermitian', False)
        super().__init__(A, is_symmetric=is_symmetric, is_positive_definite=is_positive_definite, is_hermitian=is_hermitian, check_accuracy=check_accuracy, check_rtol=check_rtol, check_atol=check_atol, **kwargs)
        if ordering is None:
            ordering = "metis"
        self.ordering = ordering
        self.solver = Context()
        self._set_A(self.A)

    def _set_A(self, A):
        self.solver.set_matrix(
            A,
            symmetric=self.is_symmetric,
        )

    @property
    def ordering(self):
        """The ordering algorithm to use.

        Returns
        -------
        str
        """
        return self._ordering

    @ordering.setter
    def ordering(self, value):
        self._ordering = str(value)

    @property
    def _factored(self):
        return self.solver.factored

    def get_attributes(self):
        attrs = super().get_attributes()
        attrs['ordering'] = self.ordering
        return attrs

    def _do_factor(self, reuse_analysis):
        pivot_tol = 0.0 if self.is_positive_definite else 0.01
        self.solver.factor(
            ordering=self.ordering, reuse_analysis=reuse_analysis, pivot_tol=pivot_tol
        )

    def _refactor(self, A):
        # if it was previously factored then re-use the analysis.
        reuse_analysis = self._factored
        self._A = A
        self._set_A(self._shared.A)
        self._do_factor(reuse_analysis)

    def _factor(self):
        if not self._factored:
            self._do_factor(reuse_analysis=False)

    def _solve_multiple(self, rhs):
        self._factor()
        if self._transposed:
            self.solver.mumps_instance.icntl[9] = 0
        else:
            self.solver.mumps_instance.icntl[9] = 1
        sol = self.solver.solve(rhs)
        return sol

    _solve_single = _solve_multiple