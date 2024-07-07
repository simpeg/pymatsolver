from pymatsolver.solvers import Base
from mumps import Context

class Mumps(Base):
    """
    Mumps solver
    """

    _factored = False
    _transposed = False
    ordering = ''

    def __init__(self, A, context=None, **kwargs):
        self.set_kwargs(**kwargs)
        if context is None:
            self.solver = Context()
            self._analyzed = False
            self._set_A(A)
        else:
            self.solver = context
        self.A = A

    def _set_A(self, A):
        self.solver.set_matrix(
            A,
            symmetric=self.is_symmetric,
            # positive_definite=self.is_positive_definite  # doesn't (yet) support setting positive definiteness
        )

    @property
    def ordering(self):
        return getattr(self, '_ordering', "metis")

    @ordering.setter
    def ordering(self, value):
        self._ordering = value

    @property
    def _factored(self):
        return self.solver.factored

    @property
    def transpose(self):
        trans_obect = Mumps(
            self.A,
            self.solver,
            is_symmetric=self.is_symmetric,
            is_positive_definite=self.is_positive_definite,
            ordering=self.ordering,
        )
        trans_obect._transposed = not self._transposed

        return trans_obect

    T = transpose

    def factor(self, A=None):
        reuse_analysis = False
        if A is not None:
            self._set_A(A)
            self.A = A
            # if it was previously factored then re-use the analysis.
            reuse_analysis = self._factored
        if not self._factored:
            pivot_tol = 0.0 if self.is_positive_definite else 0.01
            self.solver.factor(
                ordering=self.ordering, reuse_analysis=reuse_analysis, pivot_tol=pivot_tol
            )

    def _solveM(self, rhs):
        self.factor()
        if self._transposed:
            self.solver.mumps_instance.icntl[9] = 0
        else:
            self.solver.mumps_instance.icntl[9] = 1
        sol = self.solver.solve(rhs)
        return sol

    _solve1 = _solveM