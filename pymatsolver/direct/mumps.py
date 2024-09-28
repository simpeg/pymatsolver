from pymatsolver.solvers import Base
from mumps import Context

class Mumps(Base):
    """
    Mumps solver
    """
    _transposed = False

    def __init__(self, A, ordering=None, **kwargs):
        super().__init__(A, **kwargs)
        if ordering is None:
            ordering = "metis"
        self.ordering = ordering
        self.solver = Context()
        self._set_A(self.A)

    def _set_A(self, A):
        self.solver.set_matrix(
            A,
            symmetric=self.is_symmetric,
            # positive_definite=self.is_positive_definite  # doesn't (yet) support setting positive definiteness
        )

    @property
    def ordering(self):
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

    def transpose(self):
        trans_obj = Mumps.__new__(Mumps)
        trans_obj._A = self.A
        for attr, value in self.get_attributes().items():
            setattr(trans_obj, attr, value)
        trans_obj.solver = self.solver
        trans_obj._transposed = not self._transposed
        return trans_obj

    def factor(self, A=None):
        reuse_analysis = False
        if A is not None:
            self._set_A(A)
            self._A = A
            # if it was previously factored then re-use the analysis.
            reuse_analysis = self._factored
        if not self._factored:
            pivot_tol = 0.0 if self.is_positive_definite else 0.01
            self.solver.factor(
                ordering=self.ordering, reuse_analysis=reuse_analysis, pivot_tol=pivot_tol
            )

    def _solve_multiple(self, rhs):
        self.factor()
        if self._transposed:
            self.solver.mumps_instance.icntl[9] = 0
        else:
            self.solver.mumps_instance.icntl[9] = 1
        sol = self.solver.solve(rhs)
        return sol

    _solve_single = _solve_multiple