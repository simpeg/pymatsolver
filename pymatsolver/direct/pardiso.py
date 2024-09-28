from pymatsolver.solvers import Base
from pydiso.mkl_solver import MKLPardisoSolver
from pydiso.mkl_solver import set_mkl_pardiso_threads, get_mkl_pardiso_max_threads

class Pardiso(Base):
    """
    Pardiso Solver

        https://github.com/simpeg/pydiso


    documentation::

        http://www.pardiso-project.org/
    """

    _transposed = False

    def __init__(self, A, n_threads=None, **kwargs):
        super().__init__(A, **kwargs)
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
        """
        Number of threads to use for the Pardiso solver routine. This property
        is global to all Pardiso solver objects for a single python process.
        """
        return get_mkl_pardiso_max_threads()

    @n_threads.setter
    def n_threads(self, n_threads):
        set_mkl_pardiso_threads(n_threads)

    _solve_single = _solve_multiple
