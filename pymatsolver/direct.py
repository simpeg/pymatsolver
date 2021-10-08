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

    isfactored = False

    def __init__(self, A, **kwargs):
        self.A = A
        self.set_kwargs(**kwargs)
        self.solver = MKLPardisoSolver(
            self.A,
            matrix_type=self._martixType(),
            factor=False
        )

    def _martixType(self):
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
        if self.isfactored is not True:
            if A is None:
                A = self.A
            self.solver.refactor(A)
            self.isfactored = True

    def _solveM(self, rhs):
        self.factor()
        sol = self.solver.solve(rhs)
        return sol

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

    _solve1 = _solveM
