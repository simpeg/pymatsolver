from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from pymatsolver.solvers import Base
from pyMKL import pardisoSolver as _pardisoSolver

from pyMKL import mkl_set_num_threads, mkl_get_max_threads

import os
_omp_threads = os.environ.get('OMP_NUM_THREADS')
if _omp_threads is not None:
    _max_threads = _omp_threads
else:
    _max_threads = mkl_get_max_threads()


def _set_threads(n_threads):
    global _n_threads
    try:
        mkl_set_num_threads(n_threads)
    except TypeError:
        raise TypeError('n_threads must be an Integer')
    _n_threads = n_threads


_set_threads(_max_threads)


class Pardiso(Base):
    """

    Pardiso Solver

    Wrapped by David Marchant

        https://github.com/dwfmarchant/pyMKL


    documentation::

        http://www.pardiso-project.org/


    """

    isfactored = False

    def __init__(self, A, **kwargs):
        A = A.tocsr()
        if not A.has_sorted_indices:
            A.sort_indices()
        self.A = A
        self.set_kwargs(**kwargs)
        self.solver = _pardisoSolver(
            A,
            mtype=self._martixType()
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

    def factor(self):
        if self.isfactored is not True:
            self.solver.factor()
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
        return _n_threads

    @n_threads.setter
    def n_threads(self, n_threads):
        _set_threads(n_threads)

    _solve1 = _solveM

    def clean(self):
        self.solver.clear()
