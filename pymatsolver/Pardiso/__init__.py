from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from pymatsolver.Base import BaseSolver
from pyMKL import pardisoSolver


class PardisoSolver(BaseSolver):
    """

    Pardiso Solver

    Wrapped by David Marchant

        https://github.com/dwfmarchant/pyMKL


    documentation::

        http://www.pardiso-project.org/


    """

    transpose = False
    symmetric = False
    isfactored = None
    dtype = None
    solver = pardisoSolver
    knownmatType = {"RealSymPD": 2, "RealSym": 1, "RealNonSym": 11, "RealSymID"
                    "CompSym": 3, "CompNonSym": 13,
                    "CompHermPD": 4, "CompHermID": -4}
    mtype = None

    def __init__(self, A, symmetric=True, mtype=None):
        self.A = A
        self.symmetric = symmetric
        self.dtype = A.dtype
        if mtype is None:
            self.solver = pardisoSolver(A.tocsc(), mtype=self._funhandle())
        else:
            self.solver = pardisoSolver(A.tocsc(),
                                        mtype=self.knownmatType[mtype])
        self.solver.factor()
        self.isfactored = True

    def _funhandle(self):
        """
            Set basic matrix type:

                2: real symmetric postivie definite
                11: real nonsymmetric
                6: complex symmetric
                13: complex nonsymmetric

        """
        if self.dtype == float:
            if self.symmetric:
                # real symmetric postivie definite
                return 2
            else:
                # real nonsymmetric
                return 11
        elif self.dtype == complex:
            if self.symmetric:
                # complex symmetric
                return 6
            else:
                # complex nonsymmetric
                return 13

    def factor(self):
        if self.isfactored is not True:
            self.solver.factor()
            self.isfactored = True

    def _solveM(self, rhs):
        self.factor()
        sol = self.solver.solve(rhs)
        return sol

    _solve1 = _solveM

    def clean(self):
        self.solver.clear()
