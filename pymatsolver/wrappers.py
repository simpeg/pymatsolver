from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from scipy.sparse import linalg
from pymatsolver.solvers import Base


def WrapDirect(fun, factorize=True, name=None):
    """Wraps a direct Solver.

    ::

        Solver   = pymatsolver.WrapDirect(sp.linalg.spsolve, factorize=False)
        SolverLU = pymatsolver.WrapDirect(sp.linalg.splu, factorize=True)

    """

    def __init__(self, A, **kwargs):
        self.A = A.tocsc()
        self.kwargs = kwargs
        if factorize:
            self.solver = fun(self.A, **kwargs)

    def _solve1(self, rhs):
        rhs = rhs.flatten()

        if rhs.dtype is np.dtype('O'):
            rhs = rhs.astype(type(rhs[0]))

        if factorize:
            X = self.solver.solve(rhs, **self.kwargs)
        else:
            X = fun(self.A, rhs, **self.kwargs)

        return X

    def _solveM(self, rhs):
        if rhs.dtype is np.dtype('O'):
            rhs = rhs.astype(type(rhs[0, 0]))

        X = np.empty_like(rhs)

        for i in range(rhs.shape[1]):
            if factorize:
                X[:, i] = self.solver.solve(rhs[:, i])
            else:
                X[:, i] = fun(self.A, rhs[:, i], **self.kwargs)

        return X

    def clean(self):
        if factorize and hasattr(self.solver, 'clean'):
            return self.solver.clean()

    return type(
        str(name if name is not None else fun.__name__),
        (Base,),
        {
            "__init__": __init__,
            "_solve1": _solve1,
            "_solveM": _solveM,
            "clean": clean,
        }
    )


def WrapIterative(fun, check_accuracy=True, accuracyTol=1e-5, name=None):
    """
    Wraps an iterative Solver.

    ::

        SolverCG = pymatsolver.WrapIterative(sp.linalg.cg)

    """

    def __init__(self, A, **kwargs):
        self.A = A
        self.kwargs = kwargs

    def _solve1(self, rhs):

        rhs = rhs.flatten()
        out = fun(self.A, rhs, **self.kwargs)
        if type(out) is tuple and len(out) == 2:
            # We are dealing with scipy output with an info!
            X = out[0]
            self.info = out[1]
        else:
            X = out
        return X

    def _solveM(self, rhs):

        X = np.empty_like(rhs)
        for i in range(rhs.shape[1]):
            out = fun(self.A, rhs[:, i], **self.kwargs)
            if type(out) is tuple and len(out) == 2:
                # We are dealing with scipy output with an info!
                X[:, i] = out[0]
                self.info = out[1]
            else:
                X[:, i] = out

        return X

    return type(
        str(name if name is not None else fun.__name__),
        (Base,),
        {
            "__init__": __init__,
            "_solve1": _solve1,
            "_solveM": _solveM,
        }
    )

Solver = WrapDirect(linalg.spsolve, factorize=False, name="Solver")
SolverLU = WrapDirect(linalg.splu, factorize=True, name="SolverLU")
SolverCG = WrapIterative(linalg.cg, name="SolverCG")
SolverBiCG = WrapIterative(linalg.bicgstab, name="SolverBiCG")

