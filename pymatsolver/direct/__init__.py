from scipy.sparse.linalg import spsolve

from ..wrappers import WrapDirect
from .lu import SolverLU
from .pardiso import Pardiso
from .mumps import Mumps

Solver = WrapDirect(spsolve, factorize=False, name="Solver")

__all__ = ["Solver", "SolverLU", "Pardiso", "Mumps"]
