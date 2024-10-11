from scipy.sparse.linalg import spsolve, splu

from ..wrappers import WrapDirect
from .pardiso import Pardiso
from .mumps import Mumps

Solver = WrapDirect(spsolve, factorize=False, name="Solver")
SolverLU = WrapDirect(splu, factorize=True, name="SolverLU")

__all__ = ["Solver", "SolverLU", "Pardiso", "Mumps"]
