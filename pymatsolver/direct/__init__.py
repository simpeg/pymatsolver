from ..wrappers import WrapDirect
from scipy.sparse.linalg import spsolve, splu

Solver = WrapDirect(spsolve, factorize=False, name="Solver")
SolverLU = WrapDirect(splu, factorize=True, name="SolverLU")

__all__ = ["Solver", "SolverLU"]
try:
    from .pardiso import Pardiso
    __all__ += ["Pardiso"]
except ImportError:
    pass

try:
    from .mumps import Mumps
    __all__ += ["Mumps"]
except ImportError:
    pass
