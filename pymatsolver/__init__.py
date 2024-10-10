"""
===
API
===
.. currentmodule:: pymatsolver

.. autosummary::
  :toctree: generated/

  solvers.Base

Basic Solvers
=============

Diagonal
--------
.. autosummary::
  :toctree: generated/

  Diagonal

Triangular
----------
.. autosummary::
  :toctree: generated/

  Triangle
  Forward
  Backward

Iterative Solvers
=================

.. autosummary::
  :toctree: generated/

  SolverCG
  BiCGJacobi

Direct Solvers
==============

.. autosummary::
  :toctree: generated/

  Solver
  SolverLU
  Pardiso
  Mumps
"""

SolverHelp = {}
AvailableSolvers = {
    "Diagonal": True,
    "Solver": True,
    "SolverLU": True,
    "SolverCG": True,
    "Triangle": True,
    "Pardiso": False,
    "Mumps": False
}

# Simple solvers
from .solvers import Diagonal, Triangle, Forward, Backward
from .wrappers import wrap_direct, WrapDirect
from .wrappers import wrap_iterative, WrapIterative

# Scipy Iterative solvers
from .iterative import SolverCG
from .iterative import SolverBiCG
from .iterative import BiCGJacobi

# Scipy direct solvers
from .direct import Solver
from .direct import SolverLU

from .solvers import PymatsolverAccuracyError

BicgJacobi = BiCGJacobi  # backwards compatibility

try:
    from .direct import Pardiso
    AvailableSolvers['Pardiso'] = True
    PardisoSolver = Pardiso  # backwards compatibility
except ImportError:
    SolverHelp['Pardiso'] = """Pardiso is not working

Ensure that you have pydiso installed, which may also require Python
to be installed through conda.
"""

try:
    from .direct import Mumps
    AvailableSolvers['Mumps'] = True
except ImportError:
    SolverHelp['Mumps'] = """Mumps is not working.

Ensure that you have python-mumps installed, which may also require Python
to be installed through conda.
"""

__author__ = 'SimPEG Team'
__license__ = 'MIT'
__copyright__ = '2013 - 2024, SimPEG Team, https://simpeg.xyz'

from importlib.metadata import version, PackageNotFoundError

# Version
try:
    # - Released versions just tags:       0.8.0
    # - GitHub commits add .dev#+hash:     0.8.1.dev4+g2785721
    # - Uncommitted changes add timestamp: 0.8.1.dev4+g2785721.d20191022
    __version__ = version("pymatsolver")
except PackageNotFoundError:
    # If it was not installed, then we don't know the version. We could throw a
    # warning here, but this case *should* be rare. discretize should be
    # installed properly!
    from datetime import datetime

    __version__ = "unknown-" + datetime.today().strftime("%Y%m%d")