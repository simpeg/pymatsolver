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

  Forward
  Backward

Iterative Solvers
=================

.. autosummary::
  :toctree: generated/

  SolverCG
  BicgJacobi

Direct Solvers
==============

.. autosummary::
  :toctree: generated/

  Solver
  SolverLU
  Pardiso
"""
from pymatsolver.solvers import Diagonal, Forward, Backward
from pymatsolver.wrappers import WrapDirect
from pymatsolver.wrappers import WrapIterative
from pymatsolver.wrappers import Solver
from pymatsolver.wrappers import SolverLU
from pymatsolver.wrappers import SolverCG
from pymatsolver.wrappers import SolverBiCG
from pymatsolver.iterative import BicgJacobi

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

try:
    from pymatsolver.direct import Pardiso
    AvailableSolvers['Pardiso'] = True
    PardisoSolver = Pardiso  # backwards compatibility
except ImportError:
    SolverHelp['Pardiso'] = """Pardiso is not working

Ensure that you have pydiso installed, which may also require Python
to be installed through conda.
"""

# try:
#     from pymatsolver.mumps import Mumps
#     AvailableSolvers['Mumps'] = True
#     MumpsSolver = Mumps  # backwards compatibility
# except Exception:
#     SolverHelp['Mumps'] = """Mumps is not working.
#
# Ensure that you have Mumps installed, and know where the path to it is.
# Try something like:
#     cd pymatsolver/mumps
#     make
#
# When that doesn't work, you may need to edit the make file, to modify the
# path to the mumps installation, or any other compiler flags.
# If you find a good way of doing it, please share.
#
# brew install mumps --with-scotch5 --without-mpi
# mpicc --showme
# """

__version__ = '0.2.0'
__author__ = 'Rowan Cockett'
__license__ = 'MIT'
__copyright__ = 'Copyright 2017 Rowan Cockett'
