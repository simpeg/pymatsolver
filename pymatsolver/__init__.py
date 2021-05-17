#!/usr/bin/env python
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

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
except Exception:
    SolverHelp['Pardiso'] = """Pardiso is not working

Ensure that you have pyMKL installed, which may also require Python
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

__version__ = '0.1.3'
__author__ = 'Rowan Cockett'
__license__ = 'MIT'
__copyright__ = 'Copyright 2017 Rowan Cockett'
