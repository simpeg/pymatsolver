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

from pymatsolver.iterative import BicgJacobi


SolverHelp = {}
AvailableSolvers = {
    "Diagonal": True,
    "Solver": True,
    "SolverLU": True,
    "SolverCG": True,
    "Triangle": True,
    "Pardiso": False
}

try:
    from pymatsolver.direct import Pardiso
    AvailableSolvers['Pardiso'] = True
except ImportError:
    SolverHelp['Pardiso'] = """PardisoSolver is not working."""

__version__   = '0.0.2'
__author__    = 'Rowan Cockett'
__license__   = 'MIT'
__copyright__ = 'Copyright 2017 Rowan Cockett'
