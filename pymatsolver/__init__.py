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
    "Mumps": False,
    "Pardiso": False,
}


try:
    from pymatsolver.Mumps import Mumps
    AvailableSolvers['Mumps'] = True
except ImportError:
    SolverHelp['Mumps'] = """Mumps is not working.


Ensure that you have Mumps installed, and know where the path to it is.

Try something like:

    cd pymatsolver/Mumps
    make

When that doesn't work, you may need to edit the make file, to modify the
path to the mumps installation, or any other compiler flags.

If you find a good way of doing it, please share.

"""

try:
    from pymatsolver.direct import Pardiso
    AvailableSolvers['Pardiso'] = True
except ImportError:
    SolverHelp['Pardiso'] = """PardisoSolver is not working."""
