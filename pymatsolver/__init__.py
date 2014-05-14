#!/usr/bin/env python
from Base import SolverException, DiagonalSolver


SolverHelp = {}
AvailableSolvers = {
    "DiagonalSolver":   True,
    "TriangleFortran":  False,
    "TrianglePython":   False,
    "Mumps":            False,
}


try:
    from Triangle.TriangleFortran import ForwardSolver, BackwardSolver
    from Triangle.TrianglePython import ForwardSolver as _ForwardSolver, BackwardSolver as _BackwardSolver
    AvailableSolvers['TriangleFortran'] = True
except ImportError:
    from Triangle.TrianglePython import ForwardSolver, BackwardSolver
    AvailableSolvers['TrianglePython'] = True
    SolverHelp['TriangleFortran'] = """Could not compile the Triangle Solvers
Try something like:


    cd pymatsolver/Triangle
    make

"""


try:
    from Mumps import MumpsSolver
    AvailableSolvers['Mumps'] = True
except ImportError, e:
    SolverHelp['Mumps'] = """Mumps is not working.

Ensure that you have Mumps installed, and know where the path to it is.

Try something like:

    cd pymatsolver/Mumps
    make

When that doesn't work, you may need to edit the make file, to modify the
path to the mumps installation, or any other compiler flags.

If you find a good way of doing it, please share.

"""


