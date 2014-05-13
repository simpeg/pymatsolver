#!/usr/bin/env python

SolverHelp = {}
AvailableSolvers = []

try:
    from Mumps import MumpsSolver
    AvailableSolvers += ['Mumps']
except ImportError, e:
    SolverHelp['Mumps'] = 'Mumps'
