import numpy as np
import scipy.sparse as sp
from ctypes import CDLL, cdll, RTLD_GLOBAL
from ctypes import POINTER, byref, c_int, c_longlong

path = 'libmkl_intel_lp64.dylib'
MKLlib = CDLL(path, RTLD_GLOBAL)

from pardisoInterface import pardisoinit, pardiso
from pardisoSolver import pardisoSolver
