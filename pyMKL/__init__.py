from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from future import standard_library
standard_library.install_aliases()

import platform
import numpy as np
import scipy.sparse as sp
from .loadMKL import _loadMKL

MKLlib = _loadMKL()

from .pardisoInterface import pardisoinit, pardiso
from .pardisoSolver import pardisoSolver
