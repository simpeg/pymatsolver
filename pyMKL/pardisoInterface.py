from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

from . import MKLlib
from ctypes import POINTER, c_int, c_longlong

pardisoinit = MKLlib.pardisoinit

pardisoinit.argtypes = [POINTER(c_longlong),
                        POINTER(c_int),
                        POINTER(c_int)]
pardisoinit.restype = None

pardiso = MKLlib.pardiso
pardiso.argtypes = [POINTER(c_longlong), # pt
                    POINTER(c_int),      # maxfct
                    POINTER(c_int),      # mnum
                    POINTER(c_int),      # mtype
                    POINTER(c_int),      # phase
                    POINTER(c_int),      # n
                    POINTER(None),       # a
                    POINTER(c_int),      # ia
                    POINTER(c_int),      # ja
                    POINTER(c_int),      # perm
                    POINTER(c_int),      # nrhs
                    POINTER(c_int),      # iparm
                    POINTER(c_int),      # msglvl
                    POINTER(None),       # b
                    POINTER(None),       # x
                    POINTER(c_int)]      # error)
pardiso.restype = None
