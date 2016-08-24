from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np


class SolverException(Exception):
    pass


class BaseSolver(object):

    def __init__(self, A):
        self.A = A.tocsr()

    def setKwargs(self, ignore=None,  **kwargs):
        """
            Sets key word arguments (kwargs) that are present in the object,
            throw an error if they don't exist.
        """
        if ignore is None:
            ignore = []
        for attr in kwargs:
            if attr in ignore:
                continue
            if hasattr(self, attr):
                setattr(self, attr, kwargs[attr])
            else:
                raise Exception('{0!s} attr is not recognized'.format(attr))

    @property
    def _transposeClass(self):
        return self.__class__

    @property
    def T(self):
        if self._transposeClass is None:
            raise Exception(
                'The transpose for the {} class is not possible.'.format(
                    self.__name__
                )
            )
        newS = self._transposeClass(self.A.T)
        return newS

    check_accuracy = False
    accuracy_tol = 1e-6

    def _compute_accuracy(self, rhs, x):
        nrm = np.linalg.norm(np.ravel(self.A*x - rhs), np.inf)
        nrm_rhs = np.linalg.norm(np.ravel(rhs), np.inf)
        if nrm_rhs > 0:
            nrm /= nrm_rhs
        if nrm > self.accuracy_tol:
            msg = 'Accuracy on solve is above tolerance: {0:e} > {1:e}'.format(
                nrm, self.accuracy_tol
            )
            raise SolverException(msg)

    def _solve(self, rhs):

        n = self.A.shape[0]
        assert rhs.size % n == 0, 'Incorrect shape of rhs.'
        nrhs = rhs.size // n

        if len(rhs.shape) == 1 or rhs.shape[1] == 1:
            x = self._solve1(rhs)
        else:
            x = self._solveM(rhs)

        if self.check_accuracy:
            self._compute_accuracy(rhs, x)

        if nrhs == 1:
            return x.flatten()
        elif nrhs > 1:
            return x.reshape((n, nrhs), order='F')

    def clean(self):
        print("base clean")
        pass

    def __mul__(self, val):
        if type(val) is np.ndarray:
            return self._solve(val)
        raise TypeError('Can only multiply by a numpy array.')

    @property
    def is_real(self):
        return self.A.dtype == float

    @property
    def is_symmetric(self):
        return getattr(self, '_is_symmetric', False)

    @is_symmetric.setter
    def is_symmetric(self, value):
        if value is True:
            self.is_structurally_symmetric = True
        self._is_symmetric = value

    @property
    def is_structurally_symmetric(self):
        return getattr(self, '_is_structurally_symmetric', False)

    @is_structurally_symmetric.setter
    def is_structurally_symmetric(self, value):
        self._is_structurally_symmetric = value

    @property
    def is_hermitian(self):
        if self.is_real and self.is_symmetric:
            return True
        else:
            return getattr(self, '_is_hermitian', False)

    @is_hermitian.setter
    def is_hermitian(self, value):
        self._is_hermitian = value

    @property
    def is_positive_definite(self):
        return getattr(self, '_is_positive_definite', False)

    @is_positive_definite.setter
    def is_positive_definite(self, value):
        self._is_positive_definite = value


class DiagonalSolver(BaseSolver):

    _transposeClass = None

    def __init__(self, A):
        self.A = A
        self._diagonal = A.diagonal()

    def _solve1(self, rhs):
        return rhs.flatten()/self._diagonal

    def _solveM(self, rhs):
        n = self.A.shape[0]
        nrhs = rhs.size // n
        return rhs/self._diagonal.repeat(nrhs).reshape((n, nrhs))
