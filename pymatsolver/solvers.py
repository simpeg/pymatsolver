import numpy as np
import warnings


class PymatsolverAccuracyError(Exception):
    pass


class Base():
    _accuracy_tol = 1e-6
    _check_accuracy = False

    def __init__(self, A):
        self.A = A.tocsr()

    @property
    def check_accuracy(self):
        """check the accuracy of the solve?"""
        return self._check_accuracy

    @check_accuracy.setter
    def check_accuracy(self, value):
        # Do this to do a lazy cast to boolean
        test = (value == True)
        if not isinstance(test, bool):
            raise TypeError("check_accuracy must be either True or False")
        self._check_accuracy = test

    @property
    def accuracy_tol(self):
        "tolerance on the accuracy of the solver"
        return self._accuracy_tol

    @accuracy_tol.setter
    def accuracy_tol(self, value):
        try:
            self._accuracy_tol = float(value)
        except Exception:
            raise TypeError("accuracy_tol must be a float value")

    def set_kwargs(self, ignore=None,  **kwargs):
        """
            Sets key word arguments (kwargs) that are present in the object,
            throw a warning if they don't exist.
        """
        if ignore is None:
            ignore = []
        for attr in kwargs:
            if attr in ignore:
                continue
            if hasattr(self, attr):
                setattr(self, attr, kwargs[attr])
            else:
                warnings.warn('{0!s} attr is not recognized and will be unused'.format(attr))

    @property
    def _transposeClass(self):
        return self.__class__

    @property
    def T(self):
        "The transpose operator for this class"
        if self._transposeClass is None:
            raise NotImplementedError(
                'The transpose for the {} class is not possible.'.format(
                    self.__name__
                )
            )
        newS = self._transposeClass(self.A.T)
        return newS

    def _compute_accuracy(self, rhs, x):
        nrm = np.linalg.norm(np.ravel(self.A*x - rhs), np.inf)
        nrm_rhs = np.linalg.norm(np.ravel(rhs), np.inf)
        if nrm_rhs > 0:
            nrm /= nrm_rhs
        if nrm > self.accuracy_tol:
            msg = 'Accuracy on solve is above tolerance: {0:e} > {1:e}'.format(
                nrm, self.accuracy_tol
            )
            raise PymatsolverAccuracyError(msg)

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
        pass

    def __del__(self):
        """Destruct to call clean when object is garbage collected."""
        try:
            self.clean()
        except:
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
        self._is_symmetric = value

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


class Diagonal(Base):

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


class Forward(Base):

    _transposeClass = None

    def __init__(self, A):
        self.A = A.tocsr()

    def _solveM(self, rhs):

        vals = self.A.data
        rowptr = self.A.indptr
        colind = self.A.indices
        x = np.empty_like(rhs)
        for i in range(self.A.shape[0]):
            ith_row = vals[rowptr[i]:rowptr[i+1]]
            cols = colind[rowptr[i]:rowptr[i+1]]
            x_vals = x[cols]
            x[i] = (rhs[i] - np.dot(ith_row[:-1], x_vals[:-1])) / ith_row[-1]
        return x

    _solve1 = _solveM


class Backward(Base):

    _transposeClass = None

    def __init__(self, A):
        self.A = A.tocsr()

    def _solveM(self, rhs):

        vals = self.A.data
        rowptr = self.A.indptr
        colind = self.A.indices
        x = np.empty_like(rhs)
        for i in reversed(range(self.A.shape[0])):
            ith_row = vals[rowptr[i]:rowptr[i+1]]
            cols = colind[rowptr[i]:rowptr[i+1]]
            x_vals = x[cols]
            x[i] = (rhs[i] - np.dot(ith_row[1:], x_vals[1:])) / ith_row[0]
        return x

    _solve1 = _solveM
