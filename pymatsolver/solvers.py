import numpy as np
import warnings
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve_triangular
from scipy.linalg import issymmetric, ishermitian
from abc import ABC, abstractmethod
import copy


class PymatsolverAccuracyError(Exception):
    pass


class Base(ABC):

    def __init__(
            self, A, is_symmetric=None, is_positive_definite=False, is_hermitian=None, check_accuracy=False, check_rtol=1e-6, check_atol=0, accuracy_tol=None, **kwargs
    ):
        # don't make any assumptions on what A is, let the individual solvers handle that
        shape = A.shape
        if len(shape) != 2:
            raise ValueError("A must be 2-dimensional.")
        if shape[0] != shape[1]:
            raise ValueError("A is not a square matrix.")
        self._A = A
        self._dtype = np.dtype(A.dtype)

        if accuracy_tol is not None:
            warnings.warn("accuracy_tol is deprecated, use check_rtol and check_atol.", FutureWarning)
            check_rtol = accuracy_tol

        self.check_accuracy = check_accuracy
        self.check_rtol = check_rtol
        self.check_atol = check_atol

        # do some symmetry checks that likely speed up the defualt solve operation
        if is_symmetric is None:
            if sp.issparse(A):
                is_symmetric = (A.T != A).nnz == 0
            else:
                is_symmetric = issymmetric(A)
        self.is_symmetric = is_symmetric
        if is_hermitian is None:
            if self.is_real:
                is_hermitian = self.is_symmetric
            else:
                if sp.issparse(A):
                    is_hermitian = (A.T.conjugate() != A).nnz == 0
                else:
                    is_hermitian = ishermitian(A)

        self.is_hermitian = is_hermitian

        # Can't check for positive definiteness until it is factored.
        # This should be defaulted to False. If the user knows ahead of time that it is positive definite
        # they should set this to be true.
        self.is_positive_definite = is_positive_definite

        if kwargs:
            warnings.warn(f"Unused keyword arguments for {self.__class__.__name__}: {kwargs.keys()}")

    @property
    def A(self):
        return self._A

    @property
    def dtype(self):
        return self._dtype

    @property
    def is_real(self):
        return np.issubdtype(self.A.dtype, np.floating)

    @property
    def is_symmetric(self):
        return self._is_symmetric

    @is_symmetric.setter
    def is_symmetric(self, value):
        if isinstance(value, bool):
            self._is_symmetric = value
        else:
            raise TypeError("is_symmetric must be a boolean.")

    @property
    def is_hermitian(self):
        if self.is_real and self.is_symmetric:
            return True
        else:
            return self._is_hermitian

    @is_hermitian.setter
    def is_hermitian(self, value):
        if isinstance(value, bool):
            self._is_hermitian = value
        else:
            raise TypeError("is_hermitian must be a boolean.")

    @property
    def is_positive_definite(self):
        return self._is_positive_definite

    @is_positive_definite.setter
    def is_positive_definite(self, value):
        if isinstance(value, bool):
            self._is_positive_definite = value
        else:
            raise TypeError("is_positive_definite must be a boolean.")

    @property
    def check_accuracy(self):
        """check the accuracy of the solve?"""
        return self._check_accuracy

    @check_accuracy.setter
    def check_accuracy(self, value):
        if isinstance(value, bool):
            self._check_accuracy = value
        else:
            raise TypeError("check_accuracy must be a boolean.")

    @property
    def check_rtol(self):
        "tolerance on the accuracy of the solver"
        return self._check_rtol

    @check_rtol.setter
    def check_rtol(self, value):
        value = float(value)
        if value > 0:
            self._check_rtol = float(value)
        else:
            raise ValueError("check_rtol must be greater than zero.")

    @property
    def check_atol(self):
        "tolerance on the accuracy of the solver"
        return self._check_atol

    @check_atol.setter
    def check_atol(self, value):
        value = float(value)
        if value >= 0:
            self._check_atol = float(value)
        else:
            raise ValueError("check_atol must be greater than or equal to zero.")

    @property
    def _transpose_class(self):
        return self.__class__

    def transpose(self):
        "The transpose operator for this class."
        if self.is_symmetric:
            return self
        if self._transpose_class is None:
            raise NotImplementedError(
                'The transpose for the {} class is not possible.'.format(
                    self.__name__
                )
            )
        newS = self._transpose_class(self.A.T, **self.get_attributes())
        return newS

    @property
    def T(self):
        return self.transpose()

    def _compute_accuracy(self, rhs, x):
        resid_norm = np.linalg.norm(rhs - self.A @ x)
        rhs_norm = np.linalg.norm(rhs)
        tolerance = max(self.check_rtol * rhs_norm, self.check_atol)
        if resid_norm > tolerance:
            raise PymatsolverAccuracyError(
                f'Accuracy on solve is above tolerance: {resid_norm} > {tolerance}'
            )

    def solve(self, rhs):
        # Make this broadcast just like numpy.linalg.solve!

        n = self.A.shape[0]
        ndim = len(rhs.shape)
        if ndim == 1:
            if len(rhs) != n:
                raise ValueError(f'Expected a vector of length {n}, got {len(rhs)}')
            x = self._solve_single(rhs)
        else:
            if ndim == 2 and rhs.shape[-1] == 1:
                warnings.warn(
                    "In the pymatsolver v0.7.0 passing a vector of shape (n, 1) to the solve method "
                    "will return an array with shape (n, 1), instead of always returning a flattened array. "
                    "This is to be consistent with numpy.linalg.solve broadcasting.",
                    FutureWarning
                )
            if rhs.shape[-2] != n:
                raise ValueError(f'Second to last dimension should be {n}, got {rhs.shape}')
            do_broadcast = rhs.ndim > 2
            if do_broadcast:
                # switch last two dimensions
                rhs = np.transpose(rhs, (*range(rhs.ndim-2), -1, -2))
                in_shape = rhs.shape
                # Then collapse all other vectors into the last dimension
                rhs = np.reshape(rhs, (-1, in_shape[-1]))
                # Then reverse the two axes to get the array to end up in fortran order
                # (which is more common for direct solvers).
                rhs = np.transpose(rhs)
                # should end up with shape (n, -1)
            x = self._solve_multiple(rhs)
            if do_broadcast:
                # undo the reshaping above
                # so first, reverse the axes again.
                x = np.transpose(x)
                # then expand out the first dimension into multiple dimensions.
                x = np.reshape(x, in_shape)
                # then switch last two dimensions again.
                x = np.transpose(x, (*range(rhs.ndim-2), -1, -2))

        if self.check_accuracy:
            self._compute_accuracy(rhs, x)

        if x.size == n:
            x = x.reshape(-1)
        return x

    @abstractmethod
    def _solve_single(self, rhs):
        ...


    def _solve_multiple(self, rhs):
        ...


    def clean(self):
        pass

    def __del__(self):
        """Destruct to call clean when object is garbage collected."""
        try:
            # make sure clean is called in case the underlying solver
            # doesn't automatically cleanup itself when garbage collected...
            self.clean()
        except:
            pass

    def __mul__(self, val):
        return self.solve(val)

    def __matmul__(self, val):
        return self.solve(val)

    def get_attributes(self):
        attrs = {
            "is_symmetric": self.is_symmetric,
            "is_hermitian": self.is_hermitian,
            "is_positive_definite": self.is_positive_definite,
            "check_accuracy": self.check_accuracy,
            "check_rtol": self.check_rtol,
            "check_atol": self.check_atol,
        }
        return attrs


class Diagonal(Base):

    def __init__(self, A, **kwargs):
        try:
            self._diagonal = np.asarray(A.diagonal())
            if not np.all(self._diagonal):
                # this works because 0 evaluates as False!
                raise ValueError("Diagonal matrix has a zero along the diagonal.")
        except AttributeError:
            raise TypeError("A must have a diagonal() method.")
        kwargs.pop("is_symmetric", None)
        kwargs.pop("is_hermitian", None)
        is_positive_definite = kwargs.pop("is_positive_definite", None)
        super().__init__(
            A, is_symmetric=True, is_hermitian=True, **kwargs
        )
        if is_positive_definite is None:
            if self.is_real:
                is_positive_definite = self._diagonal.min() > 0
            else:
                is_positive_definite = (not np.any(self._diagonal.imag)) and self._diagonal.real.min() > 0
            is_positive_definite = bool(is_positive_definite)
        self.is_positive_definite = is_positive_definite

    def _solve_single(self, rhs):
        return rhs / self._diagonal

    def _solve_multiple(self, rhs):
        # broadcast the division
        return rhs / self._diagonal[:, None]


class TriangularSolver(Base):
    def __init__(self, A, lower=True, **kwargs):
        kwargs.pop("is_hermitian", False)
        kwargs.pop("is_symmetric", False)
        if not (sp.issparse(A) and A.format in ['csr','csc']):
            A = sp.csc_matrix(A)
        A.sum_duplicates()
        super().__init__(A, is_hermitian=False, is_symmetric=False, **kwargs)
        self.lower = lower

    @property
    def lower(self):
        return self._lower

    @lower.setter
    def lower(self, value):
        if isinstance(value, bool):
            self._lower = value
        else:
            raise TypeError("lower must be a bool.")

    def _solve_multiple(self, rhs):
        return spsolve_triangular(self.A, rhs, lower=self.lower)

    _solve_single = _solve_multiple

    def transpose(self):
        transed = super().transpose()
        transed.lower = not self.lower
        return transed

class Forward(TriangularSolver):

    def __init__(self, A, **kwargs):
        kwargs.pop("lower", None)
        super().__init__(A, lower=True, **kwargs)

class Backward(TriangularSolver):

    _transpose_class = Forward

    def __init__(self, A, **kwargs):
        kwargs.pop("lower", None)
        super().__init__(A, lower=False, **kwargs)


Forward._transpose_class = Backward