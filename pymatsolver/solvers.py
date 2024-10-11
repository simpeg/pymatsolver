import numpy as np
import warnings
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve_triangular
from scipy.linalg import issymmetric, ishermitian
from abc import ABC, abstractmethod
import copy


class SolverAccuracyError(Exception):
    pass


class UnusedArgumentWarning(UserWarning):
    pass


class Base(ABC):
    """Base class for all solvers used in the pymatsolver package.

    Parameters
    ----------
    A
        Matrix to solve with.
    is_symmetric : bool, optional
        Whether the matrix is symmetric. By default, it will perform some simple tests to check for symmetry, and
        default to ``False`` if those fail.
    is_positive_definite : bool, optional
        Whether the matrix is positive definite.
    is_hermitian : bool, optional
        Whether the matrix is hermitian. By default, it will perform some simple tests to check, and default to
        ``False`` if those fail.
    check_accuracy : bool, optional
        Whether to check the accuracy of the solution.
    check_rtol : float, optional
        The relative tolerance to check against for accuracy.
    check_atol : float, optional
        The absolute tolerance to check against for accuracy.
    accuracy_tol : float, optional
        Relative accuracy tolerance.
        .. deprecated:: 0.3.0
            `accuracy_tol` will be removed in pymatsolver 0.4.0. Use `check_rtol` and `check_atol` instead.
    **kwargs
        Extra keyword arguments. If there are any left here a warning will be raised.
    """

    __numpy_ufunc__ = True
    __array_ufunc__ = None

    _is_conjugate = False

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
            warnings.warn(
                "accuracy_tol is deprecated and will be removed in v0.4.0, use check_rtol and check_atol.",
                FutureWarning,
                stacklevel=3
            )
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
            warnings.warn(
                f"Unused keyword arguments for {self.__class__.__name__}: {kwargs.keys()}",
                UnusedArgumentWarning,
                stacklevel=3
            )

    @property
    def A(self):
        """The matrix to solve with."""
        return self._A

    @property
    def dtype(self):
        """The data type of the matrix.

        Returns
        -------
        numpy.dtype
        """
        return self._dtype

    @property
    def shape(self):
        """The input matrix dimensions.

        Returns
        -------
        (2, ) tuple
        """
        return self.A.shape

    @property
    def is_real(self):
        """Whether the matrix is real.

        Returns
        -------
        bool
        """
        return np.issubdtype(self.A.dtype, np.floating)

    @property
    def is_symmetric(self):
        """Whether the matrix is symmetric.

        Returns
        -------
        bool
        """
        return self._is_symmetric

    @is_symmetric.setter
    def is_symmetric(self, value):
        if isinstance(value, bool):
            self._is_symmetric = value
        else:
            raise TypeError("is_symmetric must be a boolean.")

    @property
    def is_hermitian(self):
        """Whether the matrix is hermitian.

        Returns
        -------
        bool
        """
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
        """Whether the matrix is positive definite.

        Returns
        -------
        bool
        """
        return self._is_positive_definite

    @is_positive_definite.setter
    def is_positive_definite(self, value):
        if isinstance(value, bool):
            self._is_positive_definite = value
        else:
            raise TypeError("is_positive_definite must be a boolean.")

    @property
    def check_accuracy(self):
        """Whether the check the accuracy after a solve.

        Performs a test of:
        >>> all(A @ x_solve - rhs <= max(check_rtol * norm(rhs), check_atol))

        Returns
        -------
        bool
        """
        return self._check_accuracy

    @check_accuracy.setter
    def check_accuracy(self, value):
        if isinstance(value, bool):
            self._check_accuracy = value
        else:
            raise TypeError("check_accuracy must be a boolean.")

    @property
    def check_rtol(self):
        """The relative tolerance used to check the solve operation.

        Returns
        -------
        bool
        """
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
        """The absolute tolerance used to check the solve operation.

        Returns
        -------
        bool
        """
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
        """Return the transposed solve operator.

        Returns
        -------
        pymatsolver.solvers.Base
        """

        if self.is_symmetric:
            return self
        if self._transpose_class is None:
            raise NotImplementedError(
                'The transpose for the {} class is not possible.'.format(
                    self.__class__.__name__
                )
            )
        newS = self._transpose_class(self.A.T, **self.get_attributes())
        return newS

    @property
    def T(self):
        """The transposed solve operator

        See Also
        --------
        transpose
            `T` is an alias for `transpose()`.
        """
        return self.transpose()

    def conjugate(self):
        """Return the complex conjugate version of this solver.

        Returns
        -------
        pymatsolver.solvers.Base
        """
        if self.is_real:
            return self
        else:
            # make a shallow copy of myself
            conjugated = copy.copy(self)
            conjugated._is_conjugate = not self._is_conjugate
            return conjugated

    conj = conjugate

    def _compute_accuracy(self, rhs, x):
        resid_norm = np.linalg.norm(rhs - self.A @ x)
        rhs_norm = np.linalg.norm(rhs)
        tolerance = max(self.check_rtol * rhs_norm, self.check_atol)
        if resid_norm > tolerance:
            raise SolverAccuracyError(
                f'Accuracy on solve is above tolerance: {resid_norm} > {tolerance}'
            )

    def solve(self, rhs):
        """Solves the system of equations for the given right hand side.

        Parameters
        ----------
        rhs : (..., M, N) or (M, ) array_like
            The right handside of A @ x = b.

        Returns
        -------
        x : (..., M, N) or (M, ) array_like
            The solution to the system of equations.

        See Also
        --------
        numpy.linalg.solve
            Examples of how broadcasting works for this operation.
        """
        # Make this broadcast just like numpy.linalg.solve!

        n = self.A.shape[0]
        ndim = len(rhs.shape)
        if ndim == 1:
            if len(rhs) != n:
                raise ValueError(f'Expected a vector of length {n}, got {len(rhs)}')
            if self._is_conjugate:
                rhs = rhs.conjugate()
            x = self._solve_single(rhs)
        else:
            if ndim == 2 and rhs.shape[-1] == 1:
                warnings.warn(
                    "In Future pymatsolver v0.4.0, passing a vector of shape (n, 1) to the solve method "
                    "will return an array with shape (n, 1), instead of always returning a flattened array. "
                    "This is to be consistent with numpy.linalg.solve broadcasting.",
                    FutureWarning,
                    stacklevel=2
                )
            if rhs.shape[-2] != n:
                raise ValueError(f'Second to last dimension should be {n}, got {rhs.shape}')
            do_broadcast = rhs.ndim > 2
            if do_broadcast:
                # swap last two dimensions
                rhs = rhs.swapaxes(-1, -2)
                in_shape = rhs.shape
                # Then collapse all other vectors into the first dimension
                rhs = rhs.reshape((-1, in_shape[-1]))
                # Then reverse the two axes to get the array to end up in fortran order
                # (which is more common for direct solvers).
                rhs = rhs.transpose()
                # should end up with shape (n, -1)
            if self._is_conjugate:
                rhs = rhs.conjugate()
            x = self._solve_multiple(rhs)
            if do_broadcast:
                # undo the reshaping above
                # so first, reverse the axes again.
                x = x.transpose()
                # then expand out the first dimension into multiple dimensions.
                x = x.reshape(in_shape)
                # then switch last two dimensions again.
                x = x.swapaxes(-1, -2)

        if self.check_accuracy:
            self._compute_accuracy(rhs, x)

        #TODO remove this in v0.4.0.
        if x.size == n:
            x = x.reshape(-1)

        if self._is_conjugate:
            x = x.conjugate()
        return x

    @abstractmethod
    def _solve_single(self, rhs):
        ...

    @abstractmethod
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
        return self.__matmul__(val)

    def __rmul__(self, val):
        return self.__rmatmul__(val)

    def __matmul__(self, val):
        return self.solve(val)

    def __rmatmul__(self, val):
        tran_solver = self.transpose()
        # transpose last two axes of val
        if val.ndim > 1:
            val = val.swapaxes(-1, -2)
        out = tran_solver.solve(val)
        if val.ndim > 1:
            out = out.swapaxes(-1, -2)
        return out

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
    """A solver for a diagonal matrix.

    Parameters
    ----------
    A
        The diagonal matrix, must have a ``diagonal()`` method.
    check_accuracy : bool, optional
        Whether to check the accuracy of the solution.
    check_rtol : float, optional
        The relative tolerance to check against for accuracy.
    check_atol : float, optional
        The absolute tolerance to check against for accuracy.
    accuracy_tol : float, optional
        Relative accuracy tolerance.
        .. deprecated:: 0.3.0
            `accuracy_tol` will be removed in pymatsolver 0.4.0. Use `check_rtol` and `check_atol` instead.
    **kwargs
        Extra keyword arguments passed to the base class.
    """

    def __init__(self, A, check_accuracy=False, check_rtol=1e-6, check_atol=0, accuracy_tol=None, **kwargs):
        try:
            self._diagonal = np.asarray(A.diagonal())
            if not np.all(self._diagonal):
                # this works because 0.0 evaluates as False!
                raise ValueError("Diagonal matrix has a zero along the diagonal.")
        except AttributeError:
            raise TypeError("A must have a diagonal() method.")
        kwargs.pop("is_symmetric", None)
        is_hermitian = kwargs.pop("is_hermitian", None)
        is_positive_definite = kwargs.pop("is_positive_definite", None)
        super().__init__(
            A, is_symmetric=True, is_hermitian=False, check_accuracy=check_accuracy, check_rtol=check_rtol, check_atol=check_atol, accuracy_tol=accuracy_tol, **kwargs
        )
        if is_positive_definite is None:
            if self.is_real:
                is_positive_definite = self._diagonal.min() > 0
            else:
                is_positive_definite = (not np.any(self._diagonal.imag)) and self._diagonal.real.min() > 0
            is_positive_definite = bool(is_positive_definite)
        self.is_positive_definite = is_positive_definite

        if is_hermitian is None:
            if self.is_real:
                is_hermitian = True
            else:
                # can only be hermitian if all imaginary components on diagonal are zero.
                is_hermitian = not np.any(self._diagonal.imag)
        self.is_hermitian = is_hermitian

    def _solve_single(self, rhs):
        return rhs / self._diagonal

    def _solve_multiple(self, rhs):
        # broadcast the division
        return rhs / self._diagonal[:, None]


class Triangle(Base):
    """A solver for a diagonal matrix.

    Parameters
    ----------
    A : scipy.sparse.sparray or scipy.sparse.spmatrix
        The matrix to solve.
    lower : bool, optional
        Whether A is lower triangular (``True``), or upper triangular (``False``).
    check_accuracy : bool, optional
        Whether to check the accuracy of the solution.
    check_rtol : float, optional
        The relative tolerance to check against for accuracy.
    check_atol : float, optional
        The absolute tolerance to check against for accuracy.
    accuracy_tol : float, optional
        Relative accuracy tolerance.
        .. deprecated:: 0.3.0
            `accuracy_tol` will be removed in pymatsolver 0.4.0. Use `check_rtol` and `check_atol` instead.
    **kwargs
        Extra keyword arguments passed to the base class.
    """

    def __init__(self, A, lower=True, check_accuracy=False, check_rtol=1e-6, check_atol=0, accuracy_tol=None, **kwargs):
        # pop off unneeded keyword arguments.
        is_hermitian = kwargs.pop("is_hermitian", False)
        is_symmetric = kwargs.pop("is_symmetric", False)
        is_positive_definite = kwargs.pop("is_positive_definite", False)
        if not (sp.issparse(A) and A.format in ['csr', 'csc']):
            A = sp.csc_matrix(A)
        A.sum_duplicates()
        super().__init__(A, is_hermitian=is_hermitian, is_symmetric=is_symmetric, is_positive_definite=is_positive_definite, check_accuracy=check_accuracy, check_rtol=check_rtol, check_atol=check_atol, accuracy_tol=accuracy_tol, **kwargs)

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
        trans = super().transpose()
        trans.lower = not self.lower
        return trans


class Forward(Triangle):
    """A solver for a lower triangular matrix.

    Parameters
    ----------
    A : scipy.sparse.sparray or scipy.sparse.spmatrix
        The lower triangular matrix to solve.
    check_accuracy : bool, optional
        Whether to check the accuracy of the solution.
    check_rtol : float, optional
        The relative tolerance to check against for accuracy.
    check_atol : float, optional
        The absolute tolerance to check against for accuracy.
    accuracy_tol : float, optional
        Relative accuracy tolerance.
        .. deprecated:: 0.3.0
            `accuracy_tol` will be removed in pymatsolver 0.4.0. Use `check_rtol` and `check_atol` instead.
    **kwargs
        Extra keyword arguments passed to the base class.
    """

    def __init__(self, A, check_accuracy=False, check_rtol=1e-6, check_atol=0, accuracy_tol=None, **kwargs):
        kwargs.pop("lower", None)
        super().__init__(A, lower=True, check_accuracy=check_accuracy, check_rtol=check_rtol, check_atol=check_atol, accuracy_tol=accuracy_tol, **kwargs)


class Backward(Triangle):
    """A solver for ann upper triangular matrix.

    Parameters
    ----------
    A : scipy.sparse.sparray or scipy.sparse.spmatrix
        The upper triangular matrix to solve.
    check_accuracy : bool, optional
        Whether to check the accuracy of the solution.
    check_rtol : float, optional
        The relative tolerance to check against for accuracy.
    check_atol : float, optional
        The absolute tolerance to check against for accuracy.
    accuracy_tol : float, optional
        Relative accuracy tolerance.
        .. deprecated:: 0.3.0
            `accuracy_tol` will be removed in pymatsolver 0.4.0. Use `check_rtol` and `check_atol` instead.
    **kwargs
        Extra keyword arguments passed to the base class.
    """

    _transpose_class = Forward

    def __init__(self, A, check_accuracy=False, check_rtol=1e-6, check_atol=0, accuracy_tol=None, **kwargs):
        kwargs.pop("lower", None)
        super().__init__(A, lower=False, check_accuracy=check_accuracy, check_rtol=check_rtol, check_atol=check_atol, accuracy_tol=accuracy_tol, **kwargs)


Forward._transpose_class = Backward