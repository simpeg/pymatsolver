import warnings
from inspect import signature
import numpy as np

from pymatsolver.solvers import Base

def _valid_kwargs_for_func(func, **kwargs):
    """Validates keyword arguments for a function by inspecting its signature.

    This will issue a warning if the function does not accept the keyword.

    Returns
    -------
    valid_kwargs : dict
        Arguments able to be passed to the function (based on its signature).

    Notes
    -----
    If a function's signature accepts `**kwargs` then all keyword arguments are
    valid by definition, even if the function might throw its own exceptions based
    off of the arguments you pass. This function will not check for those keyword
    arguments.
    """
    sig = signature(func)
    valid_kwargs = {}
    for key, value in kwargs.items():
        try:
            sig.bind_partial(**{key: value})
            valid_kwargs[key] = value
        except TypeError:
            warnings.warn(f'Unused keyword argument "{key}" for {func.__name__}.', stacklevel=3)
            # stack level of three because we want the warning issued at the call
            # to the wrapped solver's `__init__` method.
    return valid_kwargs


def WrapDirect(fun, factorize=True, name=None):
    """Wraps a direct Solver.

    Parameters
    ----------
    fun : callable
        The solver function to be wrapped.
    factorize : bool
        If `fun` returns a factorized object that has a ``solve()`` method. This allows
        it to be re-used for repeated solve calls.
    name : str, optional
        The name of the wrapped class to return.

    Returns
    -------
    wrapped : pymatsolver.solvers.Base

    Notes
    -----
    Keyword arguments passed to the returned object on initialization will be checked
    against `fun`'s signature. If `factorize` is ``True``, then they will additionally be
    checked against the factorized object's ``solve()`` method signature. These checks
    will not cause errors, but will issue warnings saying they are unused.

    Examples
    --------
    >>> import pymatsolver
    >>> from scipy.sparse.linalg import spsolve, splu

    Scipy's ``spsolve`` does not support reuse, so we must pass ``factorize=false``.
    >>> Solver   = pymatsolver.WrapDirect(spsolve, factorize=False)

    Scipy's ``splu`` returns an `SuperLU` object that has a `solve` method, and therefore
    does support reuse, so we must pass ``factorize=true``.
    >>> SolverLU = pymatsolver.WrapDirect(splu, factorize=True)
    """

    def __init__(self, A, check_accuracy=False, check_rtol=1E-6, check_atol=0, accuracy_tol=None, **kwargs):
        Base.__init__(
            self, A, check_accuracy=check_accuracy, check_rtol=check_rtol, check_atol=check_atol, accuracy_tol=accuracy_tol,
            is_symmetric=False, is_hermitian=False
        )
        self.kwargs = kwargs
        if factorize:
            self.solver = fun(self.A, **self.kwargs)
            if not hasattr(self.solver, "solve"):
                raise TypeError(f"instance returned by {fun.__name__} must have a solve() method.")

    @property
    def kwargs(self):
        return self._kwargs

    @kwargs.setter
    def kwargs(self, keyword_arguments):
        self._kwargs = _valid_kwargs_for_func(fun, **keyword_arguments)

    def _solve_single(self, rhs):
        rhs = rhs.astype(self.dtype)

        if factorize:
            X = self.solver.solve(rhs)
        else:
            X = fun(self.A, rhs, **self.kwargs)

        return X

    def _solve_multiple(self, rhs):
        rhs = rhs.astype(self.dtype)

        X = np.empty_like(rhs)
        for i in range(rhs.shape[1]):
            X[:, i] = self._solve_single(rhs[:, i])

        return X

    def clean(self):
        if factorize and hasattr(self.solver, 'clean'):
            self.solver.clean()

    return type(
        str(name if name is not None else fun.__name__),
        (Base,),
        {
            "__init__": __init__,
            "_solve_single": _solve_single,
            "_solve_multiple": _solve_multiple,
            "clean": clean,
        }
    )


def WrapIterative(fun, check_accuracy=None, accuracy_tol=None, name=None):
    """
    Wraps an iterative Solver.

    Returns
    -------
    wrapped : pymatsolver.solvers.Base

    Notes
    -----
    Keyword arguments passed to the returned object on initialization will be checked
    against `fun`'s signature.These checks will not cause errors, but will issue warnings
    saying they are unused.

    Examples
    --------
    >>> import pymatsolver
    >>> from scipy.sparse.linalg import cg
    >>> SolverCG = pymatsolver.WrapIterative(cg)

    """
    if check_accuracy is not None or accuracy_tol is not None:
        warnings.warn('check_accuracy and accuracy_tol were unused and are now deprecated. They '
                      'will be removed in pymatsolver v0.4.0. Please pass the keyword arguments `check_rtol` '
                      'and check_atol directly to the wrapped solver class.', FutureWarning)

    def __init__(self, A, check_accuracy=False, check_rtol=1E-6, check_atol=0, accuracy_tol=None, **kwargs):
        Base.__init__(
            self, A, check_accuracy=check_accuracy, check_rtol=check_rtol, check_atol=check_atol, accuracy_tol=accuracy_tol,
            is_symmetric=False, is_hermitian=False
        )
        self.kwargs = kwargs

    @property
    def kwargs(self):
        return self._kwargs

    @kwargs.setter
    def kwargs(self, keyword_arguments):
        self._kwargs = _valid_kwargs_for_func(fun, **keyword_arguments)

    def _solve_single(self, rhs):

        out = fun(self.A, rhs, **self.kwargs)
        if type(out) is tuple and len(out) == 2:
            # We are dealing with scipy output with an info!
            X = out[0]
            self.info = out[1]
        else:
            X = out
        return X

    def _solve_multiple(self, rhs):

        X = np.empty_like(rhs)
        for i in range(rhs.shape[1]):
            X[:, i] = self._solve_single(rhs[:, i])
        return X

    return type(
        str(name if name is not None else fun.__name__),
        (Base,),
        {
            "__init__": __init__,
            "_solve_single": _solve_single,
            "_solve_multiple": _solve_multiple,
        }
    )

