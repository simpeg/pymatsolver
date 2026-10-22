from pymatsolver import SolverCG, SolverLU, wrap_direct, wrap_iterative
from pymatsolver.solvers import UnusedArgumentWarning
import pytest
import scipy.sparse as sp
import warnings
import numpy.testing as npt
import numpy as np
import re


@pytest.mark.parametrize("solver_class", [SolverCG, SolverLU])
def test_wrapper_unused_kwargs(solver_class):
    A = sp.eye(10)

    with pytest.warns(UnusedArgumentWarning, match="Unused keyword argument.*"):
        solver_class(A, not_a_keyword_arg=True)


def test_good_arg_iterative():
    # Ensure this doesn't throw a warning!
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SolverCG(sp.eye(10), rtol=1e-4)


def test_good_arg_direct():
    # Ensure this doesn't throw a warning!
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SolverLU(sp.eye(10, format='csc'), permc_spec='NATURAL')


def test_bad_direct_function():
    def bad_direct_func(A):
        class Empty():
            def __init__(self, A):
                self.A = A
        # this object returned by the function doesn't have a solve method:
        return Empty(A)
    WrappedClass = wrap_direct(bad_direct_func, factorize=True)

    with pytest.raises(TypeError, match="instance returned by.*"):
        WrappedClass(sp.eye(2))


def test_bad_direct_function_on_refactor():
    class Empty():
        def __init__(self, A):
            self.A = A

    class Good(Empty):
        def solve(self, x):
            return x

    # returns a valid solver the first call (at construction), and one missing
    # a solve() method on the second call (triggered by a refactor).
    calls = []

    def flaky_direct_func(A):
        calls.append(A)
        if len(calls) == 1:
            return Good(A)
        return Empty(A)

    WrappedClass = wrap_direct(flaky_direct_func, factorize=True)
    Ainv = WrappedClass(sp.eye(2))

    with pytest.raises(TypeError, match="instance returned by.*"):
        Ainv.factor(sp.eye(2) * 2)


def test_direct_clean_function():
    def direct_func(A):
        class Empty():
            def __init__(self, A):
                self.A = A

            def solve(self, x):
                return x

            def clean(self):
                self.A = None

        return Empty(A)
    WrappedClass = wrap_direct(direct_func, factorize=True)

    A = sp.eye(2)
    Ainv = WrappedClass(A)
    assert Ainv.A is A
    assert Ainv.solver.A is A

    rhs = np.array([0.9, 1.0])
    npt.assert_equal(Ainv @ rhs, rhs)

    Ainv.clean()
    assert Ainv.solver.A is None


def test_iterative_removals():

    with pytest.raises(
        TypeError,
        match=re.escape("wrap_iterative() got an unexpected keyword argument 'check_accuracy'")
    ):
        wrap_iterative(lambda a, x: x, check_accuracy=True)

    with pytest.raises(
        TypeError,
        match=re.escape("wrap_iterative() got an unexpected keyword argument 'accuracy_tol'")
    ):
        wrap_iterative(lambda a, x: x, accuracy_tol=1E-3)


def test_non_scipy_iterative():
    def iterative_solver(A, x):
        return x

    Wrapped = wrap_iterative(iterative_solver)

    Ainv = Wrapped(sp.eye(4))
    npt.assert_equal(Ainv @ np.arange(4), np.arange(4))


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('n_rhs', [None, 3])
@pytest.mark.parametrize('conjugate', [False, True])
def test_splu_transpose_shares_factorization(dtype, n_rhs, conjugate):
    rng = np.random.default_rng(1234)
    n = 20
    A = sp.random(n, n, density=0.2, random_state=rng) + n * sp.eye(n)
    if dtype == np.complex128:
        A = A + 1j * sp.random(n, n, density=0.2, random_state=rng)
    A = A.tocsc().astype(dtype)
    rhs = rng.random(n if n_rhs is None else (n, n_rhs)).astype(dtype)

    Ainv = SolverLU(A, is_symmetric=False)
    AinvT = Ainv.T
    # the transposed solver reuses the same factorization rather than calling splu again.
    assert AinvT.solver is Ainv.solver
    assert AinvT.T.solver is Ainv.solver

    if conjugate:
        AinvT = AinvT.conj()
        A_t = A.T.conjugate()
    else:
        A_t = A.T

    expected = SolverLU(A_t.tocsc(), is_symmetric=False) @ rhs
    npt.assert_allclose(AinvT @ rhs, expected, rtol=1e-8, atol=1e-10)
    # the original is unaffected.
    npt.assert_allclose(A @ (Ainv @ rhs), rhs, rtol=1e-8, atol=1e-10)


def test_splu_transpose_symmetric_returns_self():
    A = sp.diags([2.0, 4.0, 2.0], [-1, 0, 1], shape=(10, 10), format='csc')
    Ainv = SolverLU(A)
    assert Ainv.is_symmetric
    assert Ainv.T is Ainv


def test_splu_transpose_clean_and_kwargs():
    A = sp.diags([1.0, 4.0, 2.0], [-1, 0, 1], shape=(10, 10), format='csc')
    Ainv = SolverLU(A, is_symmetric=False, permc_spec='NATURAL')
    AinvT = Ainv.T
    Ainv.clean()
    AinvT.clean()
    rhs = np.ones(10)
    npt.assert_allclose(A.T @ (AinvT @ rhs), rhs, atol=1e-10)
    with pytest.warns(UnusedArgumentWarning):
        SolverLU(A, not_a_kwarg=1)
