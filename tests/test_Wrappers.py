from pymatsolver import SolverCG, SolverLU, wrap_direct, wrap_iterative
import pytest
import scipy.sparse as sp
import warnings
import numpy.testing as npt
import numpy as np


@pytest.mark.parametrize("solver_class", [SolverCG, SolverLU])
def test_wrapper_unused_kwargs(solver_class):
    A = sp.eye(10)

    with pytest.warns(UserWarning, match="Unused keyword argument.*"):
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
    Ainv.clean()
    assert Ainv.solver.A is None

def test_iterative_deprecations():
    def iterative_solver(A, x):
        return x

    with pytest.warns(FutureWarning, match="check_accuracy and accuracy_tol were unused.*"):
        wrap_iterative(iterative_solver, check_accuracy=True)

    with pytest.warns(FutureWarning, match="check_accuracy and accuracy_tol were unused.*"):
        wrap_iterative(iterative_solver, accuracy_tol=1E-3)

def test_non_scipy_iterative():
    def iterative_solver(A, x):
        return x

    Wrapped = wrap_iterative(iterative_solver)

    Ainv = Wrapped(sp.eye(4))
    npt.assert_equal(Ainv @ np.arange(4), np.arange(4))
