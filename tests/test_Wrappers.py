from pymatsolver import SolverCG, SolverLU
import pytest
import scipy.sparse as sp
import warnings


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
