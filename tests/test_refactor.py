import numpy as np
import numpy.testing as npt
import pytest
import scipy.sparse as sp
import pymatsolver

SOLVERS = [
    pymatsolver.Diagonal,
    pymatsolver.Solver,
    pymatsolver.SolverLU,
    pymatsolver.SolverCG,
    pymatsolver.SolverBiCG,
    pymatsolver.BiCGJacobi,
    pymatsolver.Forward,
    pymatsolver.Backward,
    pymatsolver.Pardiso,
    pymatsolver.Mumps,
]

SHARED_SOLVERS = [pymatsolver.Pardiso, pymatsolver.Mumps]

TOL = 1e-4


def _skip_unavailable(solver_class):
    if solver_class is pymatsolver.Pardiso and not pymatsolver.AvailableSolvers['Pardiso']:
        pytest.skip("pydiso not installed.")
    if solver_class is pymatsolver.Mumps and not pymatsolver.AvailableSolvers['Mumps']:
        pytest.skip("python-mumps not installed.")


def _get_matrix(solver_class, n=20):
    rng = np.random.default_rng(4421)
    if solver_class is pymatsolver.Diagonal:
        return sp.diags(rng.uniform(1, 2, n)).tocsr()
    A = sp.random(n, n, density=0.2, random_state=rng)
    # symmetric and diagonally dominant, so it is positive definite for SolverCG.
    A = (A + A.T + n * sp.eye(n)).tocsr()
    if solver_class is pymatsolver.Forward:
        return sp.tril(A, format='csr')
    if solver_class is pymatsolver.Backward:
        return sp.triu(A, format='csr')
    return A


def _scaled(A):
    # same sparsity pattern and symmetry, different values.
    D = sp.diags(np.linspace(1.0, 3.0, A.shape[0]))
    return (D @ A @ D).tocsr()


def _assert_solves(Ainv, A, rhs):
    x = Ainv @ rhs
    assert x.shape == rhs.shape
    npt.assert_allclose(A @ x, rhs, rtol=TOL, atol=TOL * np.linalg.norm(rhs))


@pytest.mark.parametrize("solver_class", SOLVERS)
def test_refactor(solver_class):
    _skip_unavailable(solver_class)
    A = _get_matrix(solver_class)
    rhs = np.linspace(-1, 1, A.shape[0])
    rhs2d = np.stack([rhs, rhs ** 2], axis=-1)

    Ainv = solver_class(A)
    _assert_solves(Ainv, A, rhs)

    A2 = _scaled(A)
    Ainv.factor(A2)
    assert Ainv.A is A2 or (Ainv.A != A2).nnz == 0
    _assert_solves(Ainv, A2, rhs)
    _assert_solves(Ainv, A2, rhs2d)


@pytest.mark.parametrize("solver_class", SOLVERS)
def test_factor_noop(solver_class):
    _skip_unavailable(solver_class)
    A = _get_matrix(solver_class)
    rhs = np.linspace(-1, 1, A.shape[0])

    Ainv = solver_class(A)
    Ainv.factor()
    _assert_solves(Ainv, A, rhs)
    Ainv.factor(Ainv.A)
    _assert_solves(Ainv, A, rhs)


@pytest.mark.parametrize("solver_class", SOLVERS)
def test_refactor_errors(solver_class):
    _skip_unavailable(solver_class)
    A = _get_matrix(solver_class)
    Ainv = solver_class(A)

    with pytest.raises(ValueError, match="A must have shape"):
        Ainv.factor(_get_matrix(solver_class, n=A.shape[0] + 1))

    with pytest.raises(ValueError, match="A must have dtype"):
        Ainv.factor(A.astype(np.complex128))


@pytest.mark.parametrize("solver_class", SHARED_SOLVERS)
def test_refactor_updates_views(solver_class):
    _skip_unavailable(solver_class)
    # use a non-symmetric matrix so the transpose is distinct.
    A = _get_matrix(solver_class)
    A = (A + sp.triu(A, k=1, format='csr')).tocsr()
    rhs = np.linspace(-1, 1, A.shape[0])

    Ainv = solver_class(A, is_symmetric=False, check_accuracy=True)
    AinvT = Ainv.T
    Ainv_conj = Ainv.conj()
    _assert_solves(AinvT, A.T, rhs)

    A2 = _scaled(A)
    Ainv.factor(A2)
    _assert_solves(Ainv, A2, rhs)
    _assert_solves(AinvT, A2.T, rhs)
    _assert_solves(Ainv_conj, A2, rhs)

    # refactoring from the transposed view updates the original too.
    A3 = _scaled(A2)
    AinvT.factor(A3.T.tocsr())
    _assert_solves(Ainv, A3, rhs)
    _assert_solves(AinvT, A3.T, rhs)
