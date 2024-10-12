from pymatsolver import Solver, Diagonal, SolverCG, SolverLU
import scipy.sparse as sp
from scipy.sparse.linalg import aslinearoperator
import numpy as np
import numpy.testing as npt
import pytest


TOLD = 1e-10
TOLI = 1e-3

@pytest.fixture()
def a_matrix():
    nx, ny, nz = 10, 10, 10
    n = nx * ny * nz
    Gz = sp.kron(
        sp.eye(nx),
        sp.kron(
            sp.eye(ny),
            sp.diags([-1, 1], [-1, 0], shape=(nz+1, nz))
        )
    )
    Gy = sp.kron(
        sp.eye(nx),
        sp.kron(
            sp.diags([-1, 1], [-1, 0], shape=(ny+1, ny)),
            sp.eye(nz),
        )
    )
    Gx = sp.kron(
        sp.diags([-1, 1], [-1, 0], shape=(nx+1, nx)),
        sp.kron(
            sp.eye(ny),
            sp.eye(nz),
        )
    )
    A = Gx.T @ Gx + Gy.T @ Gy + Gz.T @ Gz
    return A


@pytest.mark.parametrize('n_rhs', [1, 5])
@pytest.mark.parametrize('solver', [Solver, SolverLU, SolverCG])
def test_solver(a_matrix, n_rhs, solver):
    if solver is SolverCG:
        tol = TOLI
    else:
        tol = TOLD

    n = a_matrix.shape[0]
    b = np.linspace(0.9, 1.1, n)
    if n_rhs > 1:
        b = np.repeat(b[:, None], n_rhs, axis=-1)
    rhs = a_matrix @ b

    Ainv = solver(a_matrix)
    x = Ainv * rhs
    Ainv.clean()

    npt.assert_allclose(x, b, atol=tol)

@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_iterative_solver_linear_op(dtype):
    n = 10
    A = aslinearoperator(sp.eye(n).astype(dtype))

    Ainv = SolverCG(A)

    rhs = np.linspace(0.9, 1.1, n)

    npt.assert_allclose(Ainv @ rhs, rhs)

@pytest.mark.parametrize('n_rhs', [1, 5])
def test_diag_solver(n_rhs):
    n = 10
    A = sp.diags(np.linspace(2, 3, n))
    b = np.linspace(0.9, 1.1, n)
    if n_rhs > 1:
        b = np.repeat(b[:, None], n_rhs, axis=-1)
    rhs = A @ b

    Ainv = Diagonal(A)
    x = Ainv * rhs

    npt.assert_allclose(x, b, atol=TOLD)