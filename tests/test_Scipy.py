from pymatsolver import Solver, Diagonal, SolverCG, SolverLU
import scipy.sparse as sp
import numpy as np
import pytest


TOLD = 1e-10
TOLI = 1e-3
numRHS = 5

np.random.seed(77)


def dotest(MYSOLVER, multi=False, A=None, **solverOpts):
    if A is None:
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
    else:
        n = A.shape[0]

    Ainv = MYSOLVER(A, **solverOpts)
    if multi:
        e = np.ones(n)
    else:
        e = np.ones((n, numRHS))
    rhs = A * e
    x = Ainv * rhs
    Ainv.clean()
    return np.linalg.norm(e-x, np.inf)


@pytest.mark.parametrize(
    ["solver", "multi"],
    [
        pytest.param(Solver, False),
        pytest.param(Solver, True),
        pytest.param(SolverLU, False),
        pytest.param(SolverLU, True),
    ]
)
def test_direct(solver, multi):
    assert dotest(solver, multi) < TOLD


@pytest.mark.parametrize(
    ["solver", "multi", "A"],
    [
        pytest.param(Diagonal, False, sp.diags(np.random.rand(10)+1.0)),
        pytest.param(Diagonal, True, sp.diags(np.random.rand(10)+1.0)),
        pytest.param(SolverCG, False, None),
        pytest.param(SolverCG, True, None),
    ]
)
def test_iterative(solver, multi, A):
    assert dotest(solver, multi, A) < TOLI
