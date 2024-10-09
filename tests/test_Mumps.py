import numpy as np
import scipy.sparse as sp
import pymatsolver
import pytest
import numpy.testing as npt

if not pymatsolver.AvailableSolvers['Mumps']:
    pytest.skip(reason="MUMPS solver is not installed", allow_module_level=True)

TOL = 1e-11

@pytest.fixture()
def test_mat_data():
    nSize = 100
    A = sp.rand(nSize, nSize, 0.05, format='csr', random_state=100)
    A = A + sp.spdiags(np.ones(nSize), 0, nSize, nSize)
    A = A.T*A
    A = A.tocsr()
    sol = np.linspace(0.9, 1.1, nSize)
    sol = np.repeat(sol[:, None], 5, axis=-1)
    return A, sol


@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('symmetric', [True, False])
def test_solve(test_mat_data, dtype, transpose, symmetric):
    A, sol = test_mat_data
    sol = sol.astype(dtype)
    A = A.astype(dtype)
    if not symmetric:
        D = sp.diags(np.linspace(2, 3, A.shape[0]))
        A = D @ A
    rhs = A @ sol
    if transpose:
        Ainv = pymatsolver.Mumps(A.T, is_symmetric=symmetric).T
    else:
        Ainv = pymatsolver.Mumps(A, is_symmetric=symmetric)
    for i in range(rhs.shape[1]):
        npt.assert_allclose(Ainv * rhs[:, i], sol[:, i], atol=TOL)
    npt.assert_allclose(Ainv * rhs, sol, atol=TOL)


def test_refactor(test_mat_data):
    A, sol = test_mat_data
    rhs = A @ sol
    Ainv = pymatsolver.Mumps(A, is_symmetric=True)
    npt.assert_allclose(Ainv * rhs, sol, atol=TOL)

    # scale rows and columns
    D = sp.diags(np.random.rand(A.shape[0]) + 1.0)
    A2 = D.T @ A @ D

    rhs2 = A2 @ sol
    Ainv.factor(A2)
    npt.assert_allclose(Ainv * rhs2, sol, atol=TOL)