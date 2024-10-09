from pymatsolver import BicgJacobi
import numpy as np
import numpy.testing as npt
import scipy.sparse as sp
import pytest

TOL = 1e-5

@pytest.fixture()
def test_mat_data():
    nSize = 100
    A = sp.rand(nSize, nSize, 0.05, format='csr', random_state=100)
    A = A + sp.spdiags(np.ones(nSize), 0, nSize, nSize)
    A = A.T*A
    A = A.tocsr()
    np.random.seed(1)
    sol = np.random.rand(nSize, 4)
    rhs = A.dot(sol)
    return A, sol, rhs


@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_solve(test_mat_data, dtype, transpose):
    A, rhs, sol = test_mat_data
    A = A.astype(dtype)
    rhs = rhs.astype(dtype)
    sol = sol.astype(dtype)
    if transpose:
        A = A.T
        Ainv = BicgJacobi(A).T
    else:
        Ainv = BicgJacobi(A)
    Ainv.maxiter = 2000
    solb = Ainv * rhs
    npt.assert_allclose(rhs, A @ solb, atol=TOL)

def test_errors_and_warnings(test_mat_data):
    A, rhs, sol = test_mat_data
    with pytest.warns(FutureWarning):
        Ainv = BicgJacobi(A, symmetric=True)

    with pytest.raises(ValueError):
        Ainv = BicgJacobi(A, rtol=0.0)

    with pytest.raises(ValueError):
        Ainv = BicgJacobi(A, atol=-1.0)

def test_shallow_copy(test_mat_data):
    A, rhs, sol = test_mat_data
    Ainv = BicgJacobi(A, maxiter=100, rtol=1.0E-3, atol=1.0E-16)

    attrs = Ainv.get_attributes()

    new_Ainv = BicgJacobi(A, **attrs)
    assert attrs == new_Ainv.get_attributes()