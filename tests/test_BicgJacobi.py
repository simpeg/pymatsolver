from pymatsolver import BicgJacobi
import numpy as np
import numpy.testing as npt
import scipy.sparse as sp
import pytest

RTOL = 1e-5

@pytest.fixture()
def test_mat_data():
    nSize = 100
    A = sp.rand(nSize, nSize, 0.05, format='csr', random_state=100)
    A = A + sp.spdiags(np.ones(nSize), 0, nSize, nSize)
    A = A.T*A
    A = A.tocsr()
    np.random.seed(1)
    sol = np.random.rand(nSize, 4)
    return A, sol


@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('symmetric', [True, False])
def test_solve(test_mat_data, dtype, transpose, symmetric):
    A, sol = test_mat_data
    A = A.astype(dtype)
    sol = sol.astype(dtype)
    if not symmetric:
        D = sp.diags(np.linspace(2, 3, A.shape[0]))
        A = D @ A
    rhs = A @ sol
    if transpose:
        Ainv = BicgJacobi(A.T, is_symmetric=symmetric).T
    else:
        Ainv = BicgJacobi(A, is_symmetric=symmetric)
    Ainv.maxiter = 2000
    solb = Ainv * rhs
    npt.assert_allclose(rhs, A @ solb, rtol=RTOL)

def test_errors_and_warnings(test_mat_data):
    A, sol = test_mat_data
    with pytest.warns(FutureWarning):
        Ainv = BicgJacobi(A, symmetric=True)

    with pytest.raises(ValueError):
        Ainv = BicgJacobi(A, rtol=0.0)

    with pytest.raises(ValueError):
        Ainv = BicgJacobi(A, atol=-1.0)

def test_shallow_copy(test_mat_data):
    A, sol = test_mat_data
    Ainv = BicgJacobi(A, maxiter=100, rtol=1.0E-3, atol=1.0E-16)

    attrs = Ainv.get_attributes()

    new_Ainv = BicgJacobi(A, **attrs)
    assert attrs == new_Ainv.get_attributes()