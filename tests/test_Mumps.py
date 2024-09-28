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
    n = 5
    irn = np.r_[0, 1, 3, 4, 1, 0, 4, 2, 1, 2, 0, 2]
    jcn = np.r_[1, 2, 2, 4, 0, 0, 1, 3, 4, 1, 2, 2]
    a = np.r_[
        3.0, -3.0, 2.0, 1.0, 3.0, 2.0,
        4.0, 2.0, 6.0, -1.0, 4.0, 1.0
    ]
    rhs = np.r_[20.0, 24.0, 9.0, 6.0, 13.0]
    rhs = np.c_[rhs, 10 * rhs, 100 * rhs]
    sol = np.r_[1., 2., 3., 4., 5.]
    sol = np.c_[sol, 10 * sol, 100 * sol]
    A = sp.coo_matrix((a, (irn, jcn)), shape=(n, n))
    return A, rhs, sol

@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('symmetric', [True, False])
def test_solve(test_mat_data, dtype, transpose, symmetric):
    A, rhs, sol = test_mat_data
    sol = sol.astype(dtype)
    rhs = rhs.astype(dtype)
    A = A.astype(dtype)
    if transpose:
        Ainv = pymatsolver.Mumps(A.T).T
    else:
        Ainv = pymatsolver.Mumps(A)
    for i in range(3):
        npt.assert_allclose(Ainv * rhs[:, i], sol[:, i], atol=TOL)
    npt.assert_allclose(Ainv * rhs, sol, atol=TOL)


# def test_singular(self):
#     A = sp.identity(5).tocsr()
#     A[-1, -1] = 0
#     self.assertRaises(Exception, pymatsolver.Mumps, A)