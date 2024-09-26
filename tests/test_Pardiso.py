import pymatsolver
try:
    from pydiso.mkl_solver import (
        get_mkl_pardiso_max_threads,
        PardisoTypeConversionWarning
    )
except ImportError:
    Pardiso = None
import numpy as np
import pytest
import scipy.sparse as sp
import os

if not pymatsolver.AvailableSolvers['Pardiso']:
    pytest.skip(reason="Pardiso solver is not installed", allow_module_level=True)
else:
    from pymatsolver.direct.pardiso import get_mkl_pardiso_max_threads

TOL = 1e-10

@pytest.fixture()
def test_mat_data():
    nSize = 100
    A = sp.rand(nSize, nSize, 0.05, format='csr', random_state=100)
    A = A + sp.spdiags(np.ones(nSize), 0, nSize, nSize)
    A = A.T*A
    A = A.tocsr()
    np.random.seed(1)
    sol = np.random.rand(nSize, 5)
    rhs = A.dot(sol)
    return A, rhs, sol

@pytest.fixture('transpose', [True, False])
@pytest.fixture('dtype', [np.float64, np.complex128])
def test_solve(test_mat_data, dtype, transpose):
    A, rhs, sol = test_mat_data
    sol = sol.astype(dtype)
    rhs = rhs.astype(dtype)
    A = A.astype(dtype)
    if transpose:
        Ainv = Pardiso(A.T).T
    else:
        Ainv = Pardiso(A)
    for i in range(rhs.shape[1]):
        np.testing.assert_allclose(Ainv * rhs[:, i], sol[:, i], atol=TOL)
    np.testing.assert_allclose(Ainv * rhs, sol, atol=TOL)

def test_symmetric_solve(test_mat_data):
    A, rhs, sol = test_mat_data
    Ainv = Pardiso(A, is_symmetric=True)
    for i in range(rhs.shape[1]):
        np.testing.assert_allclose(Ainv * rhs[:, i], sol[:, i], atol=TOL)
    np.testing.assert_allclose(Ainv * rhs, sol, atol=TOL)


def test_refactor(test_mat_data):
    A, rhs, sol = test_mat_data
    Ainv = Pardiso(A, is_symmetric=True)
    np.testing.assert_allclose(Ainv * rhs, sol, atol=TOL)

    # scale rows and columns
    D = sp.diags(np.random.rand(A.shape[0]) + 1.0)
    A2 = D.T @ A @ D

    rhs2 = A2 @ sol
    Ainv.factor(A2)
    np.testing.assert_allclose(Ainv * rhs2, sol, atol=TOL)

def test_n_threads(test_mat_data):
    A, rhs, sol = test_mat_data

    max_threads = get_mkl_pardiso_max_threads()
    print(f'testing setting n_threads to 1 and {max_threads}')
    Ainv = Pardiso(A, is_symmetric=True, n_threads=1)
    assert Ainv.n_threads == 1

    Ainv2 = Pardiso(A, is_symmetric=True, n_threads=max_threads)
    assert Ainv2.n_threads == max_threads

    # the n_threads setting is global so setting Ainv2's n_threads will
    # change Ainv's n_threads.
    assert Ainv2.n_threads == Ainv.n_threads

    # setting one object's n_threads should change all
    Ainv.n_threads = 1
    assert Ainv.n_threads == 1
    assert Ainv2.n_threads == Ainv.n_threads

    with pytest.raises(TypeError):
        Ainv.n_threads = "2"

# class TestPardisoNotSymmetric:
#
#     @classmethod
#     def setup_class(cls):
#         cls.A = A
#         cls.rhs = rhs
#         cls.sol = sol
#
#     def test(self):
#         rhs = self.rhs
#         sol = self.sol
#         Ainv = Pardiso(self.A, is_symmetric=True, check_accuracy=True)
#         with pytest.raises(Exception):
#             Ainv * rhs
#         Ainv.clean()
#
#         Ainv = Pardiso(self.A)
#         for i in range(3):
#             assert np.linalg.norm(Ainv * rhs[:, i] - sol[:, i]) < TOL
#         assert np.linalg.norm(Ainv * rhs - sol, np.inf) < TOL
#         Ainv.clean()


def test_pardiso_fdem:

    base_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'fdem')

    data = np.load(os.path.join(base_path, 'A_data.npy'))
    indices = np.load(os.path.join(base_path, 'A_indices.npy'))
    indptr = np.load(os.path.join(base_path, 'A_indptr.npy'))

    A = sp.csr_matrix((data, indices, indptr), shape=(13872, 13872))
    rhs = np.load(os.path.join(base_path, 'RHS.npy'))

    Ainv = Pardiso(A, check_accuracy=True)

    with pytest.warns(PardisoTypeConversionWarning):
        sol = Ainv * rhs.real