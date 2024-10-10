import pymatsolver
import numpy as np
import numpy.testing as npt
import pytest
import scipy.sparse as sp
import os

if not pymatsolver.AvailableSolvers['Pardiso']:
    pytest.skip(reason="Pardiso solver is not installed", allow_module_level=True)
else:
    from pydiso.mkl_solver import get_mkl_pardiso_max_threads

TOL = 1e-10

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
@pytest.mark.parametrize('symmetry', ["S", "H", None])
def test_solve(test_mat_data, dtype, transpose, symmetry):

    A, sol = test_mat_data

    if symmetry is None:
        D = sp.diags(np.linspace(2, 3, A.shape[0]))
        A = D @ A
        symmetric = False
        hermitian = False
    elif symmetry == "H":
        D = sp.diags(np.linspace(2, 3, A.shape[0]))
        if np.issubdtype(dtype, np.complexfloating):
            D = D + 1j * sp.diags(np.linspace(3, 4, A.shape[0]))
        A = D @ A @ D.T.conjugate()
        symmetric = False
        hermitian = True
    else:
        symmetric = True
        hermitian = False

    sol = sol.astype(dtype)
    A = A.astype(dtype)

    Ainv = pymatsolver.Pardiso(A, is_symmetric=symmetric, is_hermitian=hermitian)
    if transpose:
        rhs = A.T @ sol
        Ainv = Ainv.T
    else:
        rhs = A @ sol

    for i in range(rhs.shape[1]):
        npt.assert_allclose(Ainv * rhs[:, i], sol[:, i], atol=TOL)
    npt.assert_allclose(Ainv * rhs, sol, atol=TOL)

@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_pardiso_positive_definite(dtype, transpose):
    n = 5
    if dtype == np.float64:
        L = sp.diags([1, -1], [0, -1], shape=(n, n))
    else:
        L = sp.diags([1, -1j], [0, -1], shape=(n, n))
    D = sp.diags(np.linspace(1, 2, n))
    A_pd = L @ D @ (L.T.conjugate())

    sol = np.linspace(0.9, 1.1, n)

    is_symmetric = dtype == np.float64
    Ainv = pymatsolver.Pardiso(A_pd, is_symmetric=is_symmetric, is_hermitian=True, is_positive_definite=True)
    if transpose:
        rhs = A_pd.T @ sol
        Ainv = Ainv.T
    else:
        rhs = A_pd @ sol

    npt.assert_allclose(Ainv @ rhs, sol)


def test_refactor(test_mat_data):
    A, sol = test_mat_data
    rhs = A @ sol
    Ainv = pymatsolver.Pardiso(A, is_symmetric=True)
    npt.assert_allclose(Ainv * rhs, sol, atol=TOL)

    # scale rows and columns
    D = sp.diags(np.random.rand(A.shape[0]) + 1.0)
    A2 = D.T @ A @ D

    rhs2 = A2 @ sol
    Ainv.factor(A2)
    npt.assert_allclose(Ainv * rhs2, sol, atol=TOL)

def test_n_threads(test_mat_data):
    A, sol = test_mat_data

    max_threads = get_mkl_pardiso_max_threads()
    print(f'testing setting n_threads to 1 and {max_threads}')
    Ainv = pymatsolver.Pardiso(A, is_symmetric=True, n_threads=1)
    assert Ainv.n_threads == 1

    Ainv2 = pymatsolver.Pardiso(A, is_symmetric=True, n_threads=max_threads)
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

def test_inacurrate_symmetry(test_mat_data):
    A, sol = test_mat_data
    rhs = A @ sol
    # make A not symmetric
    D = sp.diags(np.linspace(2, 3, A.shape[0]))
    A = A @ D
    Ainv = pymatsolver.Pardiso(A, is_symmetric=True, check_accuracy=True)
    with pytest.raises(pymatsolver.PymatsolverAccuracyError):
        Ainv * rhs



def test_pardiso_fdem():
    base_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'fdem')

    data = np.load(os.path.join(base_path, 'A_data.npy'))
    indices = np.load(os.path.join(base_path, 'A_indices.npy'))
    indptr = np.load(os.path.join(base_path, 'A_indptr.npy'))

    A = sp.csr_matrix((data, indices, indptr), shape=(13872, 13872))
    rhs = np.load(os.path.join(base_path, 'RHS.npy'))

    Ainv = pymatsolver.Pardiso(A, check_accuracy=True)
    print(Ainv.is_symmetric)

    sol = Ainv * rhs

    npt.assert_allclose(A @ sol, rhs, atol=TOL)