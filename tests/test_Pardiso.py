import unittest
from pymatsolver import Pardiso
from pydiso.mkl_solver import (
    get_mkl_pardiso_max_threads,
    PardisoTypeConversionWarning
)
import numpy as np
import scipy.sparse as sp
import os

TOL = 1e-10


class TestPardiso(unittest.TestCase):

    def setUp(self):

        nSize = 100
        A = sp.rand(nSize, nSize, 0.05, format='csr', random_state=100)
        A = A + sp.spdiags(np.ones(nSize), 0, nSize, nSize)
        A = A.T*A
        A = A.tocsr()
        np.random.seed(1)
        sol = np.random.rand(nSize, 5)
        rhs = A.dot(sol)

        self.A = A
        self.rhs = rhs
        self.sol = sol

    def test(self):
        rhs = self.rhs
        sol = self.sol
        Ainv = Pardiso(self.A, is_symmetric=True)
        for i in range(3):
            self.assertLess(np.linalg.norm(Ainv * rhs[:, i] - sol[:, i]), TOL)
        self.assertLess(np.linalg.norm(Ainv * rhs - sol, np.inf), TOL)

    def test_refactor(self):
        rhs = self.rhs
        sol = self.sol
        A = self.A
        Ainv = Pardiso(A, is_symmetric=True)
        for i in range(3):
            self.assertLess(np.linalg.norm(Ainv * rhs[:, i] - sol[:, i]), TOL)
        self.assertLess(np.linalg.norm(Ainv * rhs - sol, np.inf), TOL)

        # scale rows and collumns
        D = sp.diags(np.random.rand(A.shape[0]) + 1.0)
        A2 = D.T @ A @ D

        rhs2 = A2 @ sol
        Ainv.factor(A2)
        for i in range(3):
            self.assertLess(np.linalg.norm(Ainv * rhs2[:, i] - sol[:, i]), TOL)
        self.assertLess(np.linalg.norm(Ainv * rhs2 - sol, np.inf), TOL)

    def test_T(self):
        rhs = self.rhs
        sol = self.sol
        Ainv = Pardiso(self.A, is_symmetric=True)

        with self.assertWarns(PardisoTypeConversionWarning):
            AinvT = Ainv.T
            x = AinvT * rhs

            for i in range(3):
                self.assertLess(np.linalg.norm(x[:, i] - sol[:, i]), TOL)
            self.assertLess(np.linalg.norm(x - sol, np.inf), TOL)

    def test_n_threads(self):
        max_threads = get_mkl_pardiso_max_threads()
        print(f'testing setting n_threads to 1 and {max_threads}')
        Ainv = Pardiso(self.A, is_symmetric=True, n_threads=1)
        self.assertEqual(Ainv.n_threads, 1)

        Ainv2 = Pardiso(self.A, is_symmetric=True, n_threads=max_threads)
        self.assertEqual(Ainv2.n_threads, max_threads)
        self.assertEqual(Ainv2.n_threads, Ainv.n_threads)

        Ainv.n_threads = 1
        self.assertEqual(Ainv.n_threads, 1)
        self.assertEqual(Ainv2.n_threads, Ainv.n_threads)

        with self.assertRaises(TypeError):
            Ainv.n_threads = "2"


class TestPardisoNotSymmetric(unittest.TestCase):

    def setUp(self):

        nSize = 100
        A = sp.rand(nSize, nSize, 0.05, format='csr', random_state=100)
        A = A + sp.spdiags(np.ones(nSize), 0, nSize, nSize)
        A = A.tocsr()
        np.random.seed(1)
        sol = np.random.rand(nSize, 5)
        rhs = A.dot(sol)

        self.A = A
        self.rhs = rhs
        self.sol = sol

    def test(self):
        rhs = self.rhs
        sol = self.sol
        Ainv = Pardiso(self.A, is_symmetric=True, check_accuracy=True)
        self.assertRaises(Exception, lambda: Ainv * rhs)
        Ainv.clean()

        Ainv = Pardiso(self.A)
        for i in range(3):
            self.assertLess(np.linalg.norm(Ainv * rhs[:, i] - sol[:, i]), TOL)
        self.assertLess(np.linalg.norm(Ainv * rhs - sol, np.inf), TOL)
        Ainv.clean()


class TestPardisoFDEM(unittest.TestCase):

    def setUp(self):

        base_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'fdem')

        data = np.load(os.path.join(base_path, 'A_data.npy'))
        indices = np.load(os.path.join(base_path, 'A_indices.npy'))
        indptr = np.load(os.path.join(base_path, 'A_indptr.npy'))

        self.A = sp.csr_matrix((data, indices, indptr), shape=(13872, 13872))
        self.rhs = np.load(os.path.join(base_path, 'RHS.npy'))

    def test(self):
        rhs = self.rhs
        Ainv = Pardiso(self.A, check_accuracy=True)
        sol = Ainv * rhs
        with self.assertWarns(PardisoTypeConversionWarning):
            sol = Ainv * rhs.real


class TestPardisoComplex(unittest.TestCase):

    def setUp(self):
        nSize = 100
        A = sp.rand(nSize, nSize, 0.05, format='csr', random_state=100)
        A.data = A.data + 1j*np.random.rand(A.nnz)
        A = A.T.dot(A) + sp.spdiags(np.ones(nSize), 0, nSize, nSize)
        A = A.tocsr()

        np.random.seed(1)
        sol = np.random.rand(nSize, 5) + 1j*np.random.rand(nSize, 5)
        rhs = A.dot(sol)

        self.A = A
        self.rhs = rhs
        self.sol = sol

    def test(self):
        rhs = self.rhs
        sol = self.sol
        Ainv = Pardiso(self.A, is_symmetric=True)
        for i in range(3):
            self.assertLess(np.linalg.norm(Ainv * rhs[:, i] - sol[:, i]), TOL)
        self.assertLess(np.linalg.norm(Ainv * rhs - sol, np.inf), TOL)
        Ainv.clean()

    def test_T(self):
        rhs = self.rhs
        sol = self.sol
        Ainv = Pardiso(self.A, is_symmetric=True)
        with self.assertWarns(PardisoTypeConversionWarning):
            AinvT = Ainv.T
            x = AinvT * rhs
            for i in range(3):
                self.assertLess(
                    np.linalg.norm(x[:, i] - sol[:, i]), TOL
                )
            self.assertLess(np.linalg.norm(x - sol, np.inf), TOL)

if __name__ == '__main__':
    unittest.main()
