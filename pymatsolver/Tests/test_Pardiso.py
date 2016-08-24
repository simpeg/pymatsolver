import unittest
from pymatsolver import PardisoSolver
import numpy as np
import scipy.sparse as sp

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
        Ainv = PardisoSolver(self.A, symmetric=True)
        for i in range(3):
            self.assertLess(np.linalg.norm(Ainv * rhs[:, i] - sol[:, i]), TOL)
        self.assertLess(np.linalg.norm(Ainv * rhs - sol, np.inf), TOL)
        Ainv.clean()

    def test_T(self):
        rhs = self.rhs
        sol = self.sol
        Ainv = PardisoSolver(self.A, symmetric=True)
        AinvT = Ainv.T
        for i in range(3):
            self.assertLess(
                np.linalg.norm(AinvT.T * rhs[:, i] - sol[:, i]), TOL)
        self.assertLess(np.linalg.norm(AinvT.T * rhs - sol, np.inf), TOL)
        Ainv.clean()


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
        Ainv = PardisoSolver(self.A, symmetric=True)
        for i in range(3):
            self.assertLess(np.linalg.norm(Ainv * rhs[:, i] - sol[:, i]), TOL)
        self.assertLess(np.linalg.norm(Ainv * rhs - sol, np.inf), TOL)
        Ainv.clean()

    def test_T(self):
        rhs = self.rhs
        sol = self.sol
        Ainv = PardisoSolver(self.A, symmetric=True)
        AinvT = Ainv.T
        for i in range(3):
            self.assertLess(
                np.linalg.norm(AinvT.T * rhs[:, i] - sol[:, i]), TOL)
        self.assertLess(np.linalg.norm(AinvT.T * rhs - sol, np.inf), TOL)


if __name__ == '__main__':
    unittest.main()
