from pymatsolver import BicgJacobi
import numpy as np
import scipy.sparse as sp

TOL = 1e-6


class TestBicgJacobi:

    @classmethod
    def setup_class(cls):

        nSize = 100
        A = sp.rand(nSize, nSize, 0.05, format='csr', random_state=100)
        A = A + sp.spdiags(np.ones(nSize), 0, nSize, nSize)
        A = A.T*A
        A = A.tocsr()
        np.random.seed(1)
        sol = np.random.rand(nSize, 4)
        rhs = A.dot(sol)

        cls.A = A
        cls.rhs = rhs
        cls.sol = sol

    def test(self):
        rhs = self.rhs
        Ainv = BicgJacobi(self.A, symmetric=True)
        solb = Ainv*rhs
        for i in range(3):
            err = np.linalg.norm(
                self.A*solb[:, i] - rhs[:, i]) / np.linalg.norm(rhs[:, i])
            assert err < TOL
        Ainv.clean()

    def test_T(self):
        rhs = self.rhs
        Ainv = BicgJacobi(self.A, symmetric=True)
        Ainv.maxIter = 2000
        AinvT = Ainv.T
        solb = AinvT*rhs
        for i in range(3):
            err = np.linalg.norm(
                self.A.T*solb[:, i] - rhs[:, i]) / np.linalg.norm(rhs[:, i])
            assert err < TOL
        Ainv.clean()


class TestBicgJacobiComplex:

    @classmethod
    def setup_class(cls):
        nSize = 100
        A = sp.rand(nSize, nSize, 0.05, format='csr', random_state=100)
        A.data = A.data + 1j*np.random.rand(A.nnz)
        A = A.T.dot(A) + sp.spdiags(np.ones(nSize), 0, nSize, nSize)
        A = A.tocsr()

        np.random.seed(1)
        sol = np.random.rand(nSize, 5) + 1j*np.random.rand(nSize, 5)
        rhs = A.dot(sol)

        cls.A = A
        cls.rhs = rhs
        cls.sol = sol

    def test(self):
        rhs = self.rhs
        Ainv = BicgJacobi(self.A, symmetric=True)
        solb = Ainv*rhs
        for i in range(3):
            err = np.linalg.norm(
                self.A*solb[:, i] - rhs[:, i]) / np.linalg.norm(rhs[:, i])
            assert err < TOL
        Ainv.clean()

    def test_T(self):
        rhs = self.rhs
        Ainv = BicgJacobi(self.A, symmetric=True)
        Ainv.maxIter = 2000
        AinvT = Ainv.T
        solb = AinvT*rhs
        for i in range(3):
            err = np.linalg.norm(
                self.A.T*solb[:, i] - rhs[:, i]) / np.linalg.norm(rhs[:, i])
            assert err < TOL
        Ainv.clean()
