import unittest
import numpy as np
import scipy.sparse as sp
from pyMKL import pardisoSolver

nSize = 100

class TestPardiso_oneRHS(unittest.TestCase):

    def test_RealNonSym(self):
        nSize = 100
        A = sp.rand(nSize, nSize, 0.05, format='csr', random_state=100)
        A = A + sp.spdiags(np.ones(nSize), 0, nSize, nSize)
        A = A.tocsr()

        np.random.seed(1)
        xTrue = np.random.rand(nSize)
        rhs = A.dot(xTrue)

        pSolve = pardisoSolver(A, mtype=11)

        pSolve.run_pardiso(12)
        x = pSolve.run_pardiso(33, rhs)
        self.assertLess(np.linalg.norm(x-xTrue)/np.linalg.norm(xTrue), 1e-12)

    def test_RealSPD(self):

        A = sp.rand(nSize, nSize, 0.05, format='csr', random_state=100)
        A = A.T.dot(A) + sp.spdiags(np.ones(nSize), 0, nSize, nSize)
        A = A.tocsr()

        np.random.seed(1)
        xTrue = np.random.rand(nSize)
        rhs = A.dot(xTrue)

        pSolve = pardisoSolver(A, mtype=2)

        pSolve.run_pardiso(12)
        x = pSolve.run_pardiso(33, rhs)

        self.assertLess(np.linalg.norm(x-xTrue)/np.linalg.norm(xTrue), 1e-12)

    def test_RealPosInd(self):

        A = sp.rand(nSize, nSize, 0.05, format='csr', random_state=100)
        d = np.ones(nSize)
        d[nSize/2:] = -1.
        A = A.T.dot(A) + sp.spdiags(d, 0, nSize, nSize)    
        A = A.tocsr()

        np.random.seed(1)
        xTrue = np.random.rand(nSize)
        rhs = A.dot(xTrue)

        pSolve = pardisoSolver(A, mtype=-2)

        pSolve.run_pardiso(12)
        x = pSolve.run_pardiso(33, rhs)
        self.assertLess(np.linalg.norm(x-xTrue)/np.linalg.norm(xTrue), 1e-12)

    def test_ComplexNonSym(self):
        nSize = 100
        A = sp.rand(nSize, nSize, 0.05, format='csr', random_state=100)
        A = A + sp.spdiags(np.ones(nSize), 0, nSize, nSize)
        A.data = A.data + 1j*np.random.rand(A.nnz)
        A = A.tocsr()

        np.random.seed(1)
        xTrue = np.random.rand(nSize) + 1j*np.random.rand(nSize)
        rhs = A.dot(xTrue)

        pSolve = pardisoSolver(A, mtype=13)
        pSolve.run_pardiso(12)
        x = pSolve.run_pardiso(33, rhs)
        self.assertLess(np.linalg.norm(x-xTrue)/np.linalg.norm(xTrue), 1e-12)

    def test_ComplexNonSym_RealRHS(self):
        nSize = 100
        A = sp.rand(nSize, nSize, 0.05, format='csr', random_state=100)
        A = A + sp.spdiags(np.ones(nSize), 0, nSize, nSize)
        A.data = A.data + 1j*np.random.rand(A.nnz)
        A = A.tocsr()

        np.random.seed(1)
        rhs = np.random.rand(nSize)

        pSolve = pardisoSolver(A, mtype=13)
        pSolve.run_pardiso(12)
        x = pSolve.run_pardiso(33, rhs)
        self.assertLess(np.linalg.norm(A.dot(x)-rhs), 1e-12)

    def test_ComplexSym(self):
        nSize = 100
        A = sp.rand(nSize, nSize, 0.05, format='csr', random_state=100)
        A.data = A.data + 1j*np.random.rand(A.nnz)
        A = A.T.dot(A) + sp.spdiags(np.ones(nSize), 0, nSize, nSize)
        A = A.tocsr()

        np.random.seed(1)
        xTrue = np.random.rand(nSize) + 1j*np.random.rand(nSize)
        rhs = A.dot(xTrue)

        pSolve = pardisoSolver(A, mtype=6)
        pSolve.run_pardiso(12)
        x = pSolve.run_pardiso(33, rhs)
        self.assertLess(np.linalg.norm(x-xTrue)/np.linalg.norm(xTrue), 1e-12)

    def test_ComplexHerm(self):
        nSize = 100
        A = sp.rand(nSize, nSize, 0.05, format='csr', random_state=100)
        A.data = A.data + 1j*np.random.rand(A.nnz)
        A = A.T.dot(A.conj()) + sp.spdiags(np.ones(nSize), 0, nSize, nSize)
        A = A.tocsr()

        np.random.seed(1)
        xTrue = np.random.rand(nSize) + 1j*np.random.rand(nSize)
        rhs = A.dot(xTrue)

        pSolve = pardisoSolver(A, mtype=4)
        pSolve.run_pardiso(12)
        x = pSolve.run_pardiso(33, rhs)
        self.assertLess(np.linalg.norm(x-xTrue)/np.linalg.norm(xTrue), 1e-12)



if __name__ == '__main__':
    unittest.main()
