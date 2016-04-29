import unittest
import numpy as np
import scipy.sparse as sp
from pyMKL import pardisoSolver

class TestPardiso(unittest.TestCase):

    def test_RealNonSym_oneRHS(self):
        nSize = 100
        A = sp.rand(nSize, nSize, 0.05, format='csr', random_state=10)
        A = A + sp.spdiags(np.ones(nSize), 0, nSize, nSize)
        A = A.tocsr()

        np.random.seed(1)
        xTrue = np.random.rand(nSize)
        rhs = A.dot(xTrue)

        pSolve = pardisoSolver(A, mtype=11)

        pSolve.run_pardiso(12)
        x = pSolve.run_pardiso(33, rhs)
        self.assertLess(np.linalg.norm(x-xTrue)/np.linalg.norm(xTrue), 1e-12)


if __name__ == '__main__':
    unittest.main()
