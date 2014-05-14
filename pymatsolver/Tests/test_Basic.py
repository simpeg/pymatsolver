import unittest
import numpy as np, scipy.sparse as sp

TOL = 1e-12

class TestBasic(unittest.TestCase):

    def test_DiagonalSolver(self):
        from pymatsolver import DiagonalSolver

        A = sp.identity(5)*2.0
        rhs = np.c_[np.arange(1,6),np.arange(2,11,2)]
        X = DiagonalSolver(A) * rhs
        x = DiagonalSolver(A) * rhs[:,0]

        sol = rhs/2.0

        self.assertLess(np.linalg.norm(sol-X,np.inf), TOL)
        self.assertLess(np.linalg.norm(sol[:,0]-x,np.inf), TOL)


if __name__ == '__main__':
    unittest.main()
