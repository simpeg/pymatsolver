import unittest
import numpy as np
import scipy.sparse as sp
from pymatsolver import Diagonal

TOL = 1e-12


class TestBasic(unittest.TestCase):

    def test_DiagonalSolver(self):

        A = sp.identity(5)*2.0
        rhs = np.c_[np.arange(1, 6), np.arange(2, 11, 2)]
        X = Diagonal(A) * rhs
        x = Diagonal(A) * rhs[:, 0]

        sol = rhs/2.0

        self.assertLess(np.linalg.norm(sol-X, np.inf), TOL)
        self.assertLess(np.linalg.norm(sol[:, 0]-x, np.inf), TOL)


if __name__ == '__main__':
    unittest.main()
