import unittest
from pymatsolver import MumpsSolver, SolverException
import numpy as np, scipy.sparse as sp

TOL = 1e-12

class TestMumps(unittest.TestCase):

    def setUp(self):
        n = 5
        irn = np.r_[0, 1, 3, 4, 1, 0, 4, 2, 1, 2, 0, 2]
        jcn = np.r_[1, 2, 2, 4, 0, 0, 1, 3, 4, 1, 2, 2]
        a=np.r_[3.0, -3.0, 2.0, 1.0, 3.0, 2.0, 4.0, 2.0, 6.0, -1.0, 4.0, 1.0]
        rhs = np.r_[20.0, 24.0, 9.0, 6.0, 13.0]
        rhs = np.c_[rhs, 10*rhs, 100*rhs]
        sol = np.r_[1.,2.,3.,4.,5.]
        sol = np.c_[sol, 10*sol, 100*sol]
        A = sp.coo_matrix((a, (irn, jcn)), shape=(n,n))
        self.A = A
        self.rhs = rhs
        self.sol = sol

    def test_1to5(self):
        rhs = self.rhs
        sol = self.sol
        Ainv = MumpsSolver(self.A)
        for i in range(3):
            self.assertLess(np.linalg.norm(Ainv * rhs[:,i] - sol[:,i]),TOL)
        self.assertLess(np.linalg.norm(Ainv * rhs - sol, np.inf),TOL)

    def test_1to5_T(self):
        rhs = self.rhs
        sol = self.sol
        Ainv = MumpsSolver(self.A)
        AinvT = Ainv.T
        for i in range(3):
            self.assertLess(np.linalg.norm(AinvT.T * rhs[:,i] - sol[:,i]),TOL)
        self.assertLess(np.linalg.norm(AinvT.T * rhs - sol, np.inf),TOL)

    def test_1to5_solve(self):
        rhs = self.rhs
        sol = self.sol
        Ainv = MumpsSolver(self.A)
        for i in range(3):
            self.assertLess(np.linalg.norm(Ainv.solve( rhs[:,i] ) - sol[:,i]),TOL)
        self.assertLess(np.linalg.norm(Ainv.solve( rhs ) - sol, np.inf),TOL)

    def test_singular(self):
        A = sp.identity(5).tocsr()
        A[-1,-1] = 0
        self.assertRaises(SolverException, MumpsSolver, A)

if __name__ == '__main__':
    unittest.main()
