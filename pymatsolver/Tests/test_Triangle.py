import unittest
import numpy as np
import scipy.sparse as sp
import pymatsolver

TOL = 1e-12


class TestTriangle(unittest.TestCase):

    def setUp(self):
        n = 50
        nrhs = 20
        self.A = sp.rand(n, n, 0.4) + sp.identity(n)
        self.sol = np.ones((n, nrhs))
        self.rhsU = sp.triu(self.A) * self.sol
        self.rhsL = sp.tril(self.A) * self.sol

    def test_directLower(self):
        from pymatsolver import Forward
        ALinv = Forward(sp.tril(self.A))
        X = ALinv * self.rhsL
        x = ALinv * self.rhsL[:, 0]
        self.assertLess(np.linalg.norm(self.sol-X, np.inf), TOL)
        self.assertLess(np.linalg.norm(self.sol[:, 0]-x, np.inf), TOL)

    def test_directLower_1(self):
        from pymatsolver import Backward
        AUinv = Backward(sp.triu(self.A))
        X = AUinv * self.rhsU
        x = AUinv * self.rhsU[:, 0]
        self.assertLess(np.linalg.norm(self.sol-X, np.inf), TOL)
        self.assertLess(np.linalg.norm(self.sol[:, 0]-x, np.inf), TOL)

    if pymatsolver.AvailableSolvers['TriangleFortran']:

        def test_directLower_python(self):
            from pymatsolver import _Forward
            ALinv = _Forward(sp.tril(self.A))
            X = ALinv * self.rhsL
            x = ALinv * self.rhsL[:, 0]
            self.assertLess(np.linalg.norm(self.sol-X, np.inf), TOL)
            self.assertLess(np.linalg.norm(self.sol[:, 0]-x, np.inf), TOL)

        def test_directLower_1_python(self):
            from pymatsolver import _Backward
            AUinv = _Backward(sp.triu(self.A))
            X = AUinv * self.rhsU
            x = AUinv * self.rhsU[:, 0]
            self.assertLess(np.linalg.norm(self.sol-X, np.inf), TOL)
            self.assertLess(np.linalg.norm(self.sol[:, 0]-x, np.inf), TOL)

if __name__ == '__main__':
    unittest.main()
