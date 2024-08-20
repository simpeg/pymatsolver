import numpy as np
import scipy.sparse as sp
import pymatsolver

TOL = 1e-12


class TestTriangle:

    @classmethod
    def setup_class(cls):
        n = 50
        nrhs = 20
        cls.A = sp.rand(n, n, 0.4) + sp.identity(n)
        cls.sol = np.ones((n, nrhs))
        cls.rhsU = sp.triu(cls.A) * cls.sol
        cls.rhsL = sp.tril(cls.A) * cls.sol

    def test_directLower(self):
        ALinv = pymatsolver.Forward(sp.tril(self.A))
        X = ALinv * self.rhsL
        x = ALinv * self.rhsL[:, 0]
        assert np.linalg.norm(self.sol-X, np.inf) < TOL
        assert np.linalg.norm(self.sol[:, 0]-x, np.inf) < TOL

    def test_directLower_1(self):
        AUinv = pymatsolver.Backward(sp.triu(self.A))
        X = AUinv * self.rhsU
        x = AUinv * self.rhsU[:, 0]
        assert np.linalg.norm(self.sol-X, np.inf) < TOL
        assert np.linalg.norm(self.sol[:, 0]-x, np.inf) < TOL
