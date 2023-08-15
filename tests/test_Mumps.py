import warnings

import numpy as np
import pytest
import scipy.sparse as sp

try:
    from pymatsolver import Mumps
    should_run = True
except ImportError:
    should_run = False

TOL = 1e-11


if should_run:

    class TestMumps:

        @classmethod
        def setup_class(cls):
            n = 5
            irn = np.r_[0, 1, 3, 4, 1, 0, 4, 2, 1, 2, 0, 2]
            jcn = np.r_[1, 2, 2, 4, 0, 0, 1, 3, 4, 1, 2, 2]
            a = np.r_[
                3.0, -3.0, 2.0, 1.0, 3.0, 2.0,
                4.0, 2.0, 6.0, -1.0, 4.0, 1.0
            ]
            rhs = np.r_[20.0, 24.0, 9.0, 6.0, 13.0]
            rhs = np.c_[rhs, 10*rhs, 100*rhs]
            sol = np.r_[1., 2., 3., 4., 5.]
            sol = np.c_[sol, 10*sol, 100*sol]
            A = sp.coo_matrix((a, (irn, jcn)), shape=(n, n))
            cls.A = A
            cls.rhs = rhs
            cls.sol = sol

        def test_1to5(self):
            rhs = self.rhs
            sol = self.sol
            Ainv = Mumps(self.A)
            for i in range(3):
                assert np.linalg.norm(Ainv * rhs[:, i] - sol[:, i]) < TOL
            assert np.linalg.norm(Ainv * rhs - sol, np.inf) < TOL

        def test_1to5_cmplx(self):
            rhs = self.rhs.astype(complex)
            sol = self.sol.astype(complex)
            self.A = self.A.astype(complex)
            Ainv = Mumps(self.A)
            for i in range(3):
                assert np.linalg.norm(Ainv * rhs[:, i] - sol[:, i]) < TOL
            assert np.linalg.norm(Ainv * rhs - sol, np.inf) < TOL

        def test_1to5_T(self):
            rhs = self.rhs
            sol = self.sol
            Ainv = Mumps(self.A)
            AinvT = Ainv.T
            for i in range(3):
                assert np.linalg.norm(AinvT.T * rhs[:, i] - sol[:, i]) < TOL
            assert np.linalg.norm(AinvT.T * rhs - sol, np.inf) < TOL

        def test_singular(self):
            A = sp.identity(5).tocsr()
            A[-1, -1] = 0
            with pytest.raises(Exception):
                Mumps(A)

        def test_multiFactorsInMem(self):
            n = 100
            A = sp.rand(n, n, 0.7)+sp.identity(n)
            x = np.ones((n, 10))
            rhs = A * x
            solvers = [Mumps(A) for _ in range(20)]

            for Ainv in solvers:
                assert np.linalg.norm(Ainv * rhs - x)/np.linalg.norm(rhs) < TOL
                Ainv.clean()

            for Ainv in solvers:
                assert np.linalg.norm(Ainv * rhs - x)/np.linalg.norm(rhs) < TOL

else:

    def test_mumps_not_available():
        """Run only when Mumps is not available."""
        warnings.warn("NOTE: Mumps is not available so we're skipping its tests.")
