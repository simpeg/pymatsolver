import os
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

            nSize = 100
            A = sp.rand(nSize, nSize, 0.05, format='csr', random_state=100)
            A = A + sp.spdiags(np.ones(nSize), 0, nSize, nSize)
            A = A.T*A
            A = A.tocsr()
            np.random.seed(1)
            sol = np.random.rand(nSize, 5)
            rhs = A.dot(sol)

            cls.A = A
            cls.rhs = rhs
            cls.sol = sol

        def test(self):
            rhs = self.rhs
            sol = self.sol
            Ainv = Mumps(self.A, is_symmetric=True)
            for i in range(3):
                assert np.linalg.norm(Ainv * rhs[:, i] - sol[:, i]) < TOL
            assert np.linalg.norm(Ainv * rhs - sol, np.inf) < TOL

        def test_T(self):
            rhs = self.rhs
            sol = self.sol
            Ainv = Mumps(self.A, is_symmetric=True)
            AinvT = Ainv.T
            x = AinvT * rhs

            for i in range(3):
                assert np.linalg.norm(x[:, i] - sol[:, i]) < TOL
            assert np.linalg.norm(x - sol, np.inf) < TOL

    class TestMumpsNotSymmetric:

        @classmethod
        def setup_class(cls):

            nSize = 100
            A = sp.rand(nSize, nSize, 0.05, format='csr', random_state=100)
            A = A + sp.spdiags(np.ones(nSize), 0, nSize, nSize)
            A = A.tocsr()
            np.random.seed(1)
            sol = np.random.rand(nSize, 5)
            rhs = A.dot(sol)

            cls.A = A
            cls.rhs = rhs
            cls.sol = sol

        def test(self):
            rhs = self.rhs
            sol = self.sol
            Ainv = Mumps(self.A, is_symmetric=True, check_accuracy=True)
            with pytest.raises(Exception):
                Ainv * rhs
            Ainv.clean()

            Ainv = Mumps(self.A, check_accuracy=True)
            for i in range(3):
                assert np.linalg.norm(Ainv * rhs[:, i] - sol[:, i]) < TOL
            assert np.linalg.norm(Ainv * rhs - sol, np.inf) < TOL
            Ainv.clean()


    class TestMumpsFDEM:

        @classmethod
        def setup_class(cls):

            base_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'fdem')

            data = np.load(os.path.join(base_path, 'A_data.npy'))
            indices = np.load(os.path.join(base_path, 'A_indices.npy'))
            indptr = np.load(os.path.join(base_path, 'A_indptr.npy'))

            cls.A = sp.csr_matrix((data, indices, indptr), shape=(13872, 13872))
            cls.rhs = np.load(os.path.join(base_path, 'RHS.npy'))

        def test(self):
            rhs = self.rhs
            Ainv = Mumps(self.A, check_accuracy=True)
            sol = Ainv * rhs
            sol = Ainv * rhs.real


    class TestMumpsComplex:

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
            sol = self.sol
            Ainv = Mumps(self.A, is_symmetric=True)
            for i in range(3):
                assert np.linalg.norm(Ainv * rhs[:, i] - sol[:, i]) < TOL
            assert np.linalg.norm(Ainv * rhs - sol, np.inf) < TOL
            Ainv.clean()

        def test_T(self):
            rhs = self.rhs
            sol = self.sol
            Ainv = Mumps(self.A, is_symmetric=True)
            AinvT = Ainv.T
            x = AinvT * rhs
            for i in range(3):
                assert np.linalg.norm(x[:, i] - sol[:, i]) < TOL
            assert np.linalg.norm(x - sol, np.inf) < TOL


    class TestMumps1to5:

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

        def test(self):
            rhs = self.rhs
            sol = self.sol
            Ainv = Mumps(self.A)
            for i in range(3):
                assert np.linalg.norm(Ainv * rhs[:, i] - sol[:, i]) < TOL
            assert np.linalg.norm(Ainv * rhs - sol, np.inf) < TOL

        def test_cmplx(self):
            rhs = self.rhs.astype(complex)
            sol = self.sol.astype(complex)
            self.A = self.A.astype(complex)
            Ainv = Mumps(self.A)
            for i in range(3):
                assert np.linalg.norm(Ainv * rhs[:, i] - sol[:, i]) < TOL
            assert np.linalg.norm(Ainv * rhs - sol, np.inf) < TOL

        def test_T(self):
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

        def test_multi_factors_in_mem(self):
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
